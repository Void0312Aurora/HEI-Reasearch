import torch
import torch.nn as nn
from typing import List, Dict
from he_core.state import ContactState
from he_core.contact_dynamics import ContactIntegrator
from he_core.generator import BaseGenerator

class TransitionMap(nn.Module):
    """
    Approximates Parallel Transport / Transition Map between Chart i and j.
    phi_{ij}: M_i -> M_j.
    """
    def __init__(self, dim_q: int):
        super().__init__()
        # Simple Linear Map + Bias (Affine) for local tangent space approx
        self.linear = nn.Linear(2*dim_q + 1, 2*dim_q + 1)
        
    def forward(self, state_i: ContactState) -> torch.Tensor:
        # Map flat state -> flat state
        return self.linear(state_i.flat)

class AtlasRouter(nn.Module):
    """
    Gating Network: Selects active charts based on global 'Context' (or q).
    """
    def __init__(self, dim_q: int, num_charts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_q, 16),
            nn.ReLU(),
            nn.Linear(16, num_charts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Returns weights (B, num_charts)"""
        return self.net(q)

class Atlas(nn.Module):
    """
    Manages K Charts with Router.
    """
    def __init__(self, num_charts: int, dim_q: int):
        super().__init__()
        self.num_charts = num_charts
        self.dim_q = dim_q
        self.states = []
        for _ in range(num_charts):
            self.states.append(ContactState(dim_q))
            
        # Transitions: Dictionarty (i, j) -> Module
        self.transitions = nn.ModuleDict()
        
        # Integrator
        self.integrator = ContactIntegrator()
        
        # Router
        self.router = AtlasRouter(dim_q, num_charts)
        
    def get_active_weights(self, q_context: torch.Tensor) -> torch.Tensor:
        """Wrapper for router"""
        return self.router(q_context)
        
    def add_transition(self, i: int, j: int):
        key = f"{i}_{j}"
        self.transitions[key] = TransitionMap(self.dim_q)
        
    def step(self, generators: List[BaseGenerator], active_mask: torch.Tensor = None, dt: float = 0.1):
        """
        Step all active charts.
        generators: List of generators, one per chart (or shared).
        """
        if isinstance(generators, list):
            assert len(generators) == self.num_charts
        else:
            generators = [generators] * self.num_charts
            
        for k in range(self.num_charts):
            # Step dynamics
            # TODO: Only step active? For now step all.
            self.states[k] = self.integrator.step(self.states[k], generators[k], dt)
            
    def compute_consistency_loss(self, i: int, j: int) -> torch.Tensor:
        """
        Computes ||phi_{ij}(x_i) - x_j||^2
        Only valid if i and j are overlapping/active.
        """
        key = f"{i}_{j}"
        if key not in self.transitions:
            return torch.tensor(0.0)
            
        map_ij = self.transitions[key]
        
        pred_j_flat = map_ij(self.states[i])
        real_j_flat = self.states[j].flat
        
        loss = (pred_j_flat - real_j_flat).pow(2).mean()
        return loss
        
    def sync_overlap(self, i: int, j: int, alpha=0.1):
        """
        Soft sync: Pull x_j towards phi_{ij}(x_i).
        Simple consistency enforcement.
        """
        key = f"{i}_{j}"
        if key not in self.transitions:
            return
            
        map_ij = self.transitions[key]
        pred_j_flat = map_ij(self.states[i])
        
        # Nudge x_j
        # detach pred_j to stop gradients flowing back to i?
        # A3/Transition Consistency says we optimize the Map or the State?
        # Usually we optimize State to be consistent.
        
        current_j = self.states[j].flat
        # target = pred_j_flat
        # new_j = (1-alpha)*current + alpha*target
        
        new_j_flat = (1.0 - alpha) * current_j + alpha * pred_j_flat.detach()
        
        # Update state j
        self.states[j] = ContactState(self.dim_q, self.states[j].batch_size, self.states[j].device, new_j_flat)

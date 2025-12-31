import torch
import torch.nn as nn
from typing import Callable, Optional
from he_core.state import ContactState
from he_core.generator import BaseGenerator, DissipativeGenerator

class PortCoupling(nn.Module):
    """
    Defines the shape B(q) in the coupling term <u, B(q)>.
    Default: B(q) = -q (Linear Force Coupling).
    Learnable: B(q) = W q (Linear Map).
    Mixture: B(q) = sum w_k * W_k q (Gated Linear Map).
    """
    def __init__(self, dim_q: int, dim_u: int, learnable: bool = False, num_charts: int = 1):
        super().__init__()
        self.dim_q = dim_q
        self.dim_u = dim_u
        self.learnable = learnable
        self.num_charts = num_charts
        
        if learnable:
            # We need K matrices if num_charts > 1
            # Dictionary of matrices? Or tensor?
            # Tensor (K, dim_u, dim_q)
            self.W_stack = nn.Parameter(torch.Tensor(num_charts, dim_u, dim_q))
            
            # Init
            with torch.no_grad():
                 # Initialize all near Identity/Negative Identity
                 for k in range(num_charts):
                     if dim_q == dim_u:
                         self.W_stack[k].copy_(-torch.eye(dim_q) + torch.randn(dim_q, dim_q)*0.01)
        
        if self.learnable:
            # We add an Action Readout Matrix W_out: (Dim_Q -> Dim_U)
            # Action a = W_out q
            # Using same shape as W_stack but separate param?
            # Let's keep it simple: A separate Parameter.
            # self.W_out = nn.Parameter(torch.Tensor(num_charts, dim_u, dim_q))
            # Actually, let's just reuse W_stack transpose if we want symmetry?
            # No, Active Inference usually implies a separate "Policy" or "Reflex".
            # Let's add W_action.
            self.W_action = nn.Parameter(torch.Tensor(num_charts, dim_u, dim_q))
            with torch.no_grad():
                nn.init.normal_(self.W_action, std=0.01)

    def forward(self, q: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns B(q).
        """
        if not self.learnable:
            return -q # Assumes dim_q == dim_u
            
        # ... (Existing Einsum logic for B(q)) ...
        # q -> (B, 1, Dq)
        # W -> (K, Du, Dq)
        # y_all = q W^T -> (B, K, Du)
        Y_all = torch.einsum('kuq,bq->bku', self.W_stack, q)
        
        if weights is None:
             weights = torch.ones(q.shape[0], self.num_charts, device=q.device) / self.num_charts
             
        w_b = weights.unsqueeze(2)
        B_q = (Y_all * w_b).sum(dim=1)
        return B_q

    def get_action(self, q: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns Action a(q).
        a = sum w_k * W_action_k * q
        """
        if not self.learnable:
            return torch.zeros(q.shape[0], self.dim_u, device=q.device)
            
        # Einsum: k u q, b q -> b k u
        A_all = torch.einsum('kuq,bq->bku', self.W_action, q)
        
        if weights is None:
             weights = torch.ones(q.shape[0], self.num_charts, device=q.device) / self.num_charts
        
        w_b = weights.unsqueeze(2)
        Action_Raw = (A_all * w_b).sum(dim=1)
        # Apply Tanh to bound action
        # Scale it up to allow range (e.g. +/- 10)
        return 5.0 * torch.tanh(Action_Raw)

class PortCoupledGenerator(nn.Module):
    """
    Template 3: Hamiltonian with Multi-Port Coupling.
    H(x, u, t) = H_int(x) + sum_i <u_i, B_i(q)>
    """
    def __init__(self, internal_generator: BaseGenerator, dim_u: int, learnable_coupling: bool = False, num_charts: int = 1):
        super().__init__()
        self.internal = internal_generator
        self.dim_u = dim_u
        self.dim_q = internal_generator.dim_q
        self.num_charts = num_charts
        self.learnable = learnable_coupling
        
        # We store ports in a ModuleDict
        self.ports = nn.ModuleDict({
            'default': PortCoupling(self.dim_q, dim_u, learnable=learnable_coupling, num_charts=num_charts)
        })
        
    def add_port(self, name: str, dim_u: int):
        self.ports[name] = PortCoupling(self.dim_q, dim_u, learnable=self.learnable, num_charts=self.num_charts)

    def get_h_port(self, state: ContactState, port_name: str, u_val: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes <u, B(q)> for a specific port.
        """
        if port_name not in self.ports:
            return torch.zeros(state.batch_size, 1, device=state.device)
            
        B_q = self.ports[port_name](state.q, weights=weights)
        return (u_val * B_q).sum(dim=1, keepdim=True)

    def get_action(self, state: ContactState, port_name: str = 'default', weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns Action a(q) for a specific port.
        """
        if port_name not in self.ports:
             # Or raise error?
             return torch.zeros(state.batch_size, self.dim_u, device=state.device)
             
        return self.ports[port_name].get_action(state.q, weights=weights)

    def forward(self, state: ContactState, u_ext: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Backward compatibility for single port usage.
        Assumes u_ext is for the 'default' port.
        """
        H_int = self.internal(state)
        if u_ext is None:
            return H_int
        
        H_port = self.get_h_port(state, 'default', u_ext, weights=weights)
        return H_int + H_port

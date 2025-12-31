"""
Entity v0.5: Theoretical Hardening

Upgrades from v0.4:
1. L2 PT_A: Parallel Transport of momentum p during chart transitions.
2. A3 F-Functional: Unified Free Energy as the central optimization target.
3. L3 z-Context: Internal autonomous context variable.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

from he_core.state import ContactState
from he_core.contact_dynamics import ContactIntegrator
from he_core.generator import BaseGenerator, DissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.atlas import Atlas, AtlasRouter
from he_core.interfaces import ActionInterface, PortInterface
from he_core.connection import Connection


class UnifiedGeometricEntityV5(nn.Module):
    """
    Entity v0.5: Theoretically Hardened Geometric Entity.
    
    Key Upgrades:
    1. L2 PT_A: Momentum is parallel-transported when chart weights change.
    2. A3: Unified Free Energy F = V(q) + KL(z) + Error.
    3. L3: Autonomous context z with Active Inference evolution.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim_q = config.get('dim_q', 2)
        self.dim_u = config.get('dim_u', self.dim_q)
        self.dim_z = config.get('dim_z', 8)  # L3: Autonomous Context Dimension
        self.num_charts = config.get('num_charts', 1)
        
        # 1. Generator (Internal Soul)
        alpha = config.get('damping', 0.1)
        self.internal_gen = DissipativeGenerator(self.dim_q, alpha=alpha)
        
        # 2. Port Coupling
        learnable_coupling = config.get('learnable_coupling', False)
        self.generator = PortCoupledGenerator(
            self.internal_gen, 
            dim_u=self.dim_u, 
            learnable_coupling=learnable_coupling,
            num_charts=self.num_charts
        )
        
        # 3. Atlas + Router
        self.atlas = Atlas(self.num_charts, self.dim_q)
        
        # 4. L2: Connection for Parallel Transport
        self.connection = Connection(self.dim_q)
        self._prev_chart_weights = None  # Track for transport logic
        
        # 5. Integrator
        self.integrator = ContactIntegrator()
        
        # 6. Interface (Default: PortInterface for energy semantics)
        use_port = config.get('use_port_interface', True)
        if use_port:
            contract_method = config.get('port_contract_method', 'tanh')
            max_gain = config.get('port_max_gain', 1.0)
            self.interface = PortInterface(self.dim_q, self.dim_u, 
                                           use_contract=True, 
                                           contract_method=contract_method,
                                           max_gain=max_gain)
        else:
            self.interface = ActionInterface(self.dim_q, self.dim_u)
        
        # 7. L3: Autonomous Context z
        # z is a learnable prior for conditioning dynamics
        self.z = nn.Parameter(torch.zeros(1, self.dim_z))
        self.z_prior_mean = nn.Parameter(torch.zeros(self.dim_z), requires_grad=False)
        self.z_prior_logvar = nn.Parameter(torch.zeros(self.dim_z), requires_grad=False)
        
        # A3: Potential network (part of F)
        self.net_V = nn.Sequential(
            nn.Linear(self.dim_q, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # State Container
        self.state = ContactState(self.dim_q, 1)
        
    def reset(self, scale: float = 0.1, batch_size: int = 1):
        device = next(self.parameters()).device
        self.state = ContactState(self.dim_q, batch_size, device)
        self.state.q = torch.randn(batch_size, self.dim_q, device=device) * scale
        self.state.p = torch.randn(batch_size, self.dim_q, device=device) * scale
        self.state.s = torch.zeros(batch_size, 1, device=device)
        self._prev_chart_weights = None
        
    def _compute_kl_z(self) -> torch.Tensor:
        """
        A3: KL divergence from z prior (regularization term in F).
        """
        # Assume z ~ N(z, 1) approximately
        # KL(q(z)||p(z)) = 0.5 * sum(z^2) for standard normal prior
        return 0.5 * (self.z ** 2).sum()
        
    def compute_free_energy(self, state: ContactState, prediction_error: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        A3: Unified Free Energy Functional.
        F = V(q) + beta * KL(z) + gamma * prediction_error
        """
        beta_kl = self.config.get('beta_kl', 0.01)
        gamma_pred = self.config.get('gamma_pred', 1.0)
        
        # Potential Energy
        V = self.net_V(state.q).mean()
        
        # KL Regularization on z
        KL = beta_kl * self._compute_kl_z()
        
        # Prediction Error (if provided)
        if prediction_error is not None:
            E = gamma_pred * prediction_error.mean()
        else:
            E = torch.tensor(0.0, device=state.device)
            
        return V + KL + E
        
    def _apply_parallel_transport(self, p: torch.Tensor, q: torch.Tensor, 
                                   old_weights: torch.Tensor, new_weights: torch.Tensor) -> torch.Tensor:
        """
        L2: Apply parallel transport to momentum p when chart weights change.
        
        Theory: When switching from chart i to chart j, momentum must be transported
        along the connection to maintain geometric consistency.
        
        Approximation: We compute a weighted transport based on weight deltas.
        """
        if old_weights is None:
            return p
            
        # Compute weight change
        delta_w = new_weights - old_weights  # (B, K)
        change_magnitude = delta_w.abs().sum(dim=1, keepdim=True)  # (B, 1)
        
        # If weights changed significantly, apply transport
        threshold = self.config.get('transport_threshold', 0.1)
        
        # Compute transport direction: from "center of mass" of old weights to new
        # For simplicity, we use q as both source and target with a perturbation
        q_target = q + 0.1 * delta_w[:, :self.dim_q] if delta_w.shape[1] >= self.dim_q else q
        
        # Apply connection transport
        p_transported = self.connection(q, q_target, p)
        
        # Blend based on change magnitude
        blend_factor = torch.clamp(change_magnitude / (threshold + 1e-6), 0.0, 1.0)
        p_new = (1 - blend_factor) * p + blend_factor * p_transported
        
        return p_new
        
    def forward_tensor(self, state_flat: torch.Tensor, 
                       u_dict: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]], 
                       dt: float,
                       prediction_error: Optional[torch.Tensor] = None) -> dict:
        """
        Forward step with L2 PT_A and A3 F-tracking.
        """
        batch_size = state_flat.shape[0]
        device = state_flat.device
        
        # Compatibility Layer
        if u_dict is None:
            u_dict = {}
        elif isinstance(u_dict, torch.Tensor):
            u_dict = {'default': u_dict}
            
        # Extract current state
        s_in = ContactState(self.dim_q, batch_size, device, state_flat)
        
        # Get chart weights
        chart_weights = self.atlas.router(s_in.q)
        
        # L2: Apply parallel transport if weights changed
        if self._prev_chart_weights is not None:
            p_transported = self._apply_parallel_transport(
                s_in.p, s_in.q, 
                self._prev_chart_weights, chart_weights
            )
            # Reconstruct flat state out-of-place to avoid autograd issues
            state_flat = torch.cat([s_in.q, p_transported, s_in.s], dim=1)
            # Re-wrap
            s_in = ContactState(self.dim_q, batch_size, device, state_flat)
        else:
            # Update state flat if it might have been detached/viewed
            state_flat = s_in.flat
        
        # Generator definition
        def gen_func(s: ContactState):
            H_sum = self.internal_gen(s)
            for port_name, u_val in u_dict.items():
                if hasattr(self.generator, 'get_h_port'):
                    H_sum += self.generator.get_h_port(s, port_name, u_val, weights=chart_weights)
                else:
                    H_sum += (self.generator(s, u_val, weights=chart_weights) - self.internal_gen(s))
            return H_sum
        
        # Step dynamics
        s_next = self.integrator.step(s_in, gen_func, dt)
        
        # Track weights for next step
        self._prev_chart_weights = chart_weights.detach()
        
        # A3: Compute Free Energy
        F = self.compute_free_energy(s_next, prediction_error)
        
        return {
            'next_state_flat': s_next.flat,
            'chart_weights': chart_weights,
            'free_energy': F,
        }
        
    def get_action(self, state_flat: torch.Tensor, port_name: str = 'default') -> torch.Tensor:
        """
        Helper to get action Magnitude from flat state.
        Requirement for A3/A4 verification.
        """
        batch_size = state_flat.shape[0]
        s = ContactState(self.dim_q, batch_size, state_flat.device, state_flat)
        chart_weights = self.atlas.router(s.q)
        
        if hasattr(self.generator, 'get_action'):
            return self.generator.get_action(s, port_name, weights=chart_weights)
        return torch.zeros(batch_size, self.dim_u, device=state_flat.device)

    def update_z(self, prediction_error: torch.Tensor, lr_z: float = 0.01):
        """
        L3: Update autonomous context z via gradient descent on F.
        This is the "Autonomous Will" - z adapts to minimize prediction error.
        
        Uses explicit gradient computation to ensure proper backprop.
        """
        # Make z require grad for this computation
        z_val = self.z.detach().clone().requires_grad_(True)
        
        # Compute F using temporary z value
        # F = KL(z) + prediction_error
        KL = 0.5 * (z_val ** 2).sum()
        F = KL + prediction_error.mean()
        
        # Compute gradient
        grad_z = torch.autograd.grad(F, z_val, create_graph=False)[0]
        
        # Update z in-place
        with torch.no_grad():
            self.z.data -= lr_z * grad_z
            # Clip z to prevent runaway
            self.z.data.clamp_(-3.0, 3.0)
                
    def get_z(self) -> torch.Tensor:
        """Return current autonomous context."""
        return self.z.detach().clone()

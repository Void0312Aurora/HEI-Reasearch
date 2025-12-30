import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from he_core.state import ContactState
from he_core.contact_dynamics import ContactIntegrator
from he_core.generator import BaseGenerator, DissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.atlas import Atlas, AtlasRouter
from he_core.interfaces import ActionInterface, AuxInterface

class UnifiedGeometricEntity(nn.Module):
    """
    Entity v0.4: Situated Geometric Entity.
    
    Components:
    1. Internal Dynamics: DissipativeGenerator (Fixed or Parametric)
    2. Port Coupling: PortCoupledGenerator (Learnable shape B(q))
    3. Structure: Atlas (Learnable Router)
    4. Integrator: ContactIntegrator
    5. Interface: ActionInterface (Default)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim_q = config.get('dim_q', 2)
        self.dim_u = config.get('dim_u', self.dim_q) # Assume dim_u = dim_q by default
        self.num_charts = config.get('num_charts', 1)
        
        # 1. Generator (Internal)
        # Using Dissipative as base "Soul"
        alpha = config.get('damping', 0.1)
        self.internal_gen = DissipativeGenerator(self.dim_q, alpha=alpha)
        
        # 2. Port Coupling (The Learnable Interaction)
        learnable_coupling = config.get('learnable_coupling', False)
        self.generator = PortCoupledGenerator(
            self.internal_gen, 
            dim_u=self.dim_u, 
            learnable_coupling=learnable_coupling,
            num_charts=self.num_charts
        )
        
        # 3. Atlas / Structure
        # If num_charts > 1, we use Atlas. 
        # For v0.4, we ALWAYS wrap in Atlas logic, even if 1 chart (trivial router).
        self.atlas = Atlas(self.num_charts, self.dim_q)
        
        # 4. Integrator
        self.integrator = ContactIntegrator()
        
        # 5. Interfaces
        # Support PortInterface (Phase 15) or legacy ActionInterface
        use_port = config.get('use_port_interface', False)
        if use_port:
            from he_core.interfaces import PortInterface
            contract_method = config.get('port_contract_method', 'tanh')
            max_gain = config.get('port_max_gain', 1.0)
            self.interface = PortInterface(self.dim_q, self.dim_u, 
                                           use_contract=True, 
                                           contract_method=contract_method,
                                           max_gain=max_gain)
            self.use_port_interface = True
        else:
            self.interface = ActionInterface(self.dim_q, self.dim_u)
            self.use_port_interface = False
        
        # State Container
        self.state = ContactState(self.dim_q, 1)
        
    def reset(self, scale: float = 0.1):
        self.state.q = torch.randn(1, self.dim_q) * scale
        self.state.p = torch.randn(1, self.dim_q) * scale
        self.state.s = torch.zeros(1, 1)
        
    def forward_tensor(self, state_flat: torch.Tensor, u_ext: Optional[torch.Tensor], dt: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Differentiable Forward Step.
        Returns Tensors (maintaining gradients).
        """
        # Temporary State wrapper
        batch_size = state_flat.shape[0]
        curr_state = ContactState(self.dim_q, batch_size, device=state_flat.device, flat_tensor=state_flat)
        
        # 1. Router (Check gradient flow here)
        chart_weights = self.atlas.router(curr_state.q)
        
        # 2. Dynamics
        def H_func(s):
            # Pass weights here!
            return self.generator(s, u_ext, weights=chart_weights)
            
        next_state = self.integrator.step(curr_state, H_func, dt)
        
        # 3. Output
        # ActionInterface implements write(state) -> action
        action = self.interface.write(next_state)
        
        # 4. H Value
        H_val = self.generator(curr_state, u_ext, weights=chart_weights)
        
        return {
            "next_state_flat": next_state.flat,
            "action": action,
            "chart_weights": chart_weights,
            "H_val": H_val
        }

    def forward(self, obs: Dict[str, Any], dt: float = 0.1) -> Dict[str, Any]:
        """
        Step the Entity (Numpy Interface for Experiment Loop).
        Wraps forward_tensor.
        """
        # 1. Process Input
        u_ext = None
        if 'x_ext' in obs:
            u_raw = obs['x_ext']
            if isinstance(u_raw, torch.Tensor):
                u_ext = u_raw
            else:
                u_ext = torch.tensor(u_raw, dtype=torch.float32).reshape(1, -1)
                
        # 2. Tensor Forward (No Grad for runtime step usually, but keep logic unified)
        # We use self.state
        with torch.set_grad_enabled(self.training): # Respect mode
             out = self.forward_tensor(self.state.flat, u_ext, dt)
             
        # Update State
        self.state.flat = out['next_state_flat'].detach() # Detach for next step in runtime loop
        
        # Log weights
        weights = out['chart_weights'].detach()
        
        # H calc for log (using detached weights)
        H_val = out['H_val'].item()
        
        return {
            "action": out['action'].detach().numpy().flatten(),
            "chart_weights": weights.numpy().flatten(),
            "internal_p": self.state.p.detach().numpy().flatten(),
            "internal_s": self.state.s.item(),
            "H_val": H_val
        }

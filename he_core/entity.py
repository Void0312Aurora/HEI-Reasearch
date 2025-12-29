import torch
import numpy as np
from typing import Dict, Optional, Any
from .kernel.kernels import SymplecticKernel, ContactKernel, FastSlowKernel, ResonantKernel, PlasticKernel
from .scheduler.scheduler import Scheduler

class Entity:
    """
    HEI Entity Core (v1).
    Encapsulates Internal Dynamics (Kernel), Policy/Scheduler (Scheduler), and State (Blanket/Internal).
    Implements the standard Active Inference loop:
    Observation -> Blanket -> Internal State Update -> Action.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dim_q = config.get('dim_q', 2)
        self.seed = config.get('seed', 42)
        
        # 1. Initialize Kernel
        self.kernel = self._init_kernel(config)
        
        # 2. Initialize Scheduler
        self.scheduler = Scheduler(config)
        
        # 3. State Initialization
        # x_int: [1, Dim_Int]
        # x_blanket: [1, Dim_Blanket]
        self._init_state()
        
        # 4. Buffers for Experience
        self.u_env_buffer = [] 
        
    def _init_kernel(self, config):
        k_type = config.get('kernel_type', 'point_mass')
        dim = self.dim_q
        if k_type == 'symplectic':
            return SymplecticKernel(dim)
        elif k_type == 'contact':
            return ContactKernel(dim, damping=config.get('damping', 0.1))
        elif k_type == 'fast_slow':
            return FastSlowKernel(dim, epsilon=config.get('epsilon', 0.1))
        elif k_type == 'resonant':
            return ResonantKernel(dim, omega=config.get('omega', 1.0))
        elif k_type == 'plastic':
            return PlasticKernel(dim, omega=config.get('omega', 1.0), 
                                eta=config.get('eta', 0.01))
        else:
            # Default fallback for v0
            return ContactKernel(dim)
            
    def _init_state(self):
        torch.manual_seed(self.seed)
        # Assuming Kernel dim logic (mostly implicit in proto)
        # We need to know x_int size.
        # Run a dummy forward pass or infer from kernel attributes?
        # Kernels in proto verify shapes lazily usually.
        # But we need an initial state.
        
        # Infer dimensions based on kernel type
        if isinstance(self.kernel, PlasticKernel):
            # q, p, r, vr, W
            # W is dim_q * dim_q flattened
            d_int = 4 * self.dim_q + self.dim_q * self.dim_q
        elif isinstance(self.kernel, ResonantKernel):
            d_int = 4 * self.dim_q
        elif isinstance(self.kernel, FastSlowKernel):
             d_int = 4 * self.dim_q
        else: # Contact/Symplectic
            d_int = 2 * self.dim_q
            
        self.x_int = torch.randn(1, d_int) * 0.1
        self.x_blanket = torch.zeros(1, 2 * self.dim_q) # [u_env, u_self]
        
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Cycle Step.
        Args:
            obs: Observation dict, containing 'x_ext_proxy' or 'u_env'
        """
        # 1. Scheduler Step
        sched_info = self.scheduler.step()
        phase = sched_info['phase']
        
        # 2. Perception (Observation -> u_env)
        # obs['x_ext_proxy'] assumed [1, 4] or [4]
        if 'x_ext_proxy' in obs:
            x_ext = torch.tensor(obs['x_ext_proxy'], dtype=torch.float32)
            if x_ext.ndim == 1: x_ext = x_ext.unsqueeze(0)
            u_env = x_ext[:, :self.dim_q]
        else:
            u_env = torch.zeros(1, self.dim_q)
            
        # 3. Policy / Readout (Internal -> u_self)
        u_self = self._compute_u_self()
        
        # 4. Input Integration (Scheduler)
        # Handle Replay Logic here or in Scheduler?
        # For Entity Core, we delegate decision to Scheduler but we enable replay buffer access
        
        if phase == 'online':
            self.u_env_buffer.append(u_env.clone())
            
        u_t = self.scheduler.process_input(u_env, u_self, sched_info)
        
        # 5. Dynamics Update (Kernel)
        x_int_next = self.kernel(self.x_int, u_t)
        self.x_int = x_int_next.detach() # Detach state across steps usually
        
        # 6. Blanket Update
        self.x_blanket = torch.cat([u_env, u_self], dim=1)
        
        # 7. Act (Output)
        # Entity output is u_self (Action)
        action = u_self.detach().numpy().flatten()
        
        return {
            "action": action,
            "x_int": self.x_int.detach().numpy(),
            "x_blanket": self.x_blanket.detach().numpy(),
            "meta": sched_info
        }
        
    def _compute_u_self(self):
        # Default Mock Policy (Damping / Negative Feedback)
        # -0.05 * p
        p_idx_start = self.dim_q
        p_curr = self.x_int[:, p_idx_start:2*self.dim_q]
        
        # If config forces zero
        if self.config.get("force_u_self_zero", False):
            return torch.zeros_like(p_curr)
            
        return -0.05 * p_curr

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
            # q, p, s, r, vr, W
            # [2d + 1 + 2d + d*d]
            d_int = 4 * self.dim_q + 1 + self.dim_q * self.dim_q
        elif isinstance(self.kernel, ResonantKernel):
            # q, p, s, r, vr
            d_int = 4 * self.dim_q + 1
        elif isinstance(self.kernel, FastSlowKernel):
             # q, p, s
             d_int = 2 * self.dim_q + 1
        else: # Contact/Symplectic
            # Contact: q, p, s => 2d + 1
            # Symplectic: q, p => 2d
            if isinstance(self.kernel, SymplecticKernel):
                d_int = 2 * self.dim_q
            else:
                d_int = 2 * self.dim_q + 1
            
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
        u_self = self._compute_u_self(u_env)
        
        # 4. Input Integration (Scheduler & Memory)
        if phase == 'online':
            self.u_env_buffer.append(u_env.clone())
        
        # Handle Replay Injection (Override u_env if replay)
        if sched_info.get("u_source") == "replay":
            u_env = self._handle_replay(sched_info['step'])
            
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
            "u_t": u_t.detach().numpy(),
            "meta": sched_info
        }
    
    def _handle_replay(self, step_idx):
        """
        Retrieves replay input from buffer based on mode (Reverse/Shuffle/Sequential).
        """
        L = len(self.u_env_buffer)
        if L == 0:
            return torch.zeros(1, self.dim_q)
            
        t = step_idx
        
        # Replay Logic
        if self.config.get("replay_reverse", False):
            # Time-Reverse
            eff_t = t % L
            replay_idx = L - eff_t - 1
        elif self.config.get("replay_block_shuffle", False):
            # Block Shuffle (Deterministic based on seed)
            # Need to lazily init map?
            if not hasattr(self, '_replay_map') or len(self._replay_map) != L:
                self._init_block_shuffle_map(L)
            eff_t = t % L
            replay_idx = self._replay_map[eff_t]
        else:
            # Standard Loop
            replay_idx = t % L
            
        return self.u_env_buffer[replay_idx]
        
    def _init_block_shuffle_map(self, L):
        # Create a permutation map for blocks
        # Use a separate RNG or standard np.random (seeded at init)
        rng = np.random.RandomState(self.seed) 
        block_size = max(1, int(L * 0.1))
        num_blocks = L // block_size
        perm = rng.permutation(num_blocks if num_blocks > 0 else 1)
        
        self._replay_map = np.arange(L)
        for b_idx in range(num_blocks):
            target_b = perm[b_idx]
            start_src = target_b * block_size
            start_dst = b_idx * block_size
            size = block_size
            if start_src + size <= L and start_dst + size <= L:
                 self._replay_map[start_dst:start_dst+size] = np.arange(start_src, start_src+size)

    def _compute_u_self(self, u_env=None):
        # 1. Config Check
        if self.config.get("force_u_self_zero", False):
            return torch.zeros(1, self.dim_q)
            
        dim = self.dim_q
        
        # 2. Extract State Components
        p_idx_start = dim
        p_curr = self.x_int[:, p_idx_start:p_idx_start+dim]
        
        # 3. Active Inference Mode
        if self.config.get("active_mode", False) and isinstance(self.kernel, PlasticKernel) and u_env is not None:
            # Layout: [q, p, s, r, vr, W]
            r_idx_start = 2 * dim + 1
            w_idx_start = 4 * dim + 1
            
            r_curr = self.x_int[:, r_idx_start:r_idx_start+dim] # [1, Dim]
            w_flat = self.x_int[:, w_idx_start:] # [1, Dim*Dim]
            
            # Reconstruct W [1, Dim, Dim]
            W = w_flat.view(1, dim, dim)
            
            # Prediction: hat_x = W * r
            pred_x = torch.bmm(W, r_curr.unsqueeze(2)).squeeze(2)
            
            # Prediction Error: e = u_env - pred_x
            # u_env acts as Sensation (y)
            error = u_env - pred_x
            
            # Action: Minimize Error Squared 
            # u_self = - K * error (Negative Feedback)
            gain = self.config.get("active_gain", 1.0)
            u_active = -1.0 * gain * error
            
            return u_active
            
        # Fallback: Passive Damping
        return -0.05 * p_curr

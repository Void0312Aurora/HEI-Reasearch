import numpy as np
import torch
from typing import Dict, Any, Tuple
from .interfaces import Env, ObsAdapter

class VanDerPolEnv(Env):
    """
    Driven Van der Pol Oscillator Environment.
    Equation: x'' - mu(1 - x^2)x' + x = A * sin(omega * t)
    Non-linear damping + Driving force.
    """
    def __init__(self, config: Dict[str, Any]):
        self.dt = config.get('dt', 0.1)
        self.mu = config.get('vdp_mu', 2.0) # Non-linearity parameter
        self.drive_amp = config.get('drive_amp', 1.0)
        self.drive_freq = config.get('drive_freq', 1.0)
        self.noise_std = config.get('env_noise', 0.0)
        self.max_steps = config.get('max_steps', 1000)
        
        self.state = None # [x, v]
        self.t = 0
        self.step_count = 0
        
    def reset(self, seed: int = None) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)
            
        # Random init state
        self.state = np.random.randn(2)
        self.t = 0
        self.step_count = 0
        
        return self._get_obs()
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        x, v = self.state
        
        # Action is additive force
        u = action[0] if len(action) > 0 else 0.0
        
        # Dynamics (RK4 or Euler? Euler for now)
        # x' = v
        # v' = mu(1-x^2)v - x + A*sin(wt) + u + noise
        
        drive = self.drive_amp * np.sin(self.drive_freq * self.t)
        noise = np.random.randn() * self.noise_std
        
        dv = self.mu * (1 - x**2) * v - x + drive + u + noise
        
        # Update
        self.state[0] += v * self.dt
        self.state[1] += dv * self.dt
        
        self.t += self.dt
        self.step_count += 1
        
        done = self.step_count >= self.max_steps
        
        # Reward: Radius stabilization around a target? Or just 0.
        reward = -np.abs(x) # Minimize magnitude?
        
        return self._get_obs(), reward, done, {}
        
    def _get_obs(self):
        return {
            "x_ext_proxy": np.concatenate([self.state, np.zeros(2)]) # Pad to 4 for Entity
        }

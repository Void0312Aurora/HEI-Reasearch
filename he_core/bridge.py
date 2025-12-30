import numpy as np
from typing import Dict, Any, Tuple
from .interfaces import Env, ObsAdapter, ActAdapter
from EXP.proto.env.limit_cycle import LimitCycleEnv

class LimitCycleGymEnv(Env):
    """
    Wrapper around EXP/proto/env/limit_cycle.py to conform to he_core.Env interface.
    """
    def __init__(self, config: Dict[str, Any]):
        # Unpack config for LimitCycleEnv
        dt = config.get('dt', 0.1)
        gamma = config.get('env_gamma', 0.1)
        omega = config.get('omega', 1.0)
        drive_amp = config.get('drive_amp', 1.0)
        drive_freq = config.get('drive_freq', 0.5)
        
        self.env = LimitCycleEnv(dt=dt, gamma=gamma, omega=omega, 
                                drive_amp=drive_amp, drive_freq=drive_freq)
        self.max_steps = config.get('max_steps', 1000)
        self.steps = 0
        
    def reset(self, seed: int = None) -> Dict[str, Any]:
        self.steps = 0
        return self.env.reset(seed)
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.steps += 1
        obs, reward, done, info = self.env.step(action)
        
        # Auto-termination
        if self.steps >= self.max_steps:
            done = True
            
        return obs, reward, done, info

class LimitCycleObsAdapter(ObsAdapter):
    """
    Passes x_ext_proxy through.
    """
    def adapt(self, env_obs: Dict[str, Any]) -> Dict[str, Any]:
        # LimitCycleEnv already returns {'x_ext_proxy': ...}
        # We ensure it's compatible.
        if 'x_ext_proxy' in env_obs:
            return env_obs
        else:
            # Fallback or error?
            return {'x_ext_proxy': np.zeros(4)} # Should not happen with Proto Env

class LinearActAdapter(ActAdapter):
    """
    Scales and clips action.
    """
    def __init__(self, scale: float = 1.0, clip: float = 10.0):
        self.scale = scale
        self.clip = clip
        
    def adapt(self, entity_action: np.ndarray) -> np.ndarray:
        # Scale
        act = entity_action * self.scale
        # Clip
        act = np.clip(act, -self.clip, self.clip)
        return act

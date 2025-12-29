import numpy as np
import dataclasses
from typing import Tuple, Dict, Any

@dataclasses.dataclass
class EnvState:
    x: np.ndarray  # Position [x, y]
    v: np.ndarray  # Velocity [vx, vy]

class PointMassEnv:
    """
    A minimal 2D point mass environment.
    State: [x, y, vx, vy]
    Action: [ax, ay] (Force/Acceleration)
    """
    def __init__(self, dt: float = 0.1, bounds: float = 5.0):
        self.dt = dt
        self.bounds = bounds
        self.state = EnvState(x=np.zeros(2), v=np.zeros(2))
        self.rng = np.random.default_rng(42)

    def reset(self, seed: int = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Random init within bounds
        self.state.x = self.rng.uniform(-self.bounds, self.bounds, size=2)
        self.state.v = self.rng.uniform(-1.0, 1.0, size=2)
        
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        # Simple symplectic Euler integration
        action = np.clip(action, -1.0, 1.0) # Clip action
        
        # v_{t+1} = v_t + a * dt
        self.state.v += action * self.dt
        # x_{t+1} = x_t + v_{t+1} * dt
        self.state.x += self.state.v * self.dt
        
        # Soft boundary bounce
        for i in range(2):
            if self.state.x[i] < -self.bounds:
                self.state.x[i] = -self.bounds
                self.state.v[i] *= -0.8
            elif self.state.x[i] > self.bounds:
                self.state.x[i] = self.bounds
                self.state.v[i] *= -0.8
                
        return self._get_obs(), 0.0, False, {}

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "x_ext_proxy": np.concatenate([self.state.x, self.state.v])
        }

import numpy as np

class LimitCycleEnv:
    """
    Iteration 1.4: Structured Environment for E2.
    Dynamics: Damped Driven Harmonic Oscillator.
    ddx + gamma * dx + omega^2 * x = F * sin(Omega * t) + noise
    
    This system has a clear arrow of time due to damping (gamma > 0).
    Time reversal (t -> -t) maps damped dynamics to anti-damped (exploding) dynamics.
    """
    def __init__(self, dt=0.1, gamma=0.2, omega=1.0, drive_amp=1.0, drive_freq=0.5):
        self.dt = dt
        self.gamma = gamma
        self.omega = omega
        self.F = drive_amp
        self.Omega = drive_freq
        
        # State: [x, v]
        self.state = np.zeros(2)
        self.t = 0.0
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(-1, 1, size=2)
        self.t = 0.0
        return self._get_obs()
        
    def step(self, action=None):
        # Action is additive noise/perturbation
        if action is None:
            action = np.zeros(2)
            
        x, v = self.state
        
        # Physics Step (Euler-Maruyama or RK4, use Euler for simplicity/noise)
        # Force = -gamma*v - omega^2*x + F*sin(Omega*t) + action
        
        drive = self.F * np.sin(self.Omega * self.t)
        
        # Acceleration
        a = -self.gamma * v - (self.omega**2) * x + drive + action[0] # Apply action as noise on accel
        
        # Update
        v_next = v + a * self.dt
        x_next = x + v_next * self.dt
        
        self.state = np.array([x_next, v_next])
        self.t += self.dt
        
        return self._get_obs(), 0.0, False, {}
        
    def _get_obs(self):
        # x_ext_proxy: [x, v, a, j] (pad to 4)
        x, v = self.state
        return {
            "x_ext_proxy": np.array([x, v, 0.0, 0.0], dtype=np.float32)
        }

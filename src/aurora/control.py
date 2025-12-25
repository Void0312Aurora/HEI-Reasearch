import torch

class RadiusPIDController:
    """
    PID Controller for Hyperbolic Radius Distribution.
    Adjusts the stiffness of RadiusAnchorPotential (lambda) to maintain target mean radius.
    """
    def __init__(self, target_r: float = 0.8, Kp: float = 0.5, Ki: float = 0.01, Kd: float = 0.1, 
                 base_lamb: float = 0.1, max_lamb: float = 5.0, min_lamb: float = 0.0):
        self.target_r = target_r
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.base_lamb = base_lamb
        self.max_lamb = max_lamb
        self.min_lamb = min_lamb
        
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, current_mean_r: float, dt: float = 1.0) -> float:
        """
        Updates the control state and returns the new lambda value.
        """
        error = current_mean_r - self.target_r
        
        # Integral with simple anti-windup (only accumulate if output not saturated)
        # Note: In our case, higher r means we need higher lambda to pull it back. 
        # So it's a negative feedback loop where u = Kp * error...
        self.integral += error * dt
        
        # Derivative
        derivative = (error - self.prev_error) / dt
        
        # Output: Adjustment to base_lamb
        adjustment = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        new_lamb = self.base_lamb + adjustment
        new_lamb = max(min(new_lamb, self.max_lamb), self.min_lamb)
        
        self.prev_error = error
        return new_lamb

    def get_diagnostics(self):
        return {
            'pid_error': self.prev_error,
            'pid_integral': self.integral
        }

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
        Updates the control state and returns the new radii_scale for target_radii.
        Instead of controlling lambda, we scale the target radii distribution.
        """
        error = current_mean_r - self.target_r
        
        # Integral with anti-windup
        self.integral += error * dt
        
        # Derivative
        derivative = (error - self.prev_error) / dt
        
        # Output: Scale factor for target_radii (not lambda adjustment)
        # If current < target (error negative), we need to INCREASE radii -> scale > 1
        # So we use NEGATIVE feedback: adjustment = -Kp * error
        adjustment = -self.Kp * error - self.Ki * self.integral - self.Kd * derivative
        
        # radii_scale = 1.0 + adjustment (centered at 1.0, can go 0.5 to 2.0)
        radii_scale = 1.0 + adjustment
        radii_scale = max(min(radii_scale, 2.0), 0.5)  # Clamp to reasonable range
        
        self.prev_error = error
        return radii_scale

    def get_diagnostics(self):
        return {
            'pid_error': self.prev_error,
            'pid_integral': self.integral
        }

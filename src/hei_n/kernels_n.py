
import numpy as np
import dataclasses
from typing import Callable

# Functional Interface
# Kernel: d -> V(d)
# Derivative: d -> V'(d)

@dataclasses.dataclass
class GaussianRepulsion:
    """
    Soft Repulsion Potential (Gaussian).
    V(d) = A * exp(-d^2 / (2 * sigma^2))
    
    Gradient:
    V'(d) = A * (-d / sigma^2) * exp(...)
          = - (d / sigma^2) * V(d)
          
    Features:
    - Force goes to 0 as d -> 0 (Soft Core), avoiding singularity.
    - Repulsion is max at d = sigma.
    """
    sigma: float = 1.0
    A: float = 1.0
    
    def __call__(self, d: np.ndarray) -> np.ndarray:
        return self.A * np.exp(-d**2 / (2 * self.sigma**2))
        
    def derivative(self, d: np.ndarray) -> np.ndarray:
        # returns dV/dd
        val = self(d)
        return - (d / self.sigma**2) * val

@dataclasses.dataclass
class ExponentialRepulsion:
    """
    Hard Repulsion (Schem A).
    V(d) = A * exp(-d)
    V'(d) = -A * exp(-d)
    
    Singular at d=0 (if used with 1/sinh(d) in H-space gradient).
    """
    A: float = 1.0
    
    def __call__(self, d: np.ndarray) -> np.ndarray:
        return self.A * np.exp(-d)
        
    def derivative(self, d: np.ndarray) -> np.ndarray:
        return -self.A * np.exp(-d)

@dataclasses.dataclass
class SpringAttraction:
    """
    Quadratic Attraction.
    V(d) = 0.5 * k * d^2
    V'(d) = k * d
    """
    k: float = 1.0
    
    def __call__(self, d: np.ndarray) -> np.ndarray:
        return 0.5 * self.k * d**2
        
    def derivative(self, d: np.ndarray) -> np.ndarray:
        return self.k * d

@dataclasses.dataclass
class SoftBarrierRepulsion:
    """
    Theoretical Fix: Bounded Force Repulsion (Softplus).
    Approximates a Minimum Distance Constraint d >= r0.
    
    V(d) = A * sigma * log(1 + exp((r0 - d) / sigma))
    
    Gradient:
    V'(d) = -A * sigmoid((r0 - d) / sigma)
    
    Properties:
    - Max Force = A (at d << r0).
    - Force -> 0 (at d >> r0).
    - Max Stiffness = A / (4 * sigma) (at d = r0).
    - No Singularity at d=0.
    """
    r0: float = 1.0 # Target minimum distance
    sigma: float = 0.1 # Soft transitions width
    A: float = 1.0 # Max Force Amplitude
    
    def __call__(self, d: np.ndarray) -> np.ndarray:
        # V = A * sigma * softplus(x)
        x = (self.r0 - d) / self.sigma
        # Numerical stable softplus:
        # log(1+exp(x)) approx x for x > 30, 0 for x < -30
        return self.A * self.sigma * np.logaddexp(0, x)
        
    def derivative(self, d: np.ndarray) -> np.ndarray:
        # V' = -A * sigmoid(x)
        x = (self.r0 - d) / self.sigma
        # Sigmoid: 1 / (1 + exp(-x))
        return -self.A / (1.0 + np.exp(-x))

@dataclasses.dataclass
class LogCoshRepulsion:
    """
    Theoretical Fix (User Rec #1): Log-Cosh Repulsion.
    
    V(d) = A * sigma * log(cosh(d / sigma))
    
    Gradient:
    V'(d) = A * tanh(d / sigma)
    
    Properties:
    - Bounded Force: |F| <= A (at infinity).
    - Stable at Origin: V'(d) ~ (A/sigma) * d.
      This cancels the 1/sinh(d) singularity in hyperbolic gradient.
    - Bounded Stiffness: |H| <= A/sigma.
    """
    sigma: float = 1.0 # Width
    A: float = 1.0 # Max Force
    
    def __call__(self, d: np.ndarray) -> np.ndarray:
        # log(cosh(x)) = x - log 2 + ... for large x
        # stable implementation: x + softplus(-2x) - log 2
        x = np.abs(d) / self.sigma
        val = x + np.log1p(np.exp(-2*x)) - np.log(2.0)
        return self.A * self.sigma * val
        
    def derivative(self, d: np.ndarray) -> np.ndarray:
        # V' = A * tanh(x)
        return self.A * np.tanh(d / self.sigma)


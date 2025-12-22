
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

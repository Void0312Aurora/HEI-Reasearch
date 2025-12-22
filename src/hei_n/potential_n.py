"""
N-Dimensional Potentials for HEI-N.
===================================

Implements potential functions V(x) and gradients in embedding space R^{n+1}.
Designed for use with ContactIntegratorN.

Modules:
- HarmonicPriorN: Global attraction to origin (Gravity).
- PairwisePotentialN: Interaction forces between particles (Lennard-Jones, Tanh, etc).
"""

import numpy as np
from typing import Callable, List, Optional
from .geometry_n import dist_n, dist_grad_n, minkowski_inner

class HarmonicPriorN:
    """
    Global quadratic potential attracting to the Origin e0=(1,0,...).
    V(x) = 0.5 * k * d(x, e0)^2.
    """
    def __init__(self, k: float = 1.0, e0: Optional[np.ndarray] = None):
        self.k = k
        self.e0 = e0
        
    def potential(self, x: np.ndarray) -> float:
        # x shape: (N, dim)
        if self.e0 is None:
            # Default Origin: (1, 0, ...)
            # d(x, e0) = arccosh(x0) since <x, e0>_M = -x0*1 + 0 = -x0.
            x0 = x[..., 0]
            d = np.arccosh(np.maximum(x0, 1.0))
        else:
            d = dist_n(x, self.e0)
            
        V = 0.5 * self.k * np.sum(d**2)
        return float(V)
        
    def gradient(self, x: np.ndarray) -> np.ndarray:
        # grad V = k * d * grad d
        if self.e0 is None:
            x0 = x[..., 0]
            d = np.arccosh(np.maximum(x0, 1.0 + 1e-7))
            denom = np.sqrt(x0**2 - 1.0)
            denom = np.maximum(denom, 1e-7)
            
            # grad d (Euclidean) wrt x
            # d = arccosh(x0). d' = 1/sqrt(x0^2-1).
            # grad d = (1/sqrt, 0, 0...)
            
            grad = np.zeros_like(x)
            factor = self.k * d / denom
            grad[..., 0] = factor
        else:
            # TODO: Implement generic gradient if needed
            # For now assume e0 is Origin for efficiency
            return self.gradient(x) # Recursion fallback if e0 is default
            
        return grad

class PairwisePotentialN:
    """
    Pairwise interactions between all particles.
    V = sum_{i<j} Kernel(d_ij)
    """
    def __init__(self, kernel_fn: Callable[[np.ndarray], np.ndarray], 
                 d_kernel_fn: Callable[[np.ndarray], np.ndarray]):
        """
        kernel_fn: d -> V
        d_kernel_fn: d -> V' (derivative wrt d)
        """
        self.kernel_fn = kernel_fn
        self.d_kernel_fn = d_kernel_fn
        
    def potential(self, x: np.ndarray) -> float:
        # x: (N, dim)
        # Compute Di distance matrix
        # Uses broadcasting: (N, 1, dim) vs (1, N, dim)
        inner = minkowski_inner(x[:, np.newaxis, :], x[np.newaxis, :, :])
        
        # d_ij = arccosh(-inner)
        # Numerical guard
        val = np.maximum(-inner, 1.0 + 1e-7)
        dists = np.arccosh(val)
        
        # Mask diagonal and lower triangle to avoid double counting
        # For potential sum, we sum upper triangle.
        mask = np.triu(np.ones(dists.shape, dtype=bool), k=1)
        
        d_valid = dists[mask]
        v_pairs = self.kernel_fn(d_valid)
        
        return float(np.sum(v_pairs))
        
    def gradient(self, x: np.ndarray) -> np.ndarray:
        # Gradient dV/dx_i = sum_j V'(d_ij) * grad_x_i (d_ij)
        # grad_x_i (d_ij) calculation:
        # inner u_ij = <x_i, x_j>. d = arccosh(-u).
        # grad_xi d = (1/sqrt) * grad_xi (-u) = (1/sqrt) * (- J x_j)
        # = -1/sqrt * J x_j.
        
        # We need to vectorized this.
        # F_ij (force on i from j) = - grad_xi V(d_ij) 
        # = - V'(d) * (-1/sqrt) * J x_j
        # = (V'(d)/sqrt) * J x_j
        
        N, dim = x.shape
        inner = minkowski_inner(x[:, np.newaxis, :], x[np.newaxis, :, :])
        val = np.maximum(-inner, 1.0 + 1e-7)
        dists = np.arccosh(val)
        
        denom = np.sqrt(val**2 - 1.0)
        denom = np.maximum(denom, 1e-7)
        
        # V'(d)
        dv = self.d_kernel_fn(dists)
        
        # Prefactor S_ij = V'(d_ij) / sinh(d_ij)
        S = dv / denom
        
        # Remove diagonal (self-interaction)
        np.fill_diagonal(S, 0.0)
        
        # Total Force on i:
        # Grad_i = sum_j (S_ij * J * x_j) ? No.
        # Check sign carefully.
        # grad_xi d_ij = (1/sinh) * (- J x_j).
        # grad_xi V = V' * (1/sinh) * (- J x_j).
        #           = - S_ij * (J x_j).
        # BUT this is only half the story? 
        # V = sum_{k<l} E(d_kl).
        # i appears in pairs (i, j) and (j, i).
        # sum_j!=i V'(d_ij) * grad_xi d_ij
        # = sum_j - S_ij * J x_j.
        
        # We need J x_j for all j.
        J = np.ones(dim); J[0] = -1.0
        Jx = x * J[np.newaxis, :] # (N, dim)
        
        # Grad_i = - J * sum_j (S_ij * x_j)
        # Matrix mult: S (N, N) @ x (N, dim) -> (N, dim)
        weighted_sum = S @ x
        
        Grad = - J[np.newaxis, :] * weighted_sum
        
        return Grad

class CompositePotentialN:
    def __init__(self, terms: List):
        self.terms = terms
        
    def potential(self, x):
        return sum(t.potential(x) for t in self.terms)
        
    def gradient(self, x):
        return sum(t.gradient(x) for t in self.terms)

# Common Kernels
def kernel_lennard_jones(d, sigma=1.0, epsilon=1.0):
    # d^-12 - d^-6 ?
    # Better for Hyperbolic:
    # Repulsion e^(-d) - Attraction e^(-d/2) ?
    # Let's use simple Repulsion - Attraction
    # V = A e^{-d/sigma} - B e^{-d/lambda}
    pass
    
def kernel_soft_cluster(d, sigma=0.5):
    # Gaussian well: - e^{-d^2 / 2sigma^2}
    return -np.exp(-d**2 / (2 * sigma**2))

def d_kernel_soft_cluster(d, sigma=0.5):
    # d/dd (-exp) = -exp * (-2d / 2sigma^2) = (d/sigma^2) * exp
    val = -np.exp(-d**2 / (2 * sigma**2))
    return val * (d / sigma**2) # Note sign is positive (Attraction increases V as d increases? No V is negative Gaussian)
    # V goes from -1 to 0. Min at d=0.
    # Force = -grad V = - (d/sigma^2 exp) grad d.
    # Gradient is positive -> Pushes towards larger d?
    # Wait.
    # V(d) looks like \_/ (inverted Gaussian). Min at 0.
    # V'(d) > 0 for d > 0.
    # Force = - V'(d) grad d.
    # grad d points away. -grad d points in.
    # So Force points IN. Correct.

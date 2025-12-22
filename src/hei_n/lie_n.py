"""
Lie Algebra utilities for SO(1, n) (N-dimensional Hyperbolic Isometries).
========================================================================

Mathematical Background:
------------------------
The isometry group of H^n is SO(1, n).
The Lie algebra so(1, n) consists of matrices X such that:
    X^T J + J X = 0
where J = diag(-1, 1, ..., 1) is the Minkowski metric.

Structure of X:
X = [[0,   u^T],
     [u,   W  ]]
where:
- u is a vector in R^n (Boost generators)
- W is a skew-symmetric matrix in so(n) (Rotation generators)

Representation:
We represent an algebra element xi as a vector if dimensions permit, 
but for general n, we work with the full (n+1)x(n+1) matrix X directly
to avoid complex parameterization issues.
"""

import numpy as np
from scipy.linalg import expm

def minkowski_metric(dim: int) -> np.ndarray:
    """Returns J = diag(-1, 1, ..., 1) of size (dim, dim)."""
    J = np.eye(dim)
    J[0, 0] = -1.0
    return J

def check_algebra_membership(X: np.ndarray, tol=1e-8) -> bool:
    """Check if X is in so(1, n). X^T J + J X = 0."""
    dim = X.shape[-1]
    J = minkowski_metric(dim)
    # Broadcasting check
    # (..., n, n)
    res = np.swapaxes(X, -1, -2) @ J + J @ X
    return np.allclose(res, 0, atol=tol)

def exp_so1n(X: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Exponential map for so(1, n).
    G = exp(dt * X)
    
    For n=2 (H^2), this matches SL(2, R) structure roughly.
    For general n, we use Padé approximation (scipy.linalg.expm) 
    or a closed form decomposition if implemented.
    
    Here we use scipy's expm for robustness in high dimensions, 
    vectorized via a loop if needed (scipy doesn't batch well).
    
    Args:
        X: (..., dim, dim) Algebra elements
        dt: float time step
    Returns:
        G: (..., dim, dim) Group elements in SO(1, n)
    """
    # Simple unbatched wrapper for clarity first
    # Optimization: For 3D (4x4 matrices), expm is fast enough.
    # X_scaled = X * dt
    
    X = np.asarray(X)
    if X.ndim == 2:
        return expm(X * dt)
    else:
        # Batched
        res = np.zeros_like(X)
        scale_X = X * dt
        for i in range(X.shape[0]):
            res[i] = expm(scale_X[i])
        return res

def project_to_algebra(M: np.ndarray) -> np.ndarray:
    """
    Project an arbitrary matrix M onto so(1, n).
    X = 0.5 * (M - J M^T J)
    Derived from X^T J + J X = 0 constraints.
    """
    dim = M.shape[-1]
    J = minkowski_metric(dim)
    # M_star = - J M^T J
    # But projection implies finding closest X.
    # The condition is X = - J X^T J.
    # Let's use the property that any M can be split into sol(1,n) and perp.
    # X = (M - J M^T J) / 2 check:
    # (X)^T J + J X = ...
    # This formula creates a matrix X satisfying the property.
    J_b = J # Broadcat if needed
    MT = np.swapaxes(M, -1, -2)
    # Term 2: J * MT * J. (Since J=J^T=J^-1)
    Term2 = J @ MT @ J
    return 0.5 * (M - Term2)

def random_algebra_element(dim: int, scale: float = 1.0) -> np.ndarray:
    """Generate random so(1, n-1) element."""
    M = np.random.randn(dim, dim) * scale
    return project_to_algebra(M)

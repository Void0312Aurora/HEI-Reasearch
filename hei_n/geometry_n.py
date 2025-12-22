"""
N-Dimensional Hyperboloid Geometry Utilities.
=============================================

Manifold M: Hyperboloid H^n embedded in R^{n+1}.
Constraint: <x, x>_M = -x0^2 + x1^2 + ... + xn^2 = -1, with x0 > 0.
Metric: Induced Minkowski metric.
"""

import numpy as np

def minkowski_inner(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute <x, y>_M = -x0*y0 + x_rest * y_rest.
    Input shape: (..., dim)
    """
    x0 = x[..., 0]
    y0 = y[..., 0]
    x_rest = x[..., 1:]
    y_rest = y[..., 1:]
    
    dot_rest = np.sum(x_rest * y_rest, axis=-1)
    return -x0 * y0 + dot_rest

def dist_n(x: np.ndarray, y: np.ndarray, eps=1e-7) -> np.ndarray:
    """
    Geodesic distance on H^n.
    d(x, y) = arccosh( -<x, y>_M )
    """
    inner = minkowski_inner(x, y)
    # Clamp for numerical stability (inner <= -1)
    inner = np.minimum(inner, -1.0 + eps)
    return np.arccosh(-inner)

def dist_grad_n(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of distance d(x, y) with respect to x.
    This is the gradient in the embedding space R^{n+1}, 
    but effectively needs to be projected to tangent space if used for dynamics.
    
    Formula:
    d = arccosh(u), u = -<x, y>
    grad_d = (1 / sqrt(u^2 - 1)) * grad_u
    grad_u = -y (in Minkowski sense? No, Euclidean gradient of bilinear form)
    d<u>/dx_i = -y_i * sgn_i?
    
    Let's use Euclidean gradient of the inner product function:
    f(x) = -x0 y0 + x1 y1 ...
    grad_E f = (-y0, y1, y2...)
    
    Wait, "Minkowski Gradient" vs "Euclidean Gradient".
    Hamiltonian usually defined on manifold.
    Force = -grad V.
    If V depends on inner product, we calculate Euclidean gradient and then project to Tangent Space.
    
    Euclidean Gradient of <x, y>_M wrt x is:
    J = diag(-1, 1, 1...)
    grad_E = J * y
    """
    inner = minkowski_inner(x, y)
    # Clip inner to avoid singularity at d=0
    inner = np.minimum(inner, -1.0 - 1e-9)
    
    dist_val = np.arccosh(-inner)
    
    # d(arccosh(u))/du = 1/sqrt(u^2 - 1)
    # u = -<x, y>_M
    denom = np.sqrt(inner**2 - 1)
    
    # Gradient of u w.r.t x (Euclidean):
    # grad_x (<x, y>_M) = J y
    dim = x.shape[-1]
    J = np.ones(dim); J[0] = -1.0
    
    # Broadcasting J * y
    grad_u_euc = J * y 
    
    # Chain rule: d = arccosh(u) -> d' = 1/sqrt(...) * du/dx * (-1 from u definition?)
    # u = - inner
    # du/dx = - J y
    # result = (1/sqrt) * (-J y)
    
    grad_d_euc = (-1.0 / denom)[..., np.newaxis] * grad_u_euc
    return dist_val, grad_d_euc

def project_to_tangent(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Project vector v onto Tangent Space T_x H^n.
    Constraint: <x, x> = -1 => <x, v> = 0.
    Proj(v) = v + <x, v>_M * x
    """
    inner = minkowski_inner(x, v)
    return v + inner[..., np.newaxis] * x

def lorentz_boost_n(xi: np.ndarray) -> np.ndarray:
    """
    Applicable if xi is algebra element.
    Just reference exp_so1n from lie_n.
    """
    pass # Not needed here

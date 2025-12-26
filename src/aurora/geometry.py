"""
Aurora Geometry: Hyperbolic Space Primitives (Torch).
=====================================================

Implements the Minkowski hyperboloid model H^n embedded in R^{n+1}.
Metric signature: (-1, 1, 1, ...)

Key components:
- Inner products & Norms
- Geodesic distances & gradients
- Tangent space projection
- Exponential map (for integration)
- Renormalization (Numerical stability)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def minkowski_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute <u, v>_M = -u_0*v_0 + sum(u_i * v_i).
    
    Args:
        u: (..., dim) tensor
        v: (..., dim) tensor
        
    Returns:
        (...,) scalar tensor
    """
    # u_0 * v_0
    t_term = u[..., 0] * v[..., 0]
    # sum(u_i * v_i)
    s_term = torch.sum(u[..., 1:] * v[..., 1:], dim=-1)
    return -t_term + s_term

def minkowski_norm_sq(u: torch.Tensor) -> torch.Tensor:
    """Compute <u, u>_M."""
    return minkowski_inner(u, u)

def dist_hyperbolic(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute geodesic distance on H^n.
    d(u, v) = acosh( -<u, v>_M )
    
    Constraint: u, v must be on H^n (<u,u>=-1, u0>0).
    Input check is not performed for performance, but clamping is applied.
    """
    inner = minkowski_inner(u, v)
    # Theoretically inner <= -1. Numerically can be -0.999...
    # Clamp to avoid NaN in acosh
    inner = torch.clamp(inner, max=-1.0 - eps)
    return torch.acosh(-inner)

def project_to_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project vector v onto tangent space at x (T_x H^n).
    
    Formula: proj_x(v) = v + <x, v>_M * x
    (Since <x, x>_M = -1)
    
    Args:
        x: Point on H^n (..., dim)
        v: Vector in R^{n+1} (..., dim)
    """
    inner = minkowski_inner(x, v)
    # v - (-<x, v> * x) / <x,x> where <x,x>=-1
    # proj = v + <x, v> * x
    return v + inner.unsqueeze(-1) * x

def renormalize_frame(G: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Orthonormalize the frame G using Minkowski Gram-Schmidt.
    
    G is (N, dim, dim).
    G[..., 0] is position x (must satisfy <x,x>=-1).
    G[..., 1:] are spatial frame vectors (must satisfy <e_i, e_j> = delta_ij, <x, e_i>=0).
    
    Returns:
        G_new: Corrected frame.
        error_mag: Magnitude of correction (diagnostic).
    """
    # 1. Normalize x (Column 0)
    x = G[..., 0]
    # <x, x> should be -1.
    norm_sq = minkowski_norm_sq(x)
    # If norm_sq > 0 (timelike turned spacelike), this is bad. Assume negative.
    # Scale factor: 1 / sqrt(-norm_sq)
    valid_mask = norm_sq < -1e-9
    
    # Correction Diagnostics
    err_x = torch.abs(norm_sq + 1.0).mean().item()
    
    scale = torch.ones_like(norm_sq)
    scale[valid_mask] = 1.0 / torch.sqrt(-norm_sq[valid_mask])
    # If invalid (positive norm), we attempt to fix by resetting to (1, 0...) ?
    # For now, just clamp/ignore if completely broken, but scaling handles small drift.
    
    x_new = x * scale.unsqueeze(-1)
    G_new = G.clone()
    G_new[..., 0] = x_new
    
    # 2. Gram-Schmidt for columns 1..dim-1
    dim = G.shape[-1]
    
    for i in range(1, dim):
        v = G_new[..., i]
        
        # Project out previous columns
        # v_new = v - sum_j ( <v, e_j> / <e_j, e_j> ) * e_j
        # For j=0 (x): <e_0, e_0> = -1. coeff = -<v, x>. term = +<v,x>x
        # For j>0: <e_j, e_j> = +1. coeff = <v, e_j>. term = -<v,e_j>e_j
        
        # J=0 case
        e0 = G_new[..., 0]
        coeff0 = minkowski_inner(v, e0) # <v, x>
        v = v + coeff0.unsqueeze(-1) * e0 # Subtract projection onto negative norm vec
        
        # J=1..i-1
        for j in range(1, i):
            ej = G_new[..., j]
            coeff = minkowski_inner(v, ej)
            v = v - coeff.unsqueeze(-1) * ej
            
        # Normalize v
        # Should be spacelike (<v, v> > 0)
        v_norm_sq = minkowski_norm_sq(v)
        v_norm_sq = torch.clamp(v_norm_sq, min=1e-9)
        v = v / torch.sqrt(v_norm_sq).unsqueeze(-1)
        
        G_new[..., i] = v
        
    return G_new, err_x

def random_hyperbolic_init(N: int, dim: int, scale: float = 0.1, device='cpu') -> torch.Tensor:
    """
    Initialize N points near origin on H^n.
    
    Args:
        N: Number of points
        dim: Embedding dimension (R^{dim}) => H^{dim-1}
        scale: Spread of initialization
        
    Returns:
        (N, dim) tensor on H^n
    """
    # Random direction in tangent space at origin
    # Origin = (1, 0, 0...)
    # Tangent vectors = (0, v1, v2...)
    v = torch.randn(N, dim-1, device=device)
    v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-9
    v = v / v_norm # Unit vectors
    
    # Random radius
    r = torch.rand(N, 1, device=device) * scale
    
    # Map to H^n via exp map at origin
    # x = cosh(r) * e0 + sinh(r) * v
    # where e0 = (1, 0...), v embedded as (0, v...)
    
    ch = torch.cosh(r)
    sh = torch.sinh(r)
    
    x = torch.zeros(N, dim, device=device)
    x[:, 0:1] = ch
    x[:, 1:] = sh * v
    
    return x

def log_map(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute Logarithmic Map Log_x(y) on H^n.
    Returns tangent vector u in T_x H^n such that Exp_x(u) = y.
    
    Formula:
        u = theta / sinh(theta) * (y + <x, y>_M * x)
        where theta = dist(x, y) = acosh(-<x,y>_M)
        
    Args:
        x: Base point (..., dim)
        y: Target point (..., dim)
        
    Returns:
        u: Tangent vector (..., dim)
    """
    inner = minkowski_inner(x, y)
    inner = torch.clamp(inner, max=-1.0 - eps)
    
    theta = torch.acosh(-inner)
    
    # y_perp = y + <x,y>x
    y_perp = y + inner.unsqueeze(-1) * x
    
    # Factor = theta / sinh(theta)
    # Using Taylor expansion for small theta to avoid 0/0
    sinh_theta = torch.sinh(theta)
    
    # Safe division
    factor = theta / (sinh_theta + 1e-9)
    
    # If theta is very small (< 1e-4), factor -> 1.0. 
    # y_perp norm is sinh(theta). 
    # Vector u is theta * normalized(y_perp).
    
    u = factor.unsqueeze(-1) * y_perp
    return u

def compute_gradient_minkowski(x: torch.Tensor, euc_grad: torch.Tensor) -> torch.Tensor:
    """
    Convert Euclidean gradient of a function f(x) to Minkowski gradient (?)
    Actually, we usually need 'Covariant Gradient' or project directly.
    
    Standard pipeline:
    1. Compute Euclidean Gradient of V(x): grad_E
    2. Minkowski Gradient grad_M = J * grad_E (where J = diag(-1, 1...))
    3. Project to Tangent: proj(grad_M)
    
    This function applies step 2: J * grad_E.
    """
    # J = diag(-1, 1, 1...)
    # res_0 = -grad_0
    # res_i = grad_i
    
    res = euc_grad.clone()
    res[..., 0] = -res[..., 0]
    return res

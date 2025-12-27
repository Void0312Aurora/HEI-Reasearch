"""
Aurora Geometry: Poincare Ball Model (CCD v3.1).
================================================

Implements the Poincare Ball D^n with curvature K=-1.
Metric: g_x = (2 / (1 - ||x||^2))^2 * g_E

Ref: 
- "Mobius Raymanian Geometry" concepts.
- Axiom 0.1.1 & 2.4.1 of CCD v3.1.

Components:
1. Mobius Addition (x (+) y)
2. ExpMap / LogMap
3. Parallel Transport (via Gyration)
4. Distance
"""

import torch

# Numerical stability
EPS = 1e-5
MIN_NORM = 1e-15

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Mobius Addition in Poincare Ball.
    x (+) y = ( (1 + 2c<x,y> + c|y|^2)x + (1 - c|x|^2)y ) / (1 + 2c<x,y> + c^2|x|^2|y|^2)
    
    Args:
        x: (..., dim) point
        y: (..., dim) vector/point
        c: Curvature parameter (c = |K| = 1 for CCD v3.1 standard)
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    
    num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    denom = 1 + 2*c*xy + c**2 * x2 * y2
    
    return num / (denom + EPS)

def mobius_sub(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """x (-) y = x (+) (-y)"""
    return mobius_add(x, -y, c)

def lambda_x(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Conformal factor lambda_x = 2 / (1 - c|x|^2)"""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    return 2.0 / (1.0 - c*x2 + EPS)

def dist(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Geodesic distance.
    d(x, y) = 2/sqrt(c) * atanh( sqrt(c) * || -x (+) y || )
    """
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    norm = torch.norm(diff, dim=-1, keepdim=True)
    return (2.0 / sqrt_c) * torch.atanh(torch.clamp(sqrt_c * norm, max=1.0-EPS))

def exp_map(x: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential Map at x: Exp_x(v).
    Maps tangent vector v in T_x D^n to manifold.
    
    Exp_x(v) = x (+) ( tanh( sqrt(c)*lambda_x*|v| / 2 ) * v / (sqrt(c)*|v|) )
    """
    sqrt_c = c ** 0.5
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    
    # Avoid div by zero
    v_unit = v / (v_norm + MIN_NORM)
    
    lam = lambda_x(x, c)
    
    # coeff = tanh( sqrt(c) * lambda * |v| / 2 ) / sqrt(c)
    arg = sqrt_c * lam * v_norm / 2.0
    coeff = torch.tanh(arg) / sqrt_c
    
    # If v is zero, result is x
    # mask = v_norm < MIN_NORM
    
    u = coeff * v_unit
    
    res = mobius_add(x, u, c)
    return res

def log_map(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic Map at x: Log_x(y).
    Maps point y on manifold to tangent vector in T_x D^n.
    
    Log_x(y) = (2 / (sqrt(c)*lambda_x)) * atanh( sqrt(c) * |-x (+) y| ) * ( -x (+) y ) / |-x (+) y|
    """
    sqrt_c = c ** 0.5
    sub = mobius_add(-x, y, c)
    sub_norm = torch.norm(sub, dim=-1, keepdim=True)
    
    sub_unit = sub / (sub_norm + MIN_NORM)
    
    lam = lambda_x(x, c)
    
    # term = atanh( sqrt(c) * |sub| )
    term = torch.atanh(torch.clamp(sqrt_c * sub_norm, max=1.0-EPS))
    
    coeff = (2.0 / (sqrt_c * lam)) * term
    
    return coeff * sub_unit

def gyration(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Gyr[u, v]w = -(u (+) v) (+) (u (+) (v (+) w))
    Operator that rotates w due to Thomas Precession from u to v.
    """
    a = mobius_add(u, v, c)
    b = mobius_add(v, w, c)
    d = mobius_add(u, b, c)
    return mobius_add(-a, d, c)

def parallel_transport(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Parallel transport vector v from T_x to T_y along geodesic x->y.
    
    Formula: P_{x->y}(v) = Gyr[y, -x] v
    
    But note: v is in T_x. The formula above usually applies to vectors 'rooted' at origin if using Mobius Gyrogroups.
    However, strictly speaking, PT in Mobius Gyrovector space is P_{x->y}(v) = Gyr[y, -x] v.
    Wait, Gyr[u, v] is a rotation.
    
    Correct property: Log_y(Exp_x(v)) ? No.
    
    The standard formula:
    P_{0->x}(v) = v (Euclidean parallel transport is correct for 0->x radial line? NO)
    Actually, 0->x is a straight line in Poincare ball.
    
    But x->y is general.
    P_{x->y} = P_{0->y} * Gyr[y, -x] * P_{x->0}
    Since P_{0->x} is Identity (Gyrovector space property?), 
    P_{x->y}(v) = Gyr[y, -x] v
    
    Ref: Ungar, "Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity"
    """
    return gyration(y, -x, v, c)

def check_boundary(x: torch.Tensor, margin: float = 1e-4) -> torch.Tensor:
    """
    Ensure x stays within unit ball.
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    mask = norm > (1.0 - margin)
    if mask.any():
        x_clamped = x / norm * (1.0 - margin)
        # Only replace violating
        x = torch.where(mask, x_clamped, x)
    return x

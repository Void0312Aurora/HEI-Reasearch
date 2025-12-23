
"""
PyTorch Core Utilities for Hyperbolic Geometry (SO(1, n)).
=========================================================

Provides GPU-accelerated primitives for:
1. Lie Algebra operations (exp_so1n).
2. Manifold constraints (renormalize_so1n).
3. Metric tensors.
"""

import torch
import torch.nn as nn

def minkowski_metric_torch(dim: int, device: torch.device) -> torch.Tensor:
    """Returns J = diag(-1, 1, ..., 1) of shape (dim, dim)."""
    J = torch.eye(dim, device=device)
    J[0, 0] = -1.0
    return J

def exp_so1n_torch(X: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """
    Batched Exponential map for so(1, n).
    G = exp(dt * X)
    
    Args:
        X: (N, dim, dim) Algebra elements
        dt: float
    Returns:
        G: (N, dim, dim) Group elements
    """
    # torch.linalg.matrix_exp handles batches efficiently
    return torch.linalg.matrix_exp(X * dt)

def project_to_algebra_torch(M: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """
    Project matrix M onto so(1, n).
    X = 0.5 * (M - J M^T J)
    """
    # M: (N, dim, dim) or (dim, dim)
    # J: (dim, dim)
    
    # Broadcast J if needed, but matmul handles (dim, dim) vs (N, dim, dim) via broadcasting?
    # No, torch.matmul(A, B) if A is (N, D, D) and B is (D, D) requires B to be (D, D) or broadcast manually?
    # PyTorch broadcasting rules for matmul:
    # If A is (..., n, m) and B is (..., m, p), it works.
    # J is (dim, dim). MT is (N, dim, dim).
    # J @ MT @ J -> (dim, dim) @ (N, dim, dim) -> Mismatch?
    # Need J to allow broadcasting.
    
    MT = M.transpose(-1, -2)
    
    # J @ MT @ J
    # We can rely on torch's broadcasting if we reshape J.
    # But explicitly:
    term2 = J @ MT @ J # works if J is (dim, dim) ? 
    # torch.matmul((D,D) , (N,D,D)) -> (N,D,D)? Let's verify or be safe.
    # Typically one expands J to (1, D, D).
    
    # Let's check implicit broadcast. usually matmul matches last 2 dims.
    # If J is (D,D), MT is (N,D,D), 'J @ MT' might act as 'J broadcasted'.
    # Actually standard matmul might not broadcast the left operand if rank differs like that?
    # Safe way:
    J_batch = J.unsqueeze(0) if M.ndim == 3 else J
    term2 = J_batch @ MT @ J_batch
    
    return 0.5 * (M - term2)

def project_to_tangent_torch(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project vector v onto tangent space T_x H^n.
    v_proj = v + <x, v>_L * x
    (Metric signature (-1, 1,...))
    """
    # x, v: (N, dim)
    dim = x.shape[-1]
    # Minkowski Inner Product
    # <x, v> = -x0*v0 + xi*vi
    # Efficient: x * v * [-1, 1...]
    
    J_vec = torch.ones(dim, device=x.device)
    J_vec[0] = -1.0
    
    inner = torch.sum(x * v * J_vec, dim=-1, keepdim=True) # (N, 1)
    
    # x is on H^n implies <x, x> = -1.
    # Generic proj formula: v - <v, n> n / <n, n>
    # Here n = x. <x, x> = -1.
    # v_proj = v - <v, x> x / (-1) = v + <v, x> x
    
    return v + inner * x

def renormalize_so1n_torch(G: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Batched Hyperbolic Gram-Schmidt.
    G: (N, dim, dim)
    Returns: G_new, max_diff
    """
    G_new = G.clone()
    G_old = G # Reference for diff
    
    dim = G.shape[-1]
    N = G.shape[0]
    device = G.device
    
    J_vec = torch.ones(dim, device=device)
    J_vec[0] = -1.0
    
    def mink_inner(a, b):
        return torch.sum(a * b * J_vec, dim=-1)
    
    basis = []
    
    # Col 0 (Timelike)
    v0 = G_new[..., 0]
    norm_sq = mink_inner(v0, v0)
    # scale = 1 / sqrt(|norm_sq|)
    scale0 = torch.rsqrt(torch.abs(norm_sq) + 1e-15)
    v0 = v0 * scale0.unsqueeze(-1)
    G_new[..., 0] = v0
    basis.append(v0)
    
    # Col 1..dim (Spacelike)
    for k in range(1, dim):
        vk = G_new[..., k]
        
        # GS Projection
        for j in range(k):
            ej = basis[j]
            factor = mink_inner(vk, ej) # (N,)
            if j == 0:
                factor = -factor # <ej, ej> = -1
            
            vk = vk - factor.unsqueeze(-1) * ej
            
        # Normalize
        norm_sq = mink_inner(vk, vk)
        scale = torch.rsqrt(torch.abs(norm_sq) + 1e-15)
        vk = vk * scale.unsqueeze(-1)
        
        G_new[..., k] = vk
        basis.append(vk)
        
    # Diff
    diff = torch.norm(G_new - G_old, dim=(1,2))
    max_diff = torch.max(diff).item()
    
    return G_new, max_diff

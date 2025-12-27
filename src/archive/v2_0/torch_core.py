
"""
PyTorch Core Utilities for Hyperbolic Geometry (SO(1, n)).
=========================================================

Provides GPU-accelerated primitives for:
1. Lie Algebra operations (exp_so1n).
2. Manifold constraints (renormalize_so1n).
3. Metric tensors.

Ported from hei_n for Aurora self-containment.
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
    MT = M.transpose(-1, -2)
    J_batch = J.unsqueeze(0) if M.ndim == 3 else J
    term2 = J_batch @ MT @ J_batch
    return 0.5 * (M - term2)

def project_to_tangent_torch(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project vector v onto tangent space T_x H^n.
    v_proj = v + <x, v>_L * x
    """
    dim = x.shape[-1]
    J_vec = torch.ones(dim, device=x.device)
    J_vec[0] = -1.0
    
    inner = torch.sum(x * v * J_vec, dim=-1, keepdim=True) # (N, 1)
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

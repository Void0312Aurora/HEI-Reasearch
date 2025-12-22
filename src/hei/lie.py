"""sl(2, R) helpers: basis maps, coadjoint action, and Möbius flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray



def vec_to_matrix(xi: ArrayLike) -> NDArray[np.float64]:
    """Map (..., 3) to sl(2) matrix (..., 2, 2) [[u, v], [w, -u]]."""
    xi_arr = np.asarray(xi, dtype=float)
    u = xi_arr[..., 0]
    v = xi_arr[..., 1]
    w = xi_arr[..., 2]
    
    # Construct matrix array using indexing tricks or stack
    # Shape: (..., 2, 2)
    zeros = np.zeros_like(u)
    # Row 1: [u, v]
    # Row 2: [w, -u]
    # There is no simple recursive constructor, stick to manual stack
    row1 = np.stack([u, v], axis=-1)
    row2 = np.stack([w, -u], axis=-1)
    return np.stack([row1, row2], axis=-2)


def matrix_to_vec(mat: ArrayLike) -> NDArray[np.float64]:
    """Inverse of vec_to_matrix. Map (..., 2, 2) -> (..., 3)."""
    m = np.asarray(mat, dtype=float)
    return np.stack([m[..., 0, 0], m[..., 0, 1], m[..., 1, 0]], axis=-1)


def ad_star(xi: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """
    Coadjoint action ad^*_xi(m) under trace pairing.
    Supports batch broadcasting if vec_to_matrix supports it.
    """
    X = vec_to_matrix(xi)
    M = vec_to_matrix(m)
    comm = M @ X - X @ M
    return matrix_to_vec(comm)


def moebius_flow(xi: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Compute z_dot = v + 2 u z - w z^2. Supports batch (N, 3) and (N,)."""
    xi_arr = np.asarray(xi, dtype=float)
    u = xi_arr[..., 0]
    v = xi_arr[..., 1]
    w = xi_arr[..., 2]
    
    zc = np.asarray(z, dtype=np.complex128)
    return v + 2.0 * u * zc - w * zc * zc


def exp_sl2(xi: ArrayLike, dt: float = 1.0) -> NDArray[np.float64]:
    """
    Exponential map for sl(2) elements. Supports batch input (N, 3).
    Returns (N, 2, 2).
    """
    xi_arr = np.asarray(xi, dtype=float)
    u = xi_arr[..., 0]
    v = xi_arr[..., 1]
    w = xi_arr[..., 2]
    
    A = vec_to_matrix(xi_arr)
    lam2 = u * u + v * w
    k = np.sqrt(np.abs(lam2))
    s = dt * k
    
    # We need to handle small angle approx per element
    # Vectorized 'where' approach
    
    # Prepare result container
    result = np.zeros_like(A)
    # Identity matrix (broadcastable)
    n_dim = A.ndim
    # Identity is last 2x2.
    I = np.eye(2, dtype=float)
    
    # Mask for small s
    mask_small = np.abs(s) < 1e-8
    
    # Small angle case: I + dt A + 0.5 dt^2 A^2
    if np.any(mask_small):
        A_small = A[mask_small]
        I_small = np.eye(2)[np.newaxis, ...] if A_small.ndim == 3 else np.eye(2)
        # Note: A[mask_small] flattens the batch dims into 1D batch.
        res_small = I_small + dt * A_small + 0.5 * (dt * dt) * (A_small @ A_small)
        result[mask_small] = res_small
        
    # Standard case
    mask_normal = ~mask_small
    if np.any(mask_normal):
        s_norm = np.clip(s[mask_normal], -50.0, 50.0)
        k_norm = k[mask_normal] + 1e-12
        lam2_norm = lam2[mask_normal]
        A_norm = A[mask_normal]
        
        # Hyperbolic case (lam2 > 0)
        mask_hyp = lam2_norm > 0
        mask_elli = ~mask_hyp
        
        # We need to combine masks carefully or just calculate both and mixing
        # Let's compute coefficients
        ch = np.zeros_like(s_norm)
        sh_over_k = np.zeros_like(s_norm)
        
        # Hyp
        if np.any(mask_hyp):
            s_hyp = s_norm[mask_hyp]
            k_hyp = k_norm[mask_hyp]
            ch[mask_hyp] = np.cosh(s_hyp)
            sh_over_k[mask_hyp] = np.sinh(s_hyp) / k_hyp
            
        # Elliptic
        if np.any(mask_elli):
            s_elli = s_norm[mask_elli]
            k_elli = k_norm[mask_elli]
            ch[mask_elli] = np.cos(s_elli)
            sh_over_k[mask_elli] = np.sin(s_elli) / k_elli
            
        # Compose matrices: ch * I + sh/k * A
        # Broadcast ch against (2,2)
        I_stack = np.eye(2)[np.newaxis, ...]
        
        res_normal = ch[:, np.newaxis, np.newaxis] * I_stack + sh_over_k[:, np.newaxis, np.newaxis] * A_norm
        result[mask_normal] = res_normal
        
    return result


def mobius_action_matrix(g: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """
    Apply SL(2) matrix g = [[a, b], [c, d]] to complex points z via Möbius action.
    Supports broadcasting: g (N, 2, 2), z (N,).
    """
    g_arr = np.asarray(g, dtype=float)
    zc = np.asarray(z, dtype=np.complex128)
    
    # Broadcast shapes if needed?
    # g[..., 0, 0] has shape (N,)
    a = g_arr[..., 0, 0]
    b = g_arr[..., 0, 1]
    c = g_arr[..., 1, 0]
    d = g_arr[..., 1, 1]
    
    denom = c * zc + d
    return (a * zc + b) / denom

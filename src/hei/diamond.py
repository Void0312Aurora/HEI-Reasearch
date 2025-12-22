"""Diamond operator utilities on the upper half-plane."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .lie import matrix_to_vec


def diamond_torque_matrix(z: complex | ArrayLike, f: complex | ArrayLike) -> NDArray[np.float64]:
    """
    Compute single-point torque matrix J in sl(2)^* from position z and force f.
    Supports batch inputs (N,) -> (N, 2, 2).
    """
    zc = np.asarray(z, dtype=np.complex128)
    fc = np.asarray(f, dtype=np.complex128)
    
    # Lever arm clipping
    MAX_LEVER_ARM = 100.0
    z_mag = np.abs(zc)
    
    # Vectorized clipping
    mask = z_mag > MAX_LEVER_ARM
    if np.any(mask):
        # Avoid in-place modification of input if it was array
        if mask.ndim > 0:
            zc = zc.copy()
            zc[mask] *= (MAX_LEVER_ARM / z_mag[mask])
        else: # Scalar
             zc *= (MAX_LEVER_ARM / z_mag)

    re_zf = np.real(zc * np.conj(fc))
    re_f = np.real(fc)
    re_fz2 = np.real(np.conj(fc) * (zc * zc))

    # Construct batch matrix
    # [[re_zf, -0.5 * re_fz2], [0.5 * re_f, -re_zf]]
    
    row1 = np.stack([re_zf, -0.5 * re_fz2], axis=-1)
    row2 = np.stack([0.5 * re_f, -re_zf], axis=-1)
    return np.stack([row1, row2], axis=-2)


def diamond_torque_vec(z: complex | ArrayLike, f: complex | ArrayLike) -> NDArray[np.float64]:
    """Return torque in (u, v, w) vector form. Supports batch."""
    return matrix_to_vec(diamond_torque_matrix(z, f))


def aggregate_torque(z_uhp: ArrayLike, forces: ArrayLike, sum_torque: bool = True) -> NDArray[np.float64]:
    """
    Compute torques over all points.
    If sum_torque is True, returns (3,).
    If sum_torque is False, returns (N, 3).
    """
    # Simply call vectorized
    torques = diamond_torque_vec(z_uhp, forces)
    if sum_torque:
        if torques.ndim == 2:
            return np.sum(torques, axis=0)
        return torques # If torques is already (3,) (e.g., single point input)
    return torques



def diamond_torque_hyperboloid_batch(h: ArrayLike, f_h: ArrayLike) -> NDArray[np.float64]:
    """
    Vectorized implementation of diamond_torque_hyperboloid.
    Returns (N, 3).
    """
    h_arr = np.asarray(h, dtype=np.float64)
    f_arr = np.asarray(f_h, dtype=np.float64)
    
    # 提取坐标分量
    X = h_arr[..., 0]
    Y = h_arr[..., 1]
    T = h_arr[..., 2]
    
    fX = f_arr[..., 0]
    fY = f_arr[..., 1]
    fT = f_arr[..., 2]
    
    denom = 1.0 + T
    denom2 = denom * denom
    
    # f_uhp = (fX + i*fY)/(1+T) - (X+iY)*fT/(1+T)^2
    term1 = (fX + 1j * fY) / denom
    z_numerator = X + 1j * Y
    term2 = z_numerator * (fT / denom2)
    
    f_uhp = term1 - term2
    
    # z = (X + iY) / (1 + T)
    z_uhp = z_numerator / denom
    
    return diamond_torque_vec(z_uhp, f_uhp)

# Legacy alias
diamond_torque_hyperboloid = diamond_torque_hyperboloid_batch


def aggregate_torque_hyperboloid(h: ArrayLike, f_h: ArrayLike, sum_torque: bool = True) -> NDArray[np.float64]:
    """Hyperboloid version. Wraps diamond_torque_hyperboloid + sum options."""
    # This calls diamond_torque_hyperboloid which handles conversion
    torques = diamond_torque_hyperboloid_batch(h, f_h)
    if sum_torque:
        if torques.ndim == 2:
            return np.sum(torques, axis=0)
    return torques
    
    n = h_arr.shape[0]
    
    if h_arr.shape != f_arr.shape:
        raise ValueError(f"h and f_h must have same shape, got {h_arr.shape} vs {f_arr.shape}")
    
    # 逐点计算 diamond 力矩
    torques = diamond_torque_hyperboloid(h_arr, f_arr)  # (N, 3)
    
    # 加权聚合
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) != n:
            raise ValueError(f"weights length {len(w)} must match number of points {n}")
        return np.sum(torques * w[:, None], axis=0)
    else:
        # 无权重时直接求和（与 aggregate_torque 保持一致）
        return np.sum(torques, axis=0)

def diamond_torque_hyperboloid_batch(h: ArrayLike, f_h: ArrayLike) -> NDArray[np.float64]:
    """
    Vectorized implementation of diamond_torque_hyperboloid.
    Returns (N, 3).
    """
    h_arr = np.asarray(h, dtype=np.float64)
    f_arr = np.asarray(f_h, dtype=np.float64)
    
    # 提取坐标分量
    X = h_arr[..., 0]
    Y = h_arr[..., 1]
    T = h_arr[..., 2]
    
    fX = f_arr[..., 0]
    fY = f_arr[..., 1]
    fT = f_arr[..., 2]
    
    denom = 1.0 + T
    denom2 = denom * denom
    
    # f_uhp = (fX + i*fY)/(1+T) - (X+iY)*fT/(1+T)^2
    term1 = (fX + 1j * fY) / denom
    z_numerator = X + 1j * Y
    term2 = z_numerator * (fT / denom2)
    
    f_uhp = term1 - term2
    
    # z = (X + iY) / (1 + T)
    z_uhp = z_numerator / denom
    
    return diamond_torque_vec(z_uhp, f_uhp)

# Legacy alias
diamond_torque_hyperboloid = diamond_torque_hyperboloid_batch


def aggregate_torque_hyperboloid(h: ArrayLike, f_h: ArrayLike, sum_torque: bool = True) -> NDArray[np.float64]:
    """Hyperboloid version. Wraps diamond_torque_hyperboloid + sum options."""
    # This calls diamond_torque_hyperboloid which handles conversion
    torques = diamond_torque_hyperboloid_batch(h, f_h)
    if sum_torque:
        if torques.ndim == 2:
            return np.sum(torques, axis=0)
    return torques

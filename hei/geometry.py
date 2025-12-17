"""Hyperbolic geometry utilities for Poincaré disk, upper half-plane, and Hyperboloid."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# 导入 Hyperboloid 模块的核心函数
from .hyperboloid import (
    disk_to_hyperboloid,
    hyperboloid_to_disk,
    hyperbolic_distance_hyperboloid,
    hyperbolic_distance_grad_hyperboloid,
    project_to_hyperboloid,
    gamma_hyperboloid,
    compute_centroid_hyperboloid,
    sl2_to_so21,
    lorentz_action,
    minkowski_inner,
)

# 重新导出以保持 API 兼容性
__all__ = [
    # 原有函数
    "clamp_disk_radius",
    "clamp_uhp",
    "cayley_disk_to_uhp",
    "cayley_uhp_to_disk",
    "uhp_distance_and_grad",
    "poincare_metric_factor_disk",
    "sample_disk_hyperbolic",
    "sample_uhp_from_disk",
    "hyperbolic_karcher_mean_disk",
    # Hyperboloid 函数
    "disk_to_hyperboloid",
    "hyperboloid_to_disk",
    "hyperbolic_distance_hyperboloid",
    "hyperbolic_distance_grad_hyperboloid",
    "project_to_hyperboloid",
    "gamma_hyperboloid",
    "compute_centroid_hyperboloid",
    "sl2_to_so21",
    "lorentz_action",
    "minkowski_inner",
    # 便捷函数
    "uhp_to_hyperboloid",
    "hyperboloid_to_uhp",
]


def uhp_to_hyperboloid(z_uhp: ArrayLike) -> NDArray[np.float64]:
    """上半平面 → Hyperboloid（通过圆盘中转）"""
    z_disk = cayley_uhp_to_disk(z_uhp)
    return disk_to_hyperboloid(z_disk)


def hyperboloid_to_uhp(h: ArrayLike) -> NDArray[np.complex128]:
    """Hyperboloid → 上半平面（通过圆盘中转）"""
    z_disk = hyperboloid_to_disk(h)
    return cayley_disk_to_uhp(z_disk)


def clamp_disk_radius(z_disk: ArrayLike, eps: float = 1e-6) -> NDArray[np.complex128]:
    """
    Clamp disk points to stay strictly inside the unit circle (avoid Cayley singularity at z=1).
    """
    z = np.asarray(z_disk, dtype=np.complex128)
    r = np.abs(z)
    r_safe = np.minimum(r, 1.0 - eps)
    # avoid divide by zero: when r == 0 keep angle 0
    scale = np.divide(r_safe, np.maximum(r, eps), out=np.ones_like(r, dtype=float), where=r > 0)
    return z * scale


def clamp_uhp(
    z_uhp: ArrayLike,
    min_im: float = 1e-6,
    max_re: float = 1e6,
    max_im: float = 1e6,
) -> NDArray[np.complex128]:
    """
    Clamp UHP points to a safe band to reduce overflow/underflow from Cayley singularities.
    """
    z = np.asarray(z_uhp, dtype=np.complex128)
    re = np.clip(np.real(z), -max_re, max_re)
    im = np.clip(np.imag(z), min_im, max_im)
    return re + 1j * im


def cayley_disk_to_uhp(z_disk: ArrayLike) -> NDArray[np.complex128]:
    """Map points from the Poincaré disk to the upper half-plane with safety clamping."""
    # Stronger radius clamp to avoid |z| -> 1 singularity
    z = clamp_disk_radius(z_disk, eps=1e-4)
    denom = 1 - z
    denom_mag = np.abs(denom)
    safe_mask = denom_mag > 1e-4
    # push unsafe denominators away from zero
    denom = np.where(safe_mask, denom, denom * (1e-4 / (denom_mag + 1e-12)))
    uhp = 1j * (1 + z) / denom
    return clamp_uhp(uhp, max_re=1e5, max_im=1e5)


def cayley_uhp_to_disk(z_uhp: ArrayLike) -> NDArray[np.complex128]:
    """Map points from the upper half-plane back to the Poincaré disk with safety clamping."""
    z = clamp_uhp(z_uhp)
    disk = (z - 1j) / (z + 1j)
    return clamp_disk_radius(disk)


def uhp_distance_and_grad(z: complex, c: complex, eps: float = 1e-9) -> tuple[float, complex]:
    """
    Hyperbolic distance on UHP and its gradient w.r.t z (real + i imag).

    d = arcosh(1 + |z-c|^2 / (2 Im z Im c)).
    Returns (distance, grad_z) where grad_z = dd/dx + i dd/dy.
    """
    x, y = float(np.real(z)), float(np.imag(z))
    u, v = float(np.real(c)), float(np.imag(c))
    y = max(y, eps)
    v = max(v, eps)

    dx = x - u
    dy = y - v
    denom = max(2.0 * y * v, eps)
    num = dx * dx + dy * dy
    A = num / denom
    uval = max(1.0 + A, 1.0 + eps)
    d = float(np.arccosh(uval))
    # derivative of acosh(u) = 1/sqrt(u^2 - 1) = 1/sqrt(A(A+2))
    common = 1.0 / np.sqrt(max(A * (A + 2.0), eps))
    dA_dx = dx / (y * v)
    dA_dy = dy / (y * v) - num / (2.0 * y * y * v)
    grad = complex(common * dA_dx, common * dA_dy)
    return d, grad


def poincare_metric_factor_disk(z_disk: ArrayLike) -> NDArray[np.float64]:
    """Conformal factor lambda^2 for the disk metric g = lambda^2 * I."""
    z = np.asarray(z_disk, dtype=np.complex128)
    r2 = np.abs(z) ** 2
    return 4.0 / (1.0 - r2) ** 2


def sample_disk_hyperbolic(
    n: int,
    max_rho: float = 3.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """
    Sample n points in the disk with respect to the hyperbolic area measure,
    truncated at geodesic radius max_rho.

    CDF for geodesic radius rho: (cosh(rho) - 1) / (cosh(max_rho) - 1).
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n)
    rho = np.arccosh(1 + u * (np.cosh(max_rho) - 1))
    r = np.tanh(0.5 * rho)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    return r * np.exp(1j * theta)


def sample_uhp_from_disk(
    n: int,
    max_rho: float = 3.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Sample disk points and map them to the upper half-plane."""
    z_disk = sample_disk_hyperbolic(n=n, max_rho=max_rho, rng=rng)
    return cayley_disk_to_uhp(z_disk)


def hyperbolic_karcher_mean_disk(
    z_disk: ArrayLike,
    max_iter: int = 64,
    tol: float = 1e-6,
    step_size: float = 0.4,
) -> complex:
    """
    Approximate Karcher mean on the Poincaré disk by gradient descent on UHP.
    """
    z_arr = np.asarray(z_disk, dtype=np.complex128).ravel()
    if z_arr.size == 0:
        return 0.0 + 0.0j
    z_uhp = cayley_disk_to_uhp(z_arr)
    mu = np.mean(z_uhp)
    mu = complex(np.real(mu), max(np.imag(mu), 1e-4))
    n = z_uhp.size

    for _ in range(max_iter):
        grad_sum = 0.0 + 0.0j
        for zi in z_uhp:
            d, grad = uhp_distance_and_grad(mu, zi)
            grad_sum += 2.0 * d * grad
        grad_norm = abs(grad_sum) / max(n, 1)
        if grad_norm < tol:
            break
        mu = mu - step_size * grad_sum / max(n, 1)
        mu = complex(np.real(mu), max(np.imag(mu), 1e-4))
    return complex(cayley_uhp_to_disk(mu))

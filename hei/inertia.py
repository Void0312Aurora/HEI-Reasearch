"""Inertia handling for semi-direct product CCD dynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def constant_inertia(diag: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> NDArray[np.float64]:
    """Return a diagonal inertia matrix in the sl(2) basis (u, v, w)."""
    d = np.array(diag, dtype=float)
    return np.diag(d)


def locked_inertia_uhp(z_uhp: ArrayLike, weights: ArrayLike | None = None) -> NDArray[np.float64]:
    """
    Compute regularized inertia with Relative Vacuum and Generator Saturation.
    """
    z_arr = np.asarray(z_uhp, dtype=np.complex128).ravel()

    # 1. 基础真空惯性 (绝对底噪，防止空集报错)
    I_vac_base = np.eye(3, dtype=float) * 1e-6
    
    if z_arr.size == 0:
        return I_vac_base

    if weights is None:
        w_arr = np.ones_like(z_arr, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()

    I_matter = np.zeros((3, 3), dtype=float)
    
    # [关键修复 1] 生成元饱和阈值
    # 限制单个点对整体惯性的最大贡献。
    # 物理意义：边缘概念的模糊性限制了其动力学权重。
    GENERATOR_CLIP = 5.0 
    
    # 坐标软截断 (保留原有的坐标限制，但可以稍微放宽)
    COORD_CLIP = 50.0 

    for zc, mass in zip(z_arr, w_arr):
        scale = mass
        
        # 坐标预处理
        z_mag = abs(zc)
        if z_mag > COORD_CLIP:
            z_eff = zc * (COORD_CLIP / z_mag)
        else:
            z_eff = zc

        # 计算原始生成元
        raw_u = 2.0 * z_eff
        raw_v = 1.0 + 0j
        raw_w = -z_eff * z_eff
        
        # [关键修复 2] 饱和函数
        def saturate(val):
            mag = abs(val)
            if mag > GENERATOR_CLIP:
                return val * (GENERATOR_CLIP / mag)
            return val

        a_u = saturate(raw_u)
        a_v = raw_v # 1.0
        a_w = saturate(raw_w) # 这里的平方项被压制住了

        # 累加物质惯性
        I_matter[0, 0] += scale * np.abs(a_u) ** 2
        I_matter[1, 1] += scale * np.abs(a_v) ** 2
        I_matter[2, 2] += scale * np.abs(a_w) ** 2

        I_uv = scale * np.real(a_u * np.conj(a_v))
        I_uw = scale * np.real(a_u * np.conj(a_w))
        I_vw = scale * np.real(a_v * np.conj(a_w))
        
        I_matter[0, 1] += I_uv; I_matter[1, 0] += I_uv
        I_matter[0, 2] += I_uw; I_matter[2, 0] += I_uw
        I_matter[1, 2] += I_vw; I_matter[2, 1] += I_vw

    # [关键修复 3] 相对真空正则化 (Relative Tikhonov)
    # 确保条件数不超过 ~100
    trace_val = np.trace(I_matter)
    avg_mass = trace_val / 3.0
    if avg_mass < 1e-9: avg_mass = 1.0
    
    REL_EPS = 1e-2 
    I_vac_rel = np.eye(3, dtype=float) * (avg_mass * REL_EPS)
    
    return I_matter + I_vac_rel + I_vac_base


def _compute_single_point_inertia_matrices(
    z: complex, scale: float, coord_clip: float, generator_clip: float
) -> tuple[float, float, float, float, float, float]:
    """Helper to compute raw inertia components for a single point."""
    z_mag = abs(z)
    if z_mag > coord_clip:
        z_eff = z * (coord_clip / z_mag)
    else:
        z_eff = z

    raw_u = 2.0 * z_eff
    raw_v = 1.0 + 0j
    raw_w = -z_eff * z_eff

    def saturate(val: complex) -> complex:
        mag = abs(val)
        if mag > generator_clip:
            return val * (generator_clip / mag)
        return val

    a_u = saturate(raw_u)
    a_v = raw_v
    a_w = saturate(raw_w)

    I_uu = scale * np.abs(a_u) ** 2
    I_vv = scale * np.abs(a_v) ** 2
    I_ww = scale * np.abs(a_w) ** 2
    I_uv = scale * np.real(a_u * np.conj(a_v))
    I_uw = scale * np.real(a_u * np.conj(a_w))
    I_vw = scale * np.real(a_v * np.conj(a_w))

    return I_uu, I_vv, I_ww, I_uv, I_uw, I_vw


def compute_kinetic_energy_gradient(
    z_uhp: ArrayLike, xi: ArrayLike, weights: ArrayLike | None = None
) -> NDArray[np.complex128]:
    """
    Compute the gradient of kinetic energy K = 0.5 * xi^T I(z) xi w.r.t z.
    Returns a complex force vector 'f' such that Re(f * delta_z) is the work done.

    This force must be added to the Diamond operator to conserve energy in a
    variable-inertia system.
    """
    z_arr = np.asarray(z_uhp, dtype=np.complex128).ravel()
    xi_vec = np.asarray(xi, dtype=float)
    u, v, w = xi_vec

    if weights is None:
        w_arr = np.ones_like(z_arr, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()
        if w_arr.shape != z_arr.shape:
            raise ValueError("weights must match z_uhp shape")

    COORD_CLIP = 50.0
    GENERATOR_CLIP = 5.0

    forces = np.zeros_like(z_arr, dtype=np.complex128)
    eps = 1e-5

    def local_kinetic(z_probe: complex, mass: float) -> float:
        vals = _compute_single_point_inertia_matrices(z_probe, mass, COORD_CLIP, GENERATOR_CLIP)
        I00, I11, I22, I01, I02, I12 = vals
        return 0.5 * (
            I00 * u * u
            + I11 * v * v
            + I22 * w * w
            + 2.0 * (I01 * u * v + I02 * u * w + I12 * v * w)
        )

    for i in range(z_arr.size):
        zc = z_arr[i]
        mass = w_arr[i]

        K_real_plus = local_kinetic(zc + eps, mass)
        K_real_minus = local_kinetic(zc - eps, mass)
        dK_dx = (K_real_plus - K_real_minus) / (2 * eps)

        K_imag_plus = local_kinetic(zc + 1j * eps, mass)
        K_imag_minus = local_kinetic(zc - 1j * eps, mass)
        dK_dy = (K_imag_plus - K_imag_minus) / (2 * eps)

        forces[i] = complex(dK_dx, dK_dy)

    return forces.reshape(z_uhp.shape)


def apply_inertia(I: ArrayLike, xi: ArrayLike) -> NDArray[np.float64]:
    """Compute m = I xi."""
    return np.asarray(I, dtype=float) @ np.asarray(xi, dtype=float)


def invert_inertia(I: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """
    Compute xi = I^{-1} m with spectral regularization to limit condition number.

    Clamp eigenvalues to [lambda_min, lambda_max] and rebuild inverse; set very
    small eigenvalues to zero (pseudo-inverse) to avoid injecting spurious modes.
    """
    I_arr = np.asarray(I, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    vals, vecs = np.linalg.eigh(I_arr)
    LAMBDA_MIN = 1e-4
    LAMBDA_MAX = 1e6
    vals_clamped = np.clip(vals, LAMBDA_MIN, LAMBDA_MAX)
    max_val = float(vals_clamped.max())
    tol = max_val / 1e6  # relative tolerance for pseudo-inverse
    inv_vals = np.where(vals_clamped < tol, 0.0, 1.0 / vals_clamped)
    I_inv = (vecs * inv_vals) @ vecs.T
    return I_inv @ m_arr


def moment_map_covariance(
    z_uhp: ArrayLike, weights: ArrayLike | None = None, alpha: float = 0.2
) -> NDArray[np.float64]:
    """
    Approximate moment map using weighted second moments on the UHP.

    We use Re(z), Im(z) to build a covariance proxy:
    c_xx = sum w x^2, c_yy = sum w y^2, c_xy = sum w x y.
    Map to sl(2) diagonal as (u,v,w) inertia contribution.
    """
    z_arr = np.asarray(z_uhp, dtype=np.complex128)
    if weights is None:
        weights = np.ones_like(z_arr, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    if z_arr.shape != w_arr.shape:
        raise ValueError("weights must match z_uhp shape")
    x = np.real(z_arr)
    y = np.imag(z_arr)
    c_xx = np.sum(w_arr * x * x)
    c_yy = np.sum(w_arr * y * y)
    c_xy = np.sum(w_arr * x * y)
    # simple mapping: assign to u (scale), v (translation), w (inversion) axes
    return alpha * np.array([c_xy, c_xx, c_yy], dtype=float)


def state_dependent_inertia(
    base_I: ArrayLike,
    z_uhp: ArrayLike,
    weights: ArrayLike | None = None,
    alpha: float = 0.2,
) -> NDArray[np.float64]:
    """
    Combine a base inertia with a covariance-derived additive term.
    """
    I = np.asarray(base_I, dtype=float)
    moment = moment_map_covariance(z_uhp, weights=weights, alpha=alpha)
    return I + np.diag(moment)

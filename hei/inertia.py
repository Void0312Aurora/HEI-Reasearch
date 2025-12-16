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
    Compute locked inertia tensor on the upper half-plane from point set {z_k}.

    Uses kinetic energy 2T = sum_k m_k |v + 2u z_k - w z_k^2|^2 / y_k^2,
    yielding I(a) so that T = 0.5 * xi^T I xi for xi=(u,v,w).
    """
    z_arr = np.asarray(z_uhp, dtype=np.complex128).ravel()
    if z_arr.size == 0:
        return np.eye(3)
    if weights is None:
        w_arr = np.ones_like(z_arr, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()
        if w_arr.shape != z_arr.shape:
            raise ValueError("weights must match z_uhp shape")
    I = np.zeros((3, 3), dtype=float)
    MAX_INERTIA_SCALE = 1e14  # allow very large inertia near boundary
    for zc, mass in zip(z_arr, w_arr):
        y = np.imag(zc)
        y_safe = max(y, 1e-12)
        raw_scale = mass / (y_safe * y_safe)
        scale = min(raw_scale, MAX_INERTIA_SCALE)
        a_u = 2.0 * zc
        a_v = 1.0 + 0j
        a_w = -zc * zc
        # diagonals
        I[0, 0] += scale * np.abs(a_u) ** 2
        I[1, 1] += scale * np.abs(a_v) ** 2
        I[2, 2] += scale * np.abs(a_w) ** 2
        # off-diagonals: Re(a_i conj(a_j))
        I_uv = scale * np.real(a_u * np.conj(a_v))
        I_uw = scale * np.real(a_u * np.conj(a_w))
        I_vw = scale * np.real(a_v * np.conj(a_w))
        I[0, 1] += I_uv
        I[1, 0] += I_uv
        I[0, 2] += I_uw
        I[2, 0] += I_uw
        I[1, 2] += I_vw
        I[2, 1] += I_vw
    # base cognitive mass once
    I += 0.5 * np.eye(3)
    # Condition spectrum to avoid ill-conditioned inversion near the boundary.
    EIG_FLOOR = 1e-3
    EIG_CAP = 1e14
    vals, vecs = np.linalg.eigh(I)
    vals_clamped = np.clip(vals, EIG_FLOOR, EIG_CAP)
    I = (vecs * vals_clamped) @ vecs.T
    return I


def apply_inertia(I: ArrayLike, xi: ArrayLike) -> NDArray[np.float64]:
    """Compute m = I xi."""
    return np.asarray(I, dtype=float) @ np.asarray(xi, dtype=float)


def invert_inertia(I: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """Compute xi = I^{-1} m."""
    return np.linalg.solve(np.asarray(I, dtype=float), np.asarray(m, dtype=float))


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

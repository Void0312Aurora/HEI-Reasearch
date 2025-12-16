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
    Compute regularized inertia tensor with vacuum inertia and soft spatial clamp.

    Physics fixes:
    1) Vacuum inertia: add epsilon * I to guarantee lambda_min > 0.
    2) Spatial regularization: soft clamp coordinate contribution to avoid lambda_max blow-up.
    """
    z_arr = np.asarray(z_uhp, dtype=np.complex128).ravel()

    # 1) Vacuum inertia baseline
    EPSILON_INERTIA = 1e-4
    I = np.eye(3, dtype=float) * EPSILON_INERTIA

    if z_arr.size == 0:
        return I

    if weights is None:
        w_arr = np.ones_like(z_arr, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()
        if w_arr.shape != z_arr.shape:
            raise ValueError("weights must match z_uhp shape")

    COORD_CLIP = 50.0  # soft clamp scale for coordinates
    for zc, mass in zip(z_arr, w_arr):
        # covariant mass (already renormalized): no 1/y^2 factor
        scale = mass

        x, y = float(np.real(zc)), float(np.imag(zc))
        # smooth clamp via tanh and magnitude limiter
        x_eff = COORD_CLIP * np.tanh(x / COORD_CLIP)
        y_eff = COORD_CLIP * np.tanh(y / COORD_CLIP)
        z_eff = complex(x_eff, y_eff)
        z_mag = abs(zc)
        if z_mag > COORD_CLIP:
            z_eff = zc * (COORD_CLIP / z_mag)

        a_u = 2.0 * z_eff
        a_v = 1.0 + 0j
        a_w = -z_eff * z_eff

        I[0, 0] += scale * np.abs(a_u) ** 2
        I[1, 1] += scale * np.abs(a_v) ** 2
        I[2, 2] += scale * np.abs(a_w) ** 2

        I_uv = scale * np.real(a_u * np.conj(a_v))
        I_uw = scale * np.real(a_u * np.conj(a_w))
        I_vw = scale * np.real(a_v * np.conj(a_w))
        I[0, 1] += I_uv
        I[1, 0] += I_uv
        I[0, 2] += I_uw
        I[2, 0] += I_uw
        I[1, 2] += I_vw
        I[2, 1] += I_vw

    # Spectral conditioning to avoid extreme condition numbers
    EIG_FLOOR = 1e-6
    EIG_CAP = 1e6
    vals, vecs = np.linalg.eigh(I)
    vals_clamped = np.clip(vals, EIG_FLOOR, EIG_CAP)
    I = (vecs * vals_clamped) @ vecs.T
    return I


def apply_inertia(I: ArrayLike, xi: ArrayLike) -> NDArray[np.float64]:
    """Compute m = I xi."""
    return np.asarray(I, dtype=float) @ np.asarray(xi, dtype=float)


def invert_inertia(I: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """Compute xi = I^{-1} m with mild Tikhonov regularization."""
    I_reg = np.asarray(I, dtype=float) + 1e-8 * np.eye(3)
    return np.linalg.solve(I_reg, np.asarray(m, dtype=float))


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

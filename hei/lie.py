"""sl(2, R) helpers: basis maps, coadjoint action, and Möbius flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vec_to_matrix(xi: ArrayLike) -> NDArray[np.float64]:
    """Map (u, v, w) to sl(2) matrix [[u, v], [w, -u]]."""
    u, v, w = np.asarray(xi, dtype=float)
    return np.array([[u, v], [w, -u]], dtype=float)


def matrix_to_vec(mat: ArrayLike) -> NDArray[np.float64]:
    """Inverse of vec_to_matrix."""
    m = np.asarray(mat, dtype=float)
    return np.array([m[0, 0], m[0, 1], m[1, 0]], dtype=float)


def ad_star(xi: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """
    Coadjoint action ad^*_xi(m) under trace pairing.

    For matrix Lie algebras with <A, B> = Tr(A^T B), ad^*_xi(m) = [m, xi].
    """
    X = vec_to_matrix(xi)
    M = vec_to_matrix(m)
    comm = M @ X - X @ M
    return matrix_to_vec(comm)


def moebius_flow(xi: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """Compute z_dot = v + 2 u z - w z^2 given xi=(u, v, w) and complex z."""
    u, v, w = np.asarray(xi, dtype=float)
    zc = np.asarray(z, dtype=np.complex128)
    return v + 2.0 * u * zc - w * zc * zc


def exp_sl2(xi: ArrayLike, dt: float = 1.0) -> NDArray[np.float64]:
    """
    Exponential map for sl(2) elements (u, v, w) -> SL(2) matrix.

    Uses closed form for traceless 2x2 matrices:
    exp(A) = cosh(s) I + sinh(s)/s * A, where s = dt * sqrt(u^2 + v w).
    Falls back to 2nd order series when s is tiny.
    """
    u, v, w = np.asarray(xi, dtype=float)
    A = np.array([[u, v], [w, -u]], dtype=float)
    lam2 = u * u + v * w
    s = dt * np.sqrt(abs(lam2))
    s_clamped = min(s, 50.0)  # avoid overflow in cosh/sinh
    if s < 1e-8:
        # series: I + dt A + dt^2/2 A^2
        I = np.eye(2)
        return I + dt * A + 0.5 * (dt * dt) * (A @ A)
    if lam2 > 0:
        ch = np.cosh(s_clamped)
        sh_over_s = np.sinh(s_clamped) / (s_clamped + 1e-12)
    else:
        # imaginary eigenvalues -> use cos/sin with |s|
        s_real = dt * np.sqrt(-lam2)
        ch = np.cos(s_real)
        sh_over_s = np.sin(s_real) / (s_real + 1e-12)
    return ch * np.eye(2) + sh_over_s * A


def mobius_action_matrix(g: ArrayLike, z: ArrayLike) -> NDArray[np.complex128]:
    """
    Apply SL(2) matrix g = [[a, b], [c, d]] to complex points z via Möbius action.
    """
    g_arr = np.asarray(g, dtype=float)
    a, b = g_arr[0, 0], g_arr[0, 1]
    c, d = g_arr[1, 0], g_arr[1, 1]
    zc = np.asarray(z, dtype=np.complex128)
    denom = c * zc + d
    return (a * zc + b) / denom

import numpy as np

from HEI.src.hei.potential import GaussianWellsPotential


def test_gaussian_wells_is_attractive() -> None:
    pot = GaussianWellsPotential(centers=np.array([0.0 + 1.0j], dtype=np.complex128), weight=2.0, width=0.3)

    V_center = pot.potential(np.array([0.0 + 1.0j], dtype=np.complex128))
    V_far = pot.potential(np.array([10.0 + 10.0j], dtype=np.complex128))

    assert V_center < V_far


def test_gaussian_wells_force_descends_and_moves_toward_center() -> None:
    center = 0.25 + 1.3j
    pot = GaussianWellsPotential(centers=np.array([center], dtype=np.complex128), weight=1.0, width=0.4)

    z0 = 1.25 + 1.9j
    V0 = pot.potential(np.array([z0], dtype=np.complex128))
    grad0 = pot.gradient(np.array([z0], dtype=np.complex128))[0]
    force0 = -grad0

    step = 1e-2
    z1 = z0 + step * force0
    V1 = pot.potential(np.array([z1], dtype=np.complex128))

    assert V1 < V0
    assert abs(z1 - center) < abs(z0 - center)


def test_gaussian_wells_gradient_matches_finite_difference() -> None:
    pot = GaussianWellsPotential(centers=np.array([0.1 + 1.2j, -0.5 + 0.9j], dtype=np.complex128), weight=1.7, width=0.35)

    z0 = 0.3 + 1.7j
    grad = pot.gradient(np.array([z0], dtype=np.complex128))[0]

    eps = 1e-6
    Vx_plus = pot.potential(np.array([z0 + eps], dtype=np.complex128))
    Vx_minus = pot.potential(np.array([z0 - eps], dtype=np.complex128))
    dVdx = (Vx_plus - Vx_minus) / (2 * eps)

    Vy_plus = pot.potential(np.array([z0 + 1j * eps], dtype=np.complex128))
    Vy_minus = pot.potential(np.array([z0 - 1j * eps], dtype=np.complex128))
    dVdy = (Vy_plus - Vy_minus) / (2 * eps)

    grad_fd = dVdx + 1j * dVdy
    err = abs(grad - grad_fd)
    scale = max(1.0, abs(grad_fd))

    assert err / scale < 5e-4


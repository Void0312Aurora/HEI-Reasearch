"""Diamond operator utilities on the upper half-plane."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .lie import matrix_to_vec


def diamond_torque_matrix(z: complex, f: complex) -> NDArray[np.float64]:
    """
    Compute single-point torque matrix J in sl(2)^* from position z and force f.
    
    Coefficients consistent with docs/积分器.md and trace pairing:
    J = [[ Re(z \bar f)      , -0.5 Re(\bar f z^2) ],
         [ 0.5 Re(\bar f)    , -Re(z \bar f)       ]]
    """
    MAX_LEVER_ARM = 100.0
    z_mag = abs(z)
    if z_mag > MAX_LEVER_ARM:
        z = z * (MAX_LEVER_ARM / z_mag)

    zc = complex(z)
    fc = complex(f)
    re_zf = np.real(zc * np.conj(fc))
    re_f = np.real(fc)  # Note: Re(f) == Re(bar f)
    re_fz2 = np.real(np.conj(fc) * (zc * zc))

    return np.array(
        [
            [re_zf, -0.5 * re_fz2],
            [0.5 * re_f, -re_zf],
        ],
        dtype=float,
    )


def diamond_torque_vec(z: complex, f: complex) -> NDArray[np.float64]:
    """Return torque in (u, v, w) vector form."""
    return matrix_to_vec(diamond_torque_matrix(z, f))


def aggregate_torque(z_uhp: ArrayLike, forces: ArrayLike) -> NDArray[np.float64]:
    """Sum torques over all points."""
    z_arr = np.asarray(z_uhp, dtype=np.complex128)
    f_arr = np.asarray(forces, dtype=np.complex128)
    if z_arr.shape != f_arr.shape:
        raise ValueError("z_uhp and forces must share shape")
    total = np.zeros(3, dtype=float)
    for zc, fc in zip(z_arr.flat, f_arr.flat):
        total += diamond_torque_vec(zc, fc)
    return total

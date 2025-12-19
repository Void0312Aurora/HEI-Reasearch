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


def diamond_torque_hyperboloid(h: ArrayLike, f_h: ArrayLike) -> NDArray[np.float64]:
    """
    在 Hyperboloid 模型上计算 Diamond 力矩
    
    理论依据：
    ---------
    将 Hyperboloid 上的切向量（力）转换为 UHP 上的切向量，
    然后使用标准 diamond 算子计算力矩。
    
    坐标转换：
        Hyperboloid → UHP:  z = (X + iY) / (1 + T)
    
    切映射（Jacobian）：
        dz/dX = 1/(1+T)
        dz/dY = i/(1+T)
        dz/dT = -(X+iY)/(1+T)²
    
    因此，Hyperboloid 切向量 f_h = (fX, fY, fT) 对应的 UHP 切向量：
        f_uhp = (fX + i*fY)/(1+T) - (X+iY)*fT/(1+T)²
    
    参数：
        h: Hyperboloid 坐标 (3,) 或 (..., 3)，形式为 (X, Y, T)
        f_h: Hyperboloid 切向量（力）(3,) 或 (..., 3)
    
    返回：
        SL(2,R) 李代数中的力矩 (3,)，形式为 (u, v, w)
    
    物理意义：
        将 Hyperboloid 上的几何力（如惯性力）转换为 Möbius 变换的力矩
    """
    h_arr = np.asarray(h, dtype=np.float64)
    f_arr = np.asarray(f_h, dtype=np.float64)
    
    # 支持单点或批量处理
    if h_arr.ndim == 1:
        h_arr = h_arr.reshape(1, 3)
        f_arr = f_arr.reshape(1, 3)
        is_single = True
    else:
        is_single = False
    
    # 提取坐标分量
    X, Y, T = h_arr[..., 0], h_arr[..., 1], h_arr[..., 2]
    fX, fY, fT = f_arr[..., 0], f_arr[..., 1], f_arr[..., 2]
    
    # 计算 UHP 坐标
    factor = 1.0 / (1.0 + T + 1e-15)  # 防止除零
    z_uhp = (X + 1j * Y) * factor
    
    # 计算 UHP 切向量（通过链式法则）
    # f_uhp = df/dh · f_h = J(h) · f_h
    # 其中 J 是 Hyperboloid → UHP 的 Jacobian
    f_uhp = (fX + 1j * fY) * factor - z_uhp * fT * factor
    
    # 使用标准 UHP diamond 算子
    if is_single:
        return diamond_torque_vec(z_uhp[0], f_uhp[0])
    else:
        # 批量处理
        torques = np.zeros((len(h_arr), 3), dtype=float)
        for i in range(len(h_arr)):
            torques[i] = diamond_torque_vec(z_uhp[i], f_uhp[i])
        return torques


def aggregate_torque_hyperboloid(
    h: ArrayLike,
    f_h: ArrayLike,
    weights: ArrayLike | None = None
) -> NDArray[np.float64]:
    """
    聚合 Hyperboloid 上的力到总力矩
    
    对每个点的力计算 diamond 力矩，然后加权求和。
    
    参数：
        h: Hyperboloid 坐标 (N, 3)，每行为 (X, Y, T)
        f_h: Hyperboloid 力 (N, 3)，每行为 (fX, fY, fT)
        weights: 权重 (N,)，可选。默认无权重（直接求和）
    
    返回：
        总力矩 (3,)，形式为 (u, v, w) 在 sl(2,R) 李代数中
    
    使用场景：
        计算几何力（如惯性力）对系统的总力矩贡献
    
    注意：
        与 aggregate_torque() 保持一致，使用求和而非平均。
        力矩是广延量，应该对各点力矩求和。
    """
    h_arr = np.asarray(h, dtype=np.float64)
    f_arr = np.asarray(f_h, dtype=np.float64)
    
    if h_arr.ndim == 1:
        h_arr = h_arr.reshape(1, 3)
        f_arr = f_arr.reshape(1, 3)
    
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

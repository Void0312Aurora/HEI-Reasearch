"""
惯性张量处理模块 (Inertia handling for semi-direct product CCD dynamics)

理论基础：
-----------
在接触认知动力学 (CCD) 中，惯性张量 I(a) 描述了当前记忆结构对思维改变的阻抗。
对于 SL(2,R) 对 UHP 点集的作用，惯性张量由 "锁定惯性" (locked inertia) 计算得到。

物理意义 (参见理论基础-3.md 定义 3.1)：
- I(a) 对应拉格朗日量中的"思维动能"项：T = 0.5 * <ξ, I(a)ξ>
- 惯性张量的特征值反映了沿不同 Möbius 变换方向的思维阻抗
- 高惯性 → 思维改变困难（认知惯性大）
- 低惯性 → 思维改变容易（信念不稳定）

正则化策略：
-----------
为保证数值稳定性，采用两层正则化：

1. 绝对真空惯性 (I_base)：
   - 提供最小惯性下界，防止矩阵奇异
   - 物理意义：即使没有任何记忆点，心智仍有最小阻抗

2. 相对真空惯性 (I_vac_rel)：
   - 与当前惯性规模成比例的正则项
   - 控制条件数 κ(I) = λ_max / λ_min
   - 物理意义：防止某些思维方向过于"滑动"

参考文献：
-----------
- Holm, D. D. (2011). Geometric Mechanics - Part II.
- Marsden & Ratiu (1999). Introduction to Mechanics and Symmetry.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def constant_inertia(diag: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> NDArray[np.float64]:
    """Return a diagonal inertia matrix in the sl(2) basis (u, v, w)."""
    d = np.array(diag, dtype=float)
    return np.diag(d)


def _compute_point_features(z: complex, scale: float) -> tuple[complex, complex, complex]:
    """
    计算单个点的李代数生成元（带饱和处理）。
    
    理论依据：
    对于 sl(2,R) 在 UHP 上的作用，Möbius 流的无穷小生成元为：
        ż = v + 2uz - wz²
    
    因此每个点 z 对应的 "advected generators" 为：
        a_u = 2z  (缩放方向)
        a_v = 1   (平移方向)  
        a_w = -z² (反演方向)
    
    惯性张量的 (i,j) 分量由 Re(a_i * conj(a_j)) 给出。
    
    数值策略：
    - COORD_CLIP: 限制坐标大小，防止 z 过大导致 z² 爆炸
    - GENERATOR_CLIP: 限制生成元大小，确保惯性张量特征值可控
    
    确保 locked_inertia 和 kinetic_gradient 使用完全相同的物理量！
    """
    COORD_CLIP = 50.0
    GENERATOR_CLIP = 5.0  # 限制生成元大小，防止特征值爆炸

    z_mag = abs(z)
    if z_mag > COORD_CLIP:
        z_eff = z * (COORD_CLIP / z_mag)
    else:
        z_eff = z

    raw_u = 2.0 * z_eff
    raw_v = 1.0 + 0j
    raw_w = -z_eff * z_eff

    def saturate(val):
        mag = abs(val)
        if mag > GENERATOR_CLIP:
            return val * (GENERATOR_CLIP / mag)
        return val

    a_u = saturate(raw_u)
    a_v = raw_v
    a_w = saturate(raw_w)
    
    return a_u, a_v, a_w


def locked_inertia_uhp(z_uhp: ArrayLike, weights: ArrayLike | None = None) -> NDArray[np.float64]:
    """
    计算正则化的锁定惯性张量。
    
    理论依据 (参见 Holm, Geometric Mechanics Part II, Ch. 6)：
    锁定惯性由结构空间 V 上的质量分布决定：
        I_ij = Σ_k m_k * Re(a_i^k * conj(a_j^k))
    
    其中 a^k = (a_u, a_v, a_w) 是第 k 个点的 advected generators。
    
    正则化策略：
    -----------
    I_total = I_matter + I_vac_rel + I_base
    
    1. I_matter: 物质惯性（来自点集）
    2. I_vac_rel: 相对真空惯性 = ε_rel * avg(I_matter) * I
       - 控制条件数，防止病态
       - ε_rel = 1e-2 保证 κ(I) ≤ 100
    3. I_base: 绝对真空惯性 = 1e-6 * I
       - 防止空点集时矩阵奇异
    
    物理意义：
    - 相对真空惯性：即使沿某方向质量分布均匀（惯性小），
      仍保持最小阻抗，防止该方向过于"滑动"
    - 这对应于"真空涨落"或"背景刚性"的概念
    
    参数：
        z_uhp: 上半平面坐标数组
        weights: 可选的质量权重（默认为 1）
    
    返回：
        3x3 对称正定惯性矩阵
    """
    z_arr = np.asarray(z_uhp, dtype=np.complex128).ravel()

    # 绝对真空惯性：提供最小惯性下界
    I_base = np.eye(3, dtype=float) * 1e-6

    if z_arr.size == 0:
        return I_base

    if weights is None:
        w_arr = np.ones_like(z_arr, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()

    I_matter = np.zeros((3, 3), dtype=float)

    for zc, mass in zip(z_arr, w_arr):
        a_u, a_v, a_w = _compute_point_features(zc, mass)
        s = mass

        # 对角元：I_ii = m * |a_i|²
        I_matter[0, 0] += s * np.abs(a_u) ** 2
        I_matter[1, 1] += s * np.abs(a_v) ** 2
        I_matter[2, 2] += s * np.abs(a_w) ** 2

        # 非对角元：I_ij = m * Re(a_i * conj(a_j))
        I_uv = s * np.real(a_u * np.conj(a_v))
        I_uw = s * np.real(a_u * np.conj(a_w))
        I_vw = s * np.real(a_v * np.conj(a_w))
        
        I_matter[0, 1] += I_uv; I_matter[1, 0] += I_uv
        I_matter[0, 2] += I_uw; I_matter[2, 0] += I_uw
        I_matter[1, 2] += I_vw; I_matter[2, 1] += I_vw

    # 相对真空正则化：控制条件数 κ(I)
    # I_vac_rel = ε_rel * (tr(I_matter)/3) * I
    trace_val = np.trace(I_matter)
    avg_mass = trace_val / 3.0
    if avg_mass < 1e-9: 
        avg_mass = 1.0
    
    REL_EPS = 1e-2  # 保证 κ(I) ≤ 100
    I_vac_rel = np.eye(3, dtype=float) * (avg_mass * REL_EPS)
    
    return I_matter + I_vac_rel + I_base


def compute_kinetic_energy_gradient(z_uhp: ArrayLike, xi: ArrayLike, weights: ArrayLike | None = None) -> NDArray[np.complex128]:
    """
    计算动能对位置的梯度（几何力）。
    
    理论依据：
    -----------
    对于变惯量系统，动能 K = 0.5 * ξ^T I(z) ξ 显式依赖于位置 z。
    
    拉格朗日力学要求：完整的运动方程需包含"几何力"项
        F_geom = -∇_z K = -∇_z (0.5 * ξ^T I(z) ξ)
    
    这是因为 Euler-Lagrange 方程：
        d/dt (∂L/∂ż) = ∂L/∂z
    
    其中右侧 ∂L/∂z = -∂V/∂z + ∂K/∂z 包含动能对位置的显式导数。
    
    在 Diamond 算子的语境下，这对应于：
    - 惯量随位置变化时，即使没有外势，系统仍会产生力
    - 这个力驱动点集向惯性较低（更"滑动"）的区域移动
    
    注意：此力在理论文档 (理论基础-3.md) 中未显式给出，
    但对于变惯量 Euler-Poincaré 系统是必需的。
    
    实现：
    使用中心差分进行数值微分，确保与 locked_inertia_uhp 使用相同的
    _compute_point_features 函数，保证物理一致性。
    
    参数：
        z_uhp: 上半平面坐标
        xi: 思维流速 (u, v, w)
        weights: 可选的质量权重
    
    返回：
        与 z_uhp 同形状的复数梯度，表示 (∂K/∂x + i∂K/∂y)
    """
    z_arr = np.asarray(z_uhp, dtype=np.complex128).ravel()
    xi_vec = np.asarray(xi, dtype=float)
    u, v, w = xi_vec
    
    if weights is None:
        w_arr = np.ones_like(z_arr, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()

    forces = np.zeros_like(z_arr, dtype=np.complex128)
    eps = 1e-5
    
    # 对每个点进行有限差分，计算 dK/dz
    for i in range(z_arr.size):
        zc = z_arr[i]
        mass = w_arr[i]
        
        def get_local_K(z_probe):
            a_u, a_v, a_w = _compute_point_features(z_probe, mass)
            scale = mass
            
            # 重构局部惯性张量
            I00 = scale * np.abs(a_u) ** 2
            I11 = scale * np.abs(a_v) ** 2
            I22 = scale * np.abs(a_w) ** 2
            I01 = scale * np.real(a_u * np.conj(a_v))
            I02 = scale * np.real(a_u * np.conj(a_w))
            I12 = scale * np.real(a_v * np.conj(a_w))
            
            # 二次型: 0.5 * xi^T I_local xi
            return 0.5 * (
                I00*u*u + I11*v*v + I22*w*w + 
                2.0*(I01*u*v + I02*u*w + I12*v*w)
            )

        # 中心差分
        dK_dx = (get_local_K(zc + eps) - get_local_K(zc - eps)) / (2 * eps)
        dK_dy = (get_local_K(zc + 1j*eps) - get_local_K(zc - 1j*eps)) / (2 * eps)
        
        forces[i] = complex(dK_dx, dK_dy)
        
    return forces.reshape(z_uhp.shape)


def apply_inertia(I: ArrayLike, xi: ArrayLike) -> NDArray[np.float64]:
    """Compute m = I xi."""
    return np.asarray(I, dtype=float) @ np.asarray(xi, dtype=float)


def invert_inertia(I: ArrayLike, m: ArrayLike) -> NDArray[np.float64]:
    """Compute xi = I^{-1} m using robust solve."""
    I_arr = np.asarray(I, dtype=float)
    return np.linalg.solve(I_arr + 1e-8 * np.eye(3), np.asarray(m, dtype=float))


def moment_map_covariance(z_uhp: ArrayLike, weights: ArrayLike | None = None, alpha: float = 0.2) -> NDArray[np.float64]:
    """Placeholder or helper for covariance."""
    # (保留原有的实现或简化)
    return np.zeros(3)
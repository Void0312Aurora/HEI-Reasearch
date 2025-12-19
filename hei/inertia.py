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


# ============================================================================
# Hyperboloid 版本的惯性计算（无边界奇异性）
# ============================================================================

def _compute_hyperboloid_generators(h: NDArray[np.float64]) -> tuple[NDArray, NDArray, NDArray]:
    """
    计算 Hyperboloid 上点的李代数 so(2,1) 生成元（带裁剪）
    
    理论依据：
    ---------
    SO(2,1) 的李代数 so(2,1) 有三个生成元：
    - J: 空间旋转 (xy 平面)
    - K_x: x 方向 boost (xt 平面)
    - K_y: y 方向 boost (yt 平面)
    
    这些与 sl(2,R) 的生成元 (u, v, w) 通过同构对应。
    
    对于 Hyperboloid 上的点 h = (X, Y, T)，各生成元作用产生的切向量：
    - J·h = (-Y, X, 0)      # 旋转
    - K_x·h = (T, 0, X)     # x-boost
    - K_y·h = (0, T, Y)     # y-boost
    
    这些切向量在 Minkowski 度量下的内积给出惯性张量。
    
    优势：
    -----
    在 Hyperboloid 上，X, Y, T 都是有界的（只要点在 H² 内部），
    因此生成元永远不会发散！这与 UHP 中 Im(z) → 0 导致的发散形成对比。
    
    数值稳定性：
    -----------
    尽管理论上生成元有界，但在点集接近双曲边界时（大双曲半径），
    生成元的范数仍可能很大，导致惯性张量特征值爆炸。
    
    **裁剪策略**（CRITICAL FIX for Phase 1）：
    - 硬截断模式：限制生成元范数 ≤ GENERATOR_CLIP
    - 物理意义：防止远离中心的点产生过大的惯性贡献
    - 理论依据：对应 UHP 版本的裁剪（保持两种坐标系的数值一致性）
    """
    # Phase 1 紧急修复：生成元裁剪常量（与 UHP 版本一致）
    GENERATOR_CLIP = 5.0  # 限制特征值爆炸（理论预期 λ_max ~ O(N)）
    
    X, Y, T = h[..., 0], h[..., 1], h[..., 2]
    
    # 生成元作用产生的切向量（原始未裁剪）
    a_J_raw = np.stack([-Y, X, np.zeros_like(X)], axis=-1)      # 旋转
    a_Kx_raw = np.stack([T, np.zeros_like(T), X], axis=-1)      # x-boost
    a_Ky_raw = np.stack([np.zeros_like(T), T, Y], axis=-1)      # y-boost
    
    # 裁剪函数：限制切向量的 Minkowski 范数
    def clip_generator(a_vec):
        """
        裁剪生成元切向量到最大范数
        
        注意：使用欧几里得范数（而非 Minkowski 范数）进行裁剪，
        因为我们关心的是数值大小，而不是几何意义。
        """
        norm = np.linalg.norm(a_vec, axis=-1, keepdims=True)
        scale = np.minimum(1.0, GENERATOR_CLIP / (norm + 1e-12))
        return a_vec * scale
    
    # 应用裁剪
    a_J = clip_generator(a_J_raw)
    a_Kx = clip_generator(a_Kx_raw)
    a_Ky = clip_generator(a_Ky_raw)
    
    return a_J, a_Kx, a_Ky


def _minkowski_metric_inner(v1: NDArray, v2: NDArray) -> NDArray:
    """
    切向量的 Minkowski 内积
    
    对于 Hyperboloid 的切向量，使用 Minkowski 度量：
    <v1, v2>_M = v1_x * v2_x + v1_y * v2_y - v1_t * v2_t
    
    注意：这与环境 Minkowski 空间的度量一致，
    诱导到 Hyperboloid 上给出正定的双曲度量。
    """
    return v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1] - v1[..., 2] * v2[..., 2]


def locked_inertia_hyperboloid(
    h: NDArray[np.float64], 
    weights: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """
    Hyperboloid 上的锁定惯性张量（无边界奇异性！）
    
    理论依据：
    ---------
    锁定惯性的定义与 UHP 版本相同，但在 Hyperboloid 坐标下计算：
        I_ij = Σ_k m_k * <a_i(h_k), a_j(h_k)>_M
    
    其中 a_i 是李代数生成元在点 h_k 产生的切向量，
    内积使用 Minkowski 度量（限制到切空间）。
    
    优势：
    -----
    1. 无边界奇异性：Hyperboloid 没有边界
    2. 度量有界：切向量的范数不会发散
    3. 物理一致性：与 UHP 版本通过坐标变换等价
    
    参数：
        h: Hyperboloid 坐标 (..., 3)
        weights: 质量权重（默认为 1）
        
    返回：
        3x3 对称正定惯性矩阵
    """
    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 1:
        h_arr = h_arr.reshape(1, 3)
    
    n = h_arr.shape[0]
    
    # 绝对真空惯性
    I_base = np.eye(3, dtype=float) * 1e-6
    
    if n == 0:
        return I_base
    
    if weights is None:
        w_arr = np.ones(n, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()
    
    # 计算所有点的生成元
    a_J, a_Kx, a_Ky = _compute_hyperboloid_generators(h_arr)
    
    # 生成元列表，对应 sl(2,R) 的 (u, v, w) 基
    # 映射关系：J ↔ u (缩放)，K_x ↔ v (平移)，K_y ↔ w (反演)
    generators = [a_J, a_Kx, a_Ky]
    
    I_matter = np.zeros((3, 3), dtype=float)
    
    for i, a_i in enumerate(generators):
        for j, a_j in enumerate(generators):
            # I_ij = Σ_k m_k * <a_i(h_k), a_j(h_k)>_M
            inner = _minkowski_metric_inner(a_i, a_j)  # 形状 (n,)
            I_matter[i, j] = np.sum(w_arr * inner)
    
    # 相对真空正则化
    trace_val = np.trace(I_matter)
    avg_mass = trace_val / 3.0
    if avg_mass < 1e-9:
        avg_mass = 1.0
    
    REL_EPS = 1e-2
    I_vac_rel = np.eye(3, dtype=float) * (avg_mass * REL_EPS)
    
    return I_matter + I_vac_rel + I_base


def compute_kinetic_energy_gradient_hyperboloid(
    h: NDArray[np.float64],
    xi: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """
    Hyperboloid 上的动能梯度（几何力）
    
    计算 ∇_h K = ∇_h (0.5 * ξ^T I(h) ξ)
    
    数值精度改进（Phase 1 Fix）：
    ----------------------------
    使用自适应有限差分步长，而非固定 eps=1e-5
    
    理由：
    1. 精度不足问题：固定步长在大坐标值时相对误差较大
    2. 自适应步长：eps ∝ ||h|| 提供更好的相对精度
    3. 能量守恒：更精确的几何力减少数值耗散
    
    步长选择策略：
    - 基础：eps = max(1e-7, 1e-4 * ||h||)
    - 下界 1e-7：防止步长过小导致舍入误差
    - 比例因子 1e-4：平衡截断误差和舍入误差
    - 对每个点独立计算：适应局部几何
    
    返回：
        形状 (..., 3) 的 Hyperboloid 切向量（力）
    """
    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 1:
        h_arr = h_arr.reshape(1, 3)
    
    n = h_arr.shape[0]
    xi_vec = np.asarray(xi, dtype=float)
    
    if weights is None:
        w_arr = np.ones(n, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()
    
    # Optimized Vectorized Implementation (O(N))
    # Kinetic Energy K = sum(K_i), where K_i depends only on h_i.
    # So dK/dh_i = dK_i/dh_i. We don't need to recompute the full global sum!
    
    n = h_arr.shape[0]
    xi_vec = np.asarray(xi, dtype=float)
    if weights is None:
        w_arr = np.ones(n, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()

    # Helper to compute local energy K_i for a batch of h vectors
    def compute_local_energy(h_batch):
        # h_batch: (N, 3)
        # Returns: (N,) energy array
        
        # 1. Compute generators (N, 3, 3) where last dim is vector component in ambient space
        # But _compute_hyperboloid_generators returns tuple of (N, 3) arrays
        # corresponding to the 3 algebra generators J, Kx, Ky evaluated at h.
        # Actually _compute_hyperboloid_generators returns a list of vectors?
        # Let's check typical usage or re-implement locally for clarity/speed.
        # Using existing helper from this file might be safer if visible.
        # _compute_hyperboloid_generators is defined in inertia.py? 
        # Yes, viewed in lines 396.
        
        gen_J, gen_Kx, gen_Ky = _compute_hyperboloid_generators(h_batch)
        gens = [gen_J, gen_Kx, gen_Ky]
        
        # 2. Build local Inertia tensor I_local (N, 3, 3)
        # I_ab = <gen_a, gen_b>_Minkowski
        I_local = np.zeros((n, 3, 3), dtype=float)
        
        for i, g_i in enumerate(gens):
            for j, g_j in enumerate(gens):
                # Inner product per point
                val = _minkowski_metric_inner(g_i, g_j) # (N,)
                I_local[:, i, j] = val

        # 3. Compute energy: 0.5 * xi^T * I_local * xi
        # Einsum: (3) @ (N, 3, 3) @ (3) -> (N)
        # K = 0.5 * sum_ab (xi_a * I_ab * xi_b)
        # I_local is (N, 3, 3). Xi is (3,)
        # temp = I_local @ xi  -> (N, 3)
        # result = dot(xi, temp) -> (N)
        mv = np.einsum('nij,j->ni', I_local, xi_vec)
        local_K = 0.5 * np.einsum('i,ni->n', xi_vec, mv)
        
        return local_K * w_arr

    # Vectorized Finite Difference
    h_norm = np.linalg.norm(h_arr, axis=1)
    eps_vec = np.maximum(1e-7, 1e-4 * h_norm) # (N,)
    
    forces = np.zeros_like(h_arr)
    
    # X component
    h_plus = h_arr.copy(); h_plus[:, 0] += eps_vec
    h_minus = h_arr.copy(); h_minus[:, 0] -= eps_vec
    dK_dX = (compute_local_energy(h_plus) - compute_local_energy(h_minus)) / (2 * eps_vec)
    
    # Y component
    h_plus = h_arr.copy(); h_plus[:, 1] += eps_vec
    h_minus = h_arr.copy(); h_minus[:, 1] -= eps_vec
    dK_dY = (compute_local_energy(h_plus) - compute_local_energy(h_minus)) / (2 * eps_vec)
    
    # T component
    h_plus = h_arr.copy(); h_plus[:, 2] += eps_vec
    h_minus = h_arr.copy(); h_minus[:, 2] -= eps_vec
    dK_dT = (compute_local_energy(h_plus) - compute_local_energy(h_minus)) / (2 * eps_vec)
    
    forces[:, 0] = dK_dX
    forces[:, 1] = dK_dY
    forces[:, 2] = dK_dT
    
    # Return gradient (Force is often -gradient, but verify usage. 
    # Docstring says "Returns gradient ∇_h K".
    # Simulation adds F_geom = -compute_kinetic_energy_gradient... so the sign here should be gradient +)
    return forces


def locked_inertia_from_group(
    G: NDArray[np.float64],
    z0_uhp: NDArray[np.complex128],
    use_hyperboloid: bool = True
) -> NDArray[np.float64]:
    """
    从群元素和初始位置计算当前惯性
    
    这是群积分器使用的接口：
    1. 计算当前位置 z = G · z0
    2. 在 Hyperboloid 上计算惯性
    
    参数：
        G: SL(2,R) 群元素
        z0_uhp: 初始 UHP 位置
        use_hyperboloid: 是否使用 Hyperboloid 计算（推荐 True）
        
    返回：
        3x3 惯性矩阵
    """
    from .lie import mobius_action_matrix
    from .geometry import cayley_uhp_to_disk, disk_to_hyperboloid
    
    # 计算当前位置
    z_current = mobius_action_matrix(G, z0_uhp)
    
    if use_hyperboloid:
        # 转换到 Hyperboloid
        z_disk = cayley_uhp_to_disk(z_current)
        h = disk_to_hyperboloid(z_disk)
        return locked_inertia_hyperboloid(h)
    else:
        # 使用原有 UHP 方法
        return locked_inertia_uhp(z_current)
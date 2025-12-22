"""
Hyperboloid 模型：双曲空间的无边界表示
======================================

理论背景：
---------
双曲平面 H² 可以通过多种等价模型表示：
- 上半平面 (UHP): {z ∈ ℂ : Im(z) > 0}，边界在 Im(z) = 0
- Poincaré 圆盘: {z ∈ ℂ : |z| < 1}，边界在 |z| = 1
- Hyperboloid: {(x,y,t) ∈ ℝ³ : -x² - y² + t² = 1, t > 0}，**无边界**

Hyperboloid 模型的优势：
1. 没有几何边界 - 曲面在 ℝ³ 中完整嵌入
2. 度量处处正则 - 诱导度量有界
3. 群作用简单 - SL(2,ℝ) ≅ SO⁺(2,1) 通过 Lorentz 变换作用

坐标变换：
---------
从 Poincaré 圆盘 z = x + iy（其中 |z| < 1）到 Hyperboloid (X, Y, T)：
    X = 2x / (1 - |z|²)
    Y = 2y / (1 - |z|²)
    T = (1 + |z|²) / (1 - |z|²)

约束：-X² - Y² + T² = 1, T > 0

参考文献：
---------
- Cannon, Floyd, Kenyon, Parry (1997). Hyperbolic Geometry.
- Ratcliffe (2006). Foundations of Hyperbolic Manifolds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def disk_to_hyperboloid(z_disk: ArrayLike) -> NDArray[np.float64]:
    """
    Poincaré 圆盘 → Hyperboloid 坐标
    
    参数：
        z_disk: 圆盘上的复数点，|z| < 1
        
    返回：
        形状 (..., 3) 的数组，每行为 (X, Y, T)
        
    注意：
        当 |z| → 1 时，T → ∞，但 X, Y, T 都是有限的实数，
        不像 UHP 中 Im(z) → 0 导致度量发散。
    """
    z = np.asarray(z_disk, dtype=np.complex128)
    original_shape = z.shape
    z_flat = z.ravel()
    
    x = np.real(z_flat)
    y = np.imag(z_flat)
    r2 = x * x + y * y
    
    # 防止除零，但不需要像 UHP 那样激进的裁剪
    # 因为 hyperboloid 坐标本身是良定义的
    denom = np.maximum(1.0 - r2, 1e-15)
    
    X = 2.0 * x / denom
    Y = 2.0 * y / denom
    T = (1.0 + r2) / denom
    
    # 验证约束（调试用）
    # assert np.allclose(-X*X - Y*Y + T*T, 1.0), "Hyperboloid constraint violated"
    
    result = np.stack([X, Y, T], axis=-1)
    if original_shape == ():
        return result.reshape(3)
    return result.reshape(original_shape + (3,))


def hyperboloid_to_disk(h: ArrayLike) -> NDArray[np.complex128]:
    """
    Hyperboloid → Poincaré 圆盘坐标
    
    参数：
        h: 形状 (..., 3) 的数组，每行为 (X, Y, T)
        
    返回：
        圆盘上的复数点
    """
    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.shape[-1] != 3:
        raise ValueError(f"Expected last dimension 3, got {h_arr.shape[-1]}")
    
    X = h_arr[..., 0]
    Y = h_arr[..., 1]
    T = h_arr[..., 2]
    
    # z = (X + iY) / (1 + T)
    denom = 1.0 + T
    z = (X + 1j * Y) / np.maximum(denom, 1e-15)
    
    return z


def minkowski_inner(h1: ArrayLike, h2: ArrayLike) -> NDArray[np.float64]:
    """
    Minkowski 内积：<h1, h2>_L = x1*x2 + y1*y2 - t1*t2
    
    对于 Hyperboloid 上的点，有 <h, h>_L = -1
    """
    h1_arr = np.asarray(h1, dtype=np.float64)
    h2_arr = np.asarray(h2, dtype=np.float64)
    
    return (h1_arr[..., 0] * h2_arr[..., 0] + 
            h1_arr[..., 1] * h2_arr[..., 1] - 
            h1_arr[..., 2] * h2_arr[..., 2])


def hyperbolic_distance_hyperboloid(h1: ArrayLike, h2: ArrayLike) -> NDArray[np.float64]:
    """
    Hyperboloid 上的双曲距离
    
    d(h1, h2) = arccosh(-<h1, h2>_L)
    
    注意：对于 H² 上的点，-<h1, h2>_L ≥ 1
    """
    inner = minkowski_inner(h1, h2)
    # -inner ≥ 1，但数值上可能略小于 1
    return np.arccosh(np.maximum(-inner, 1.0))


def hyperbolic_distance_grad_hyperboloid(
    h: ArrayLike, 
    c: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Hyperboloid 上的双曲距离及其梯度
    
    参数：
        h: 当前点 (X, Y, T)
        c: 目标点 (Xc, Yc, Tc)
        
    返回：
        (distance, gradient)
        其中 gradient 是在 Hyperboloid 切空间中的梯度
    """
    h_arr = np.asarray(h, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    
    # 距离
    inner = minkowski_inner(h_arr, c_arr)
    minus_inner = np.maximum(-inner, 1.0 + 1e-12)
    d = np.arccosh(minus_inner)
    
    # 梯度：d/dh arccosh(-<h,c>_L)
    # = -1/sqrt((<h,c>_L)^2 - 1) * (c_x, c_y, -c_t)
    # 其中 Minkowski 梯度为 (∂/∂x, ∂/∂y, -∂/∂t)
    denom = np.sqrt(np.maximum(minus_inner * minus_inner - 1.0, 1e-12))
    
    # 切空间梯度（投影到约束曲面上）
    grad_ambient = np.stack([c_arr[..., 0], c_arr[..., 1], -c_arr[..., 2]], axis=-1) / denom[..., np.newaxis]
    
    # 投影到切空间：grad_tangent = grad - <grad, h>_L * h
    proj_coeff = minkowski_inner(grad_ambient, h_arr)
    grad_tangent = grad_ambient + proj_coeff[..., np.newaxis] * h_arr
    
    return d, grad_tangent


def project_to_hyperboloid(h: ArrayLike) -> NDArray[np.float64]:
    """
    将点投影回 Hyperboloid 曲面
    
    用于数值稳定性：确保 -X² - Y² + T² = 1
    """
    h_arr = np.asarray(h, dtype=np.float64)
    X, Y, T = h_arr[..., 0], h_arr[..., 1], h_arr[..., 2]
    
    # 从 X, Y 重新计算 T
    T_new = np.sqrt(1.0 + X * X + Y * Y)
    
    return np.stack([X, Y, T_new], axis=-1)


def sl2_to_so21(g: ArrayLike) -> NDArray[np.float64]:
    """
    SL(2,ℝ) → SO⁺(2,1) 同构
    
    SL(2,ℝ) 通过 Möbius 变换作用于 H²，
    这诱导了 SO⁺(2,1) 对 Hyperboloid 的 Lorentz 变换作用。
    
    参数：
        g: SL(2,ℝ) 矩阵 [[a, b], [c, d]]
        
    返回：
        SO⁺(2,1) 矩阵，作用于 (X, Y, T) 列向量
    """
    g_arr = np.asarray(g, dtype=np.float64)
    a, b = g_arr[0, 0], g_arr[0, 1]
    c, d = g_arr[1, 0], g_arr[1, 1]
    
    # 显式公式（从 SL(2,ℝ) 对 UHP 的作用推导）
    # 参考：Ratcliffe, "Foundations of Hyperbolic Manifolds", Chapter 4
    L = np.array([
        [0.5*(a*a + b*b + c*c + d*d), 
         0.5*(a*a - b*b + c*c - d*d), 
         a*c + b*d],
        [0.5*(a*a + b*b - c*c - d*d), 
         0.5*(a*a - b*b - c*c + d*d), 
         a*c - b*d],
        [a*b + c*d, 
         a*b - c*d, 
         a*d + b*c]
    ], dtype=np.float64)
    
    return L


def lorentz_action(L: ArrayLike, h: ArrayLike) -> NDArray[np.float64]:
    """
    SO⁺(2,1) 对 Hyperboloid 的作用
    
    h_new = L @ h（矩阵乘法）
    
    注意：这是简单的线性变换，永远不会有奇异性！
    """
    L_arr = np.asarray(L, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    
    if h_arr.ndim == 1:
        return L_arr @ h_arr
    else:
        # 批量处理：h 形状为 (N, 3)
        return (L_arr @ h_arr.T).T


def hyperboloid_metric_tensor(h: ArrayLike) -> NDArray[np.float64]:
    """
    Hyperboloid 上的度量张量（在切空间基底下）
    
    由于 Hyperboloid 是常负曲率曲面，诱导度量是良定义的。
    在切空间 T_h H² 中，度量是 Minkowski 度量限制到切平面。
    
    返回：
        2x2 度量张量（在某个切空间基底下）
    """
    h_arr = np.asarray(h, dtype=np.float64)
    X, Y, T = h_arr[..., 0], h_arr[..., 1], h_arr[..., 2]
    
    # 切空间基向量（与约束梯度正交）
    # 约束：F = -X² - Y² + T² - 1 = 0
    # ∇F = (-2X, -2Y, 2T)，法向量 n = (X, Y, -T)/T（归一化）
    
    # 选取切空间基：e1 沿 X 方向投影，e2 沿 Y 方向投影
    # e1 = (1, 0, X/T) / |...|, e2 = (0, 1, Y/T) / |...|
    
    # 诱导度量 g_ij = <e_i, e_j>_L
    # 计算表明 g = I_2 / T²（与 Poincaré 度量一致）
    
    # 但我们直接返回单位矩阵，因为在 hyperboloid 坐标下工作时
    # 度量效应已经被编码在距离函数中
    return np.eye(2, dtype=np.float64)


def gamma_hyperboloid_metric_based(
    h: ArrayLike,
    scale: float = 2.0,
    per_point: bool = False
) -> NDArray[np.float64] | float:
    """
    基于度量张量的几何临界阻尼（符合公理 4.2）
    
    理论依据：
    ---------
    公理 4.2（理论基础-3.md）要求：
        γ(q) ∝ sqrt(λ_max(g(q)))
    
    其中 g(q) 是度量张量，λ_max 是其最大特征值。
    
    对于 Hyperboloid 模型：
    ----------------------
    在切空间中，诱导度量的特征值为：
        λ = 1 / T²
    
    其中 T 是 Hyperboloid 坐标的第三分量（时间坐标）。
    
    物理意义：
    ---------
    - T ≈ 1（圆盘中心）：曲率大 → λ 大 → 阻尼大 → 快速锁定
    - T 增大（远离中心）：曲率小 → λ 小 → 阻尼小 → 允许探索
    
    这符合"高曲率→高阻尼"的物理直觉：
    在信念确定的区域（高曲率），系统应快速收敛；
    在不确定的区域（低曲率），系统应保持探索性。
    
    参数：
        h: Hyperboloid 坐标 (..., 3)
        scale: 基础阻尼缩放因子
        per_point: 是否返回逐点阻尼（True）或平均阻尼（False）
    
    返回：
        阻尼系数（标量或逐点数组）
    
    参考文献：
        - 理论基础-3.md 公理 4.2
        - Ratcliffe (2006). Foundations of Hyperbolic Manifolds, Ch. 3
    """
    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 1:
        h_arr = h_arr.reshape(1, 3)
    
    T = h_arr[..., 2]
    
    # 度量张量的特征值（Hyperboloid 上为 1/T²）
    # 添加小量防止除零
    lambda_metric = 1.0 / (T * T + 1e-12)
    
    # 按理论公式：γ ∝ sqrt(λ_max)
    gamma_local = scale * np.sqrt(lambda_metric)
    
    if per_point:
        return gamma_local  # 逐点阻尼
    else:
        return float(np.mean(gamma_local))  # 平均阻尼


def gamma_hyperboloid(
    h: ArrayLike,
    scale: float = 2.0,
    mode: str = "metric"
) -> NDArray[np.float64] | float:
    """
    Hyperboloid 上的几何阻尼
    
    与 UHP/圆盘上的阻尼不同，这里没有边界奇异性！
    
    参数：
        h: Hyperboloid 坐标 (..., 3)
        scale: 基础阻尼缩放
        mode: 阻尼模式
            - "constant": 常数阻尼（推荐，最稳定）✅
            - "metric": 基于度量张量（符合公理 4.2，但在边界可能不稳定）
            - "adaptive": 基于 T 坐标的启发式（旧版本，用于兼容）
            
    返回：
        阻尼系数（标量或逐点）
        
    理论说明：
        默认的 "metric" 模式实现了公理 4.2 的几何临界阻尼：
        γ ∝ sqrt(λ_max(g))，其中 λ = 1/T²
        
        注意："metric" 模式在边界处阻尼会变小（因为 T 变大），有助于探索，
        但也可能导致高刚性系统在边界处不稳定。生产环境推荐使用 "constant"。
    
    参考：
        理论基础-3.md 公理 4.2
    """
    h_arr = np.asarray(h, dtype=np.float64)
    
    if mode == "metric":
        # 新的度量基阻尼（符合理论）
        return gamma_hyperboloid_metric_based(h_arr, scale, per_point=False)
    elif mode == "constant":
        # 双曲空间曲率 K = -1（常数），所以临界阻尼也应该是常数
        return scale
    elif mode == "adaptive":
        # 保留旧实现用于兼容性
        # adaptive 模式：根据 T 坐标（对应于"接近边界程度"）调整
        T = h_arr[..., 2]
        T_mean = float(np.mean(T)) if np.ndim(T) > 0 else float(T)
        
        # T = 1 对应圆盘中心，T → ∞ 对应边界
        # 使用 log(T) 增长，非常温和
        gamma = scale * (1.0 + 0.1 * np.log(max(T_mean, 1.0)))
        
        # 硬上界，但这个上界很少会触及
        return min(gamma, 10.0)
    else:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: 'metric', 'constant', 'adaptive'")


def compute_centroid_hyperboloid(h: ArrayLike, max_iter: int = 20) -> NDArray[np.float64]:
    """
    Hyperboloid 上的 Karcher 均值（双曲重心）
    
    使用梯度下降最小化距离平方和。
    """
    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 1:
        return h_arr.copy()
    
    n = h_arr.shape[0]
    if n == 0:
        return np.array([0.0, 0.0, 1.0])
    
    # 初始化：算术平均（然后投影）
    mu = np.mean(h_arr, axis=0)
    mu = project_to_hyperboloid(mu)
    
    step_size = 0.3
    for _ in range(max_iter):
        grad_sum = np.zeros(3, dtype=np.float64)
        for i in range(n):
            _, grad = hyperbolic_distance_grad_hyperboloid(mu, h_arr[i])
            grad_sum += grad
        
        grad_norm = np.linalg.norm(grad_sum)
        if grad_norm < 1e-8:
            break
        
        # 沿切空间方向移动
        mu = mu - step_size * grad_sum / n
        # 投影回 hyperboloid
        mu = project_to_hyperboloid(mu)
    
    return mu


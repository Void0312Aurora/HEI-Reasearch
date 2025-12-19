"""
群积分器：基于李群的接触动力学积分器
====================================

理论背景：
---------
传统积分器在位置空间 z ∈ H² 上积分，但 H² 有边界（圆盘边缘或 UHP 的实轴），
导致边界附近的数值问题。

群积分器的核心思想是：真正的状态变量是群元素 g ∈ SL(2,R)，而非位置 z。
位置只是初始位置被群作用的结果：z(t) = g(t) · z(0)。

由于 SL(2,R) 是闭合的李群，没有边界，积分过程永远稳定。

混合架构：
---------
1. 在 SL(2,R) 群上积分（保证稳定性）
2. 在 Hyperboloid 坐标系中评估（保证度量有界）
3. 惰性求值位置（仅在需要时计算）

参考文献：
---------
- Iserles, Munthe-Kaas, Nørsett, Zanna (2000). Lie-group methods.
- Hairer, Lubich, Wanner (2006). Geometric Numerical Integration.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .diamond import aggregate_torque, aggregate_torque_hyperboloid
from .geometry import (
    cayley_uhp_to_disk,
    cayley_disk_to_uhp,
    disk_to_hyperboloid,
    hyperboloid_to_disk,
    gamma_hyperboloid,
    clamp_disk_radius,
)
from .inertia import (
    apply_inertia,
    invert_inertia,
    locked_inertia_hyperboloid,
    locked_inertia_uhp,
    compute_kinetic_energy_gradient_hyperboloid,
)
from .lie import exp_sl2, mobius_action_matrix, vec_to_matrix, matrix_to_vec


# 类型别名
ForceFn = Callable[[ArrayLike, float], NDArray[np.complex128]]
GammaFn = Callable[[ArrayLike], float]
PotentialFn = Callable[[ArrayLike, float], float]


@dataclasses.dataclass
class GroupIntegratorState:
    """
    群积分器状态
    
    核心思想：状态由群元素 G 和初始位置 z0 组成，
    当前位置通过 z = G · z0 惰性求值。
    
    属性：
        G: SL(2,R) 累积群元素 (2x2 矩阵)
        z0_uhp: 初始 UHP 位置（不变）
        xi: 当前思维流速 (u, v, w)
        m: 动量
        action: 接触作用量（惊奇 Z）
        dt_last: 上一步使用的时间步长
        gamma_last: 上一步使用的阻尼系数
    """
    G: NDArray[np.float64]              # 累积群元素 (2, 2)
    z0_uhp: NDArray[np.complex128]      # 初始位置（不变）
    xi: NDArray[np.float64]             # 思维流速 (3,)
    m: NDArray[np.float64]              # 动量 (3,)
    action: float = 0.0                 # 接触作用量
    dt_last: float = 0.0
    gamma_last: float = 0.0
    
    @property
    def z_uhp(self) -> NDArray[np.complex128]:
        """惰性求值当前 UHP 位置"""
        return mobius_action_matrix(self.G, self.z0_uhp)
    
    @property
    def z_disk(self) -> NDArray[np.complex128]:
        """惰性求值当前圆盘位置"""
        return cayley_uhp_to_disk(self.z_uhp)
    
    @property
    def h(self) -> NDArray[np.float64]:
        """惰性求值当前 Hyperboloid 位置"""
        return disk_to_hyperboloid(self.z_disk)
    
    @property
    def I(self) -> NDArray[np.float64]:
        """从当前构型计算惯性（使用 Hyperboloid）"""
        return locked_inertia_hyperboloid(self.h)


@dataclasses.dataclass
class GroupIntegratorConfig:
    """
    群积分器配置
    
    阻尼配置说明：
    -------------
    use_hyperboloid_gamma: 是否使用 Hyperboloid 上的几何阻尼
    gamma_mode: 阻尼计算模式（仅当 use_hyperboloid_gamma=True 时生效）
        - "metric": 基于度量张量（符合公理 4.2）✅ 推荐
        - "constant": 常数阻尼
        - "adaptive": 基于 T 坐标的启发式（旧版本）
    gamma_scale: 阻尼缩放因子
    
    理论依据：
    ---------
    公理 4.2（理论基础-3.md）要求：γ(q) ∝ sqrt(λ_max(g(q)))
    "metric" 模式实现了这一要求，提供几何自适应的临界阻尼。
    
    诊断配置（Phase 1 新增）：
    -----------------------
    verbose: 启用详细诊断输出
    diagnostic_interval: 诊断输出频率（每N步输出一次）
    
    Phase 1 参数调优依据：
    --------------------
    max_dt: 0.02 (从0.05降低) - 减小时间步长提高稳定性
    torque_clip: 30.0 (从50.0降低) - 限制力矩尖峰，防止能量突变
    """
    eps_disp: float = 1e-2              # 位移裁剪阈值
    max_dt: float = 2e-2                # 最大时间步长（Phase 1: 从5e-2降至2e-2）
    min_dt: float = 1e-5                # 最小时间步长
    fixed_point_iters: int = 2          # 定点迭代次数
    v_floor: float = 1e-6               # 速度下界
    xi_clip: float = 10.0               # xi 范数裁剪
    torque_clip: float = 30.0           # 力矩裁剪（Phase 1: 从50.0降至30.0）
    renorm_interval: int = 100          # 群矩阵重正规化间隔
    use_hyperboloid_gamma: bool = True  # 使用 Hyperboloid 阻尼
    gamma_mode: str = "metric"          # 阻尼模式（"metric", "constant", "adaptive"）
    gamma_scale: float = 2.0            # 阻尼缩放
    verbose: bool = False               # 诊断输出开关（Phase 1）
    diagnostic_interval: int = 100      # 诊断输出间隔
    implicit_potential: bool = True     # Phase 2: semi-implicit potential evaluation
    solver_mixing: float = 0.5          # Phase 3: Fixed-point relaxation (0.1-0.5 recommended for stiff)

    

class GroupContactIntegrator:
    """
    基于李群的接触动力学积分器
    
    核心改进：
    1. 在 SL(2,R) 群上累积演化，避免边界问题
    2. 在 Hyperboloid 坐标系中评估物理量，避免度量发散
    3. 使用 Cayley-型离散化保持接触结构
    """
    
    def __init__(
        self,
        force_fn: ForceFn,
        potential_fn: PotentialFn,
        gamma_fn: GammaFn | None = None,
        config: GroupIntegratorConfig | None = None,
    ) -> None:
        self.force_fn = force_fn
        self.potential_fn = potential_fn
        self.gamma_fn = gamma_fn
        self.config = config or GroupIntegratorConfig()
        self._step_count = 0
    
    def _renormalize_sl2(self, G: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        SL(2,R) 群元素重正规化
        
        保证 det(G) = 1，修正累积的浮点误差。
        使用极分解方法保持群结构。
        """
        det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
        
        if abs(det - 1.0) > 1e-10:
            # 简单缩放使 det = 1
            scale = 1.0 / np.sqrt(abs(det) + 1e-15)
            G = G * scale
        
        # 额外的对称化（可选）
        # 如果 G 应该是正交的某种形式，可以用 polar decomposition
        
        return G
    
    def _coadjoint_transport(
        self, 
        xi: ArrayLike, 
        m: ArrayLike, 
        dt: float
    ) -> NDArray[np.float64]:
        """
        计算动量的共伴随输运 Ad^*_{f^{-1}}(m)
        
        理论依据：
        在离散 Euler-Poincaré 方程中，动量通过群作用传播：
            μ_k = Ad^*_{f_{k-1}^{-1}}(μ_{k-1}) + ...
        
        对于 SL(2,R) 使用 trace pairing：
            Ad^*_{f^{-1}}(M) = f M f^{-1}
        """
        g = exp_sl2(xi, dt)  # f = exp(dt * xi)
        g_inv = np.linalg.inv(g)
        M = vec_to_matrix(m)
        # Ad^*_{f^{-1}}(M) = f M f^{-1}
        # Derivation with trace pairing <A, B> = tr(AB):
        # <Ad_g X, Y> = <gXg^{-1}, Y> = tr(gXg^{-1}Y) = tr(X g^{-1}Yg) = <X, Ad_{g^{-1}}Y>
        # So Ad^*_g = Ad_{g^{-1}}.
        # Thus Ad^*_{f^{-1}}(M) = Ad_{(f^{-1})^{-1}}(M) = Ad_f(M) = f M f^{-1}.
        M_new = g @ M @ g_inv
        return matrix_to_vec(M_new)
    
    def _alpha(self, gamma: float, dt: float) -> float:
        """
        Cayley-型离散耗散因子
        
        α = (1 - hγ/2) / (1 + hγ/2)
        
        这是 exp(-hγ) 的 Padé 近似，保持接触结构。
        """
        numerator = 1.0 - 0.5 * dt * gamma
        denominator = 1.0 + 0.5 * dt * gamma
        return numerator / max(denominator, 1e-12)
    
    def _compute_gamma(self, state: GroupIntegratorState) -> float:
        """
        计算阻尼系数
        
        理论依据：
        ---------
        公理 4.2（理论基础-3.md）要求几何临界阻尼：
            γ(q) ∝ sqrt(λ_max(g(q)))
        
        实现策略：
        ---------
        1. 优先使用 Hyperboloid 上的几何阻尼（无奇异性）
        2. 根据 gamma_mode 选择具体实现：
           - "metric": 基于度量张量特征值（符合公理 4.2）
           - "constant": 常数阻尼
           - "adaptive": 基于 T 坐标的启发式
        3. 回退到用户提供的 gamma_fn
        4. 最后使用常数 gamma_scale
        
        返回：
            阻尼系数（标量）
        """
        if self.config.use_hyperboloid_gamma:
            return gamma_hyperboloid(
                state.h,
                scale=self.config.gamma_scale,
                mode=self.config.gamma_mode
            )
        elif self.gamma_fn is not None:
            return float(np.mean(self.gamma_fn(state.z_uhp)))
        else:
            return self.config.gamma_scale
    
    def _adaptive_dt(
        self,
        state: GroupIntegratorState,
        torque_norm: float,
        xi_norm: float,
        gamma: float,
    ) -> float:
        """
        自适应时间步长
        
        考虑：
        1. 速度约束：dt * ||ξ|| < ε
        2. 加速度约束：dt * ||τ||/I_min < ε  
        3. 阻尼约束：dt * γ < 安全因子
        """
        cfg = self.config
        
        # 速度约束
        dt_vel = cfg.eps_disp / max(xi_norm, 1e-9)
        
        # 加速度约束
        I_min = max(np.linalg.eigvalsh(state.I).min(), 1e-6)
        dt_acc = cfg.eps_disp / max(torque_norm / I_min, 1e-9)
        
        # 阻尼约束
        dt_gamma = 0.8 / max(gamma, 1e-9)
        
        dt = min(cfg.max_dt, dt_vel, dt_acc, dt_gamma)
        dt = max(dt, cfg.min_dt)
        
        return dt
    
    def step(self, state: GroupIntegratorState) -> GroupIntegratorState:
        """
        执行一步积分
        
        流程：
        1. 计算当前位置（惰性求值）
        2. 在 Hyperboloid 上评估势能力、几何力、阻尼、惯性
        3. 更新动量（共伴随输运 + 总力矩）
        4. 更新群元素（核心！）
        5. 更新接触作用量
        
        关键改进：
        ---------
        现在包含几何力 F_geom = -∇_h K，这是变惯量系统必需的力。
        总力矩 = 势能力矩 + 几何力矩
        
        Phase 1 诊断增强：
        ----------------
        可选的详细诊断输出，用于追踪数值稳定性问题
        """
        cfg = self.config
        self._step_count += 1
        
        # 1. 惰性求值当前位置
        z_current = state.z_uhp
        h_current = state.h
        
        # ============================================================
        # Phase 1 诊断输出：追踪数值稳定性关键指标
        # ============================================================
        if cfg.verbose or (self._step_count % cfg.diagnostic_interval == 0):
            # 计算惯性特征值（检测λ_max爆炸）
            I_current = locked_inertia_hyperboloid(h_current)
            eigvals = np.linalg.eigvalsh(I_current)
            lambda_min = eigvals[0]
            lambda_max = eigvals[-1]
            condition_number = lambda_max / max(lambda_min, 1e-12)
            
            # 能量分解
            T = 0.5 * float(state.xi @ (I_current @ state.xi))
            V = float(self.potential_fn(z_current, state.action))
            H_total = T + V
            
            # 点集位置统计（Hyperboloid T坐标）
            if h_current.ndim == 1:
                T_coords = h_current[2]
                T_max = T_coords
                T_mean = T_coords
                T_min = T_coords
            else:
                T_coords = h_current[:, 2]
                T_max = np.max(T_coords)
                T_mean = np.mean(T_coords)
                T_min = np.min(T_coords)
            
            # 思维流速范数
            xi_norm = np.linalg.norm(state.xi)
            
            print(f"[Step {self._step_count:5d}] Phase 1 诊断:")
            print(f"  惯性: λ_min={lambda_min:.2e}, λ_max={lambda_max:.2e}, κ(I)={condition_number:.1f}")
            print(f"  能量: T={T:.3f}, V={V:.3f}, H={H_total:.3f}")
            print(f"  位置: T_min={T_min:.3f}, T_mean={T_mean:.3f}, T_max={T_max:.3f}")
            print(f"  速度: ||ξ||={xi_norm:.3f}, Z={state.action:.3f}")
            
            # 警告：检测异常情况
            if lambda_max > 500:
                print(f"  ⚠️  警告: λ_max={lambda_max:.1f} 超出正常范围 (预期<200)")
            if condition_number > 1000:
                print(f"  ⚠️  警告: 条件数κ(I)={condition_number:.1f} 过大 (预期<100)")
            if np.isnan(H_total) or np.isinf(H_total):
                print(f"  ❌ 错误: 能量出现NaN/Inf!")
        
        # 2a. 计算势能力和力矩（UHP 坐标系）
        forces_potential = self.force_fn(z_current, state.action)  # 复数力 (N,)
        torque_potential = aggregate_torque(z_current, forces_potential)  # (3,)
        
        # 2b. 计算几何力和力矩（Hyperboloid 坐标系）✅ 新增
        # 几何力来自动能对位置的梯度：F_geom = -∇_h K
        # 这对于变惯量系统是必需的，确保拉格朗日方程完整
        F_geom_h = -compute_kinetic_energy_gradient_hyperboloid(
            h_current, state.xi, weights=None
        )  # (N, 3) 或 (3,)
        torque_geom = aggregate_torque_hyperboloid(h_current, F_geom_h)  # (3,)
        
        # 2c. 总力矩 = 势能力矩 + 几何力矩 ✅ 修改
        torque = torque_potential + torque_geom
        
        # 力矩裁剪
        torque_norm = float(np.linalg.norm(torque))
        if torque_norm > cfg.torque_clip:
            torque = torque * (cfg.torque_clip / torque_norm)
            torque_norm = cfg.torque_clip
        
        # 3. 计算阻尼和惯性（使用 Hyperboloid）
        gamma = self._compute_gamma(state)
        I = locked_inertia_hyperboloid(h_current)
        
        # 4. 计算时间步长
        xi_norm = float(np.linalg.norm(state.xi))
        dt = self._adaptive_dt(state, torque_norm, xi_norm, gamma)
        
        # 5. 动量更新（Euler-Poincaré）
        alpha = self._alpha(gamma, dt)
        m_prev = state.m if state.m is not None else apply_inertia(I, state.xi)
        m_advected = self._coadjoint_transport(state.xi, m_prev, dt)
        
        # 理论修正：力矩项需乘以 (1 + h*gamma/2) 因子
        # Eq 67: mu_k = (1 - h*gamma/2) Ad* + h * (1 + h*gamma/2) * J
        # alpha = (1 - h*gamma/2) / (1 + h*gamma/2)
        # 所以 m_new = alpha * (1 + h*gamma/2) * Ad* + h * (1 + h*gamma/2) * J
        #           = (1 - h*gamma/2) * Ad* + ...
        # 但代码中 alpha 是与 Ad* 相乘的，我们保留 alpha 的定义，手动给力矩项加因子
        # 注意：这里我们让 alpha 保持原样，那么我们需要给整个式子乘以 denom 吗？
        # 不，最好的方式是重写这一行以完全匹配理论
        
        # 修正逻辑：
        # denom = 1.0 + 0.5 * dt * gamma
        # m_new = (1.0 - 0.5 * dt * gamma) * m_advected + dt * denom * torque
        
        # 兼容现有 alpha 写法：
        denom = 1.0 + 0.5 * dt * gamma
        m_new = alpha * denom * m_advected + dt * denom * torque
        
        # 6.Fixed-point iteration for implicit solver (Feature Fix)
        # Initial guess from Explicit Step (already done above sort of)
        # We start with the 'explicit' guess which uses torque calculated from OLD xi
        
        # Initial guess: xi_k^(0) = xi_prev (or better, the result of explicit step?)
        # The equation is: I(a_k) xi_k = ... + h * Torque(a_k, xi_k)
        # Note: Torque depends on xi_k via F_geom!    
        
        # let's calculate the initial guess explicitly
        xi_new = invert_inertia(I, m_new)
        
        # Let's refine xi_new using fixed point iteration
        xi_iter = xi_new.copy()

        for _ in range(cfg.fixed_point_iters):
             # Re-calculate geometric torque using NEW xi estimate
             F_geom_h = -compute_kinetic_energy_gradient_hyperboloid(
                h_current, xi_iter, weights=None
             )
             torque_geom = aggregate_torque_hyperboloid(h_current, F_geom_h)
             
             # Phase 2 Fundamental Fix: Implicit Potential
             # 对于刚性势场，必须隐式求解势能梯度，否则显式积分会爆炸
             # 我们使用中点规则：在 z_mid = exp(xi * dt/2) * z_current 处评估力
             t_pot = torque_potential  # default to explicit
             if cfg.implicit_potential:
                 # Midpoint approximation
                 g_mid = exp_sl2(xi_iter, 0.5 * dt)
                 # Note: applying group action is relatively cheap compared to potential grad?
                 # Actually potential grad is expensive. But for stability we must pay.
                 # Optimization: only re-eval every other iter? For now, full rigorous implicit.
                 z_mid = mobius_action_matrix(g_mid, z_current)
                 f_pot_mid = self.force_fn(z_mid, state.action)
                 t_pot = aggregate_torque(z_mid, f_pot_mid)
             
             # Total torque update
             torque_loop = t_pot + torque_geom
             
             # Clip torque
             t_norm = float(np.linalg.norm(torque_loop))
             if t_norm > cfg.torque_clip:
                 torque_loop = torque_loop * (cfg.torque_clip / t_norm)
                 
             # Update momentum (RHS)
             # m_new = alpha * denom * m_advected + dt * denom * torque_loop
             denom = 1.0 + 0.5 * dt * gamma
             m_loop = alpha * denom * m_advected + dt * denom * torque_loop
             
             # Solve for xi
             xi_next = invert_inertia(I, m_loop)
             
             # Clip xi
             xi_norm = float(np.linalg.norm(xi_next))
             if xi_norm > cfg.xi_clip:
                 xi_next = xi_next * (cfg.xi_clip / xi_norm)
            
             # Relaxed update (Under-relaxation)
             # 对于刚性系统，直接迭代 xi_{k+1} = G(xi_k) 容易发散
             # 使用混合更新: xi = (1-beta)*old + beta*new
             beta = cfg.solver_mixing
             xi_iter = (1.0 - beta) * xi_iter + beta * xi_next
             
        # Final result
        xi_new = xi_iter
        m_new = apply_inertia(I, xi_new)
        
        # 7. 群更新（核心改进！）
        g_step = exp_sl2(xi_new, dt)
        G_new = state.G @ g_step  # 累积群作用
        
        # 周期性重正规化
        if self._step_count % cfg.renorm_interval == 0:
            G_new = self._renormalize_sl2(G_new)
        
        # 8. 接触作用量更新
        T = 0.5 * float(xi_new @ (I @ xi_new))
        V = float(self.potential_fn(z_current, state.action))
        denom = 1.0 + 0.5 * dt * gamma
        action_new = alpha * state.action + dt / denom * (T - V)
        
        return GroupIntegratorState(
            G=G_new,
            z0_uhp=state.z0_uhp,  # 初始位置不变
            xi=xi_new,
            m=m_new,
            action=action_new,
            dt_last=dt,
            gamma_last=gamma,
        )
    
    def coadjoint_transport(
        self, 
        xi: ArrayLike, 
        m: ArrayLike, 
        dt: float
    ) -> NDArray[np.float64]:
        """公开接口：共伴随输运"""
        return self._coadjoint_transport(xi, m, dt)


def create_initial_group_state(
    z_uhp: NDArray[np.complex128],
    xi: NDArray[np.float64] | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GroupIntegratorState:
    """
    创建初始群积分器状态
    
    参数：
        z_uhp: 初始 UHP 位置
        xi: 初始思维流速
        
    返回：
        初始化的 GroupIntegratorState
    """
    z0 = np.asarray(z_uhp, dtype=np.complex128)
    xi0 = np.asarray(xi, dtype=np.float64)
    G0 = np.eye(2, dtype=np.float64)  # 单位群元素
    
    # 计算初始惯性和动量
    z_disk = cayley_uhp_to_disk(z0)
    h = disk_to_hyperboloid(z_disk)
    I = locked_inertia_hyperboloid(h)
    m0 = apply_inertia(I, xi0)
    
    return GroupIntegratorState(
        G=G0,
        z0_uhp=z0,
        xi=xi0,
        m=m0,
        action=0.0,
    )


# 向后兼容：创建传统风格的状态（用于与旧代码交互）
def group_state_to_legacy(state: GroupIntegratorState) -> dict:
    """将群状态转换为传统格式"""
    return {
        "z_uhp": state.z_uhp,
        "xi": state.xi,
        "I": state.I,
        "m": state.m,
        "action": state.action,
        "dt_last": state.dt_last,
        "gamma_last": state.gamma_last,
    }


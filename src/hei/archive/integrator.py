"""Legacy contact splitting integrator for CCD on the upper half-plane."""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..diamond import aggregate_torque
from ..inertia import apply_inertia, invert_inertia, locked_inertia_uhp, compute_kinetic_energy_gradient
from ..lie import exp_sl2, mobius_action_matrix, matrix_to_vec, vec_to_matrix
from ..geometry import cayley_uhp_to_disk, cayley_disk_to_uhp, clamp_disk_radius, clamp_uhp
from ..metrics import poincare_distance_disk


ForceFn = Callable[[ArrayLike, float], NDArray[np.complex128]]
GammaFn = Callable[[ArrayLike], float]
PotentialFn = Callable[[ArrayLike, float], float]


@dataclasses.dataclass
class IntegratorState:
    z_uhp: NDArray[np.complex128]
    xi: NDArray[np.float64]
    I: NDArray[np.float64]
    m: NDArray[np.float64] | None = None
    action: float = 0.0
    dt_last: float = 0.0
    gamma_last: float = 0.0


@dataclasses.dataclass
class IntegratorConfig:
    eps_disp: float = 1e-2
    max_dt: float = 1e-3
    min_dt: float = 1e-6
    fixed_point_iters: int = 2
    v_floor: float = 1e-6
    torque_dt_scale: float = 1e-4
    xi_clip: float = 10.0  # 降低默认值以防止爆炸
    step_clip: float = 2.0  # 降低默认值
    relax_eta: float = 0.0
    dt_damping_safety: float = 1.5
    # 力矩裁剪：限制力矩范数以防止加速度爆炸
    torque_clip: float = 50.0
    # 是否包含动能梯度力（几何力）
    # 在标准 Euler-Poincaré 框架中，变惯量效应已通过锁定惯量自动处理，
    # 不需要额外的几何力。设为 False 以禁用（默认），设为 True 以启用。
    include_kinetic_gradient: bool = False
    # 是否动态更新惯性张量。设为 True 时使用锁定惯性（默认），
    # 设为 False 时使用传入的固定惯性（用于调试/测试）。
    update_inertia: bool = True


class ContactSplittingIntegrator:
    def __init__(
        self,
        force_fn: ForceFn,
        potential_fn: PotentialFn,
        gamma_fn: GammaFn,
        config: IntegratorConfig | None = None,
    ) -> None:
        self.force_fn = force_fn
        self.potential_fn = potential_fn
        self.gamma_fn = gamma_fn
        self.config = config or IntegratorConfig()

    def _coadjoint_transport(self, xi: ArrayLike, m: ArrayLike, dt: float) -> NDArray[np.float64]:
        """
        计算动量的共伴随输运 Ad^*_{f^{-1}}(m)，其中 f = exp(dt * xi)。
        
        理论依据 (积分器.md)：
        在离散 Euler-Poincaré 方程中，动量通过群元素的逆作用传播：
            μ_k = Ad^*_{f_{k-1}^{-1}}(μ_{k-1}) + ...
        
        对于 SL(2,R) 使用 trace pairing <A,B> = Tr(AB)：
            Ad^*_g(M) = g^{-1} M g
        
        所以 Ad^*_{f^{-1}}(M) = f M f^{-1} = g M g^{-1}
        """
        g = exp_sl2(xi, dt)  # g = f = exp(dt * xi)
        g_inv = np.linalg.inv(g)
        M = vec_to_matrix(m)
        # Ad^*_{f^{-1}}(M) = f M f^{-1} = g M g^{-1}
        M_new = g @ M @ g_inv
        return matrix_to_vec(M_new)

    def coadjoint_transport(self, xi: ArrayLike, m: ArrayLike, dt: float) -> NDArray[np.float64]:
        return self._coadjoint_transport(xi, m, dt)

    def _alpha(self, gamma_val: ArrayLike, dt: float, use_cayley: bool = True) -> ArrayLike:
        """
        计算离散耗散因子 α。
        
        理论依据 (积分器.md)：
        使用 Cayley 型离散化以保证几何精确性：
            α = (1 - h*γ/2) / (1 + h*γ/2)
        
        这与文档中的离散接触演化方程一致，保证了耗散的几何精确性。
        
        当 use_cayley=False 时，使用指数离散化 α = exp(-h*γ)（向后兼容）。
        """
        gamma = np.asarray(gamma_val, dtype=float)
        h = dt
        
        if use_cayley:
            # Cayley 型离散化：α = (1 - hγ/2) / (1 + hγ/2)
            # 这是 exp(-hγ) 的一阶 Padé 近似，保持接触结构
            numerator = 1.0 - 0.5 * h * gamma
            denominator = 1.0 + 0.5 * h * gamma
            # 确保分母不为零
            denominator = np.maximum(denominator, 1e-12)
            return numerator / denominator
        else:
            # 指数离散化（向后兼容）
            return np.exp(-h * gamma)

    def _max_hyperbolic_disp(self, z_old: NDArray[np.complex128], z_new: NDArray[np.complex128]) -> float:
        disk_old = cayley_uhp_to_disk(z_old)
        disk_new = cayley_uhp_to_disk(z_new)
        max_d = 0.0
        for a, b in zip(disk_old.flat, disk_new.flat):
            d = poincare_distance_disk(a, b)
            if d > max_d:
                max_d = d
        return max_d

    def _stabilize_uhp(self, z: NDArray[np.complex128]) -> NDArray[np.complex128]:
        disk = clamp_disk_radius(cayley_uhp_to_disk(z))
        z_safe = cayley_disk_to_uhp(disk)
        z_safe = clamp_uhp(z_safe)
        return np.nan_to_num(z_safe, nan=0.0)

    def _backtrack_dt(self, state: IntegratorState, torque: ArrayLike, xi: ArrayLike) -> tuple[float, NDArray[np.complex128]]:
        xi_norm = float(np.linalg.norm(xi))
        torque_norm = float(np.linalg.norm(torque))
        vals = np.linalg.eigvalsh(state.I)
        eig_min = float(max(vals.min(), 1e-6))
        inv_spec = 1.0 / eig_min
        denom = max(1e-9, xi_norm + self.config.torque_dt_scale * inv_spec * torque_norm)
        dt_geo = self.config.eps_disp / denom
        gamma_field = self.gamma_fn(state.z_uhp)
        gamma_curr = float(np.mean(gamma_field)) if np.asarray(gamma_field).size else 1.0
        SAFETY_FACTOR = 0.8
        dt_thermo = SAFETY_FACTOR / (gamma_curr + 1e-9)
        if gamma_curr > 0:
            dt_thermo = min(dt_thermo, 2.0 / gamma_curr)
        dt = min(self.config.max_dt, dt_geo, dt_thermo)
        dt = max(dt, self.config.min_dt)
        z_trial = state.z_uhp
        for _ in range(5):
            g_step = exp_sl2(xi, dt)
            z_trial = mobius_action_matrix(g_step, state.z_uhp)
            z_trial = self._stabilize_uhp(z_trial)
            disp = self._max_hyperbolic_disp(state.z_uhp, z_trial)
            if np.isnan(disp) or np.isinf(disp) or disp > self.config.eps_disp * 1.5:
                dt *= 0.5
                if dt < self.config.min_dt:
                    dt = self.config.min_dt
                    break
            else:
                break
        return dt, z_trial

    def step(self, state: IntegratorState) -> IntegratorState:
        xi_mag = float(np.linalg.norm(state.xi))
        if xi_mag > 1e4:
            state.xi = state.xi * 0.01
            state.m = apply_inertia(state.I, state.xi)

        gamma_field_prev = self.gamma_fn(state.z_uhp)
        gamma_prev = float(np.mean(gamma_field_prev)) if np.asarray(gamma_field_prev).size else 0.0
        xi_prev = state.xi
        m_prev = state.m if state.m is not None else apply_inertia(state.I, xi_prev)

        z_guess = state.z_uhp
        action_guess = state.action
        xi_new = xi_prev
        m_new = m_prev

        # determine step size
        forces_pot = self.force_fn(z_guess, action_guess)
        
        # 可选：几何力（动能梯度力）
        # 注意：在标准 Euler-Poincaré 框架中，变惯量效应已通过锁定惯量的
        # 动态更新自动处理。添加几何力可能导致能量不守恒。
        # 默认禁用此项，除非有特殊物理需求。
        if self.config.include_kinetic_gradient:
            grad_K = compute_kinetic_energy_gradient(z_guess, xi_prev)
            forces_kin = -grad_K  # 几何力 = -∇K
            forces_total = forces_pot + forces_kin
        else:
            forces_total = forces_pot
        
        torque0 = aggregate_torque(z_guess, forces_total)
        dt, z_guess = self._backtrack_dt(state, torque0, xi_prev)
        alpha_prev = self._alpha(gamma_prev, dt)

        for _ in range(max(1, self.config.fixed_point_iters)):
            gamma_field = self.gamma_fn(z_guess)
            gamma_eff = float(np.mean(gamma_field)) if np.asarray(gamma_field).size else 0.0
            alpha_prev = self._alpha(gamma_eff, dt)

            forces_pot = self.force_fn(z_guess, action_guess)
            
            # 可选几何力
            if self.config.include_kinetic_gradient:
                grad_K = compute_kinetic_energy_gradient(z_guess, xi_prev)
                forces_kin = -grad_K
                forces_total = forces_pot + forces_kin
            else:
                forces_total = forces_pot

            if np.ndim(gamma_field) > 0:
                # 阻尼作用于总有效力
                forces_total = forces_total / (1.0 + 0.5 * dt * gamma_field)
            
            torque = aggregate_torque(z_guess, forces_total)
            
            # 力矩裁剪：防止加速度爆炸
            torque_norm = float(np.linalg.norm(torque))
            if torque_norm > self.config.torque_clip:
                torque = torque * (self.config.torque_clip / torque_norm)
            
            m_advected = self._coadjoint_transport(xi_prev, m_prev, dt)
            m_new = alpha_prev * m_advected + dt * torque
            
            # 惯性更新：可配置是否使用锁定惯性
            if self.config.update_inertia:
                I_new = locked_inertia_uhp(z_guess)
                state.I = I_new
            else:
                I_new = state.I  # 使用传入的惯性
            
            gamma_curr = gamma_eff
            xi_new = invert_inertia(I_new, m_new)
            
            # (Limiting logic ...)
            max_xi_norm = 10.0 / max(dt, 1e-8)
            xi_norm_val = float(np.linalg.norm(xi_new))
            if xi_norm_val > max_xi_norm:
                xi_new = xi_new * (max_xi_norm / xi_norm_val)
                m_new = apply_inertia(I_new, xi_new)
            
            if xi_norm_val > self.config.xi_clip:
                xi_new = xi_new * (self.config.xi_clip / xi_norm_val)
                m_new = apply_inertia(I_new, xi_new)
            
            step_norm = float(np.linalg.norm(xi_new)) * dt
            if step_norm > self.config.step_clip:
                scale = self.config.step_clip / max(step_norm, 1e-8)
                xi_new = xi_new * scale
                m_new = apply_inertia(I_new, xi_new)

            g_step = exp_sl2(xi_new, dt)
            z_guess = mobius_action_matrix(g_step, state.z_uhp)
            z_guess = self._stabilize_uhp(z_guess)
            
            if self.config.relax_eta > 0.0:
                relax_force = self.force_fn(z_guess, action_guess)
                y = np.maximum(np.imag(z_guess), 1e-6)
                relax_step = (self.config.relax_eta * dt) * (relax_force * (y * y))
                z_relaxed = z_guess + relax_step
                z_guess = self._stabilize_uhp(z_relaxed)

            # 接触作用量（惊奇 Z）更新
            # 理论依据 (积分器.md)：
            # Z_{k+1} = α_k * Z_k + h/(1 + hγ/2) * (T_k - V_k)
            # 其中 α_k = (1 - hγ/2)/(1 + hγ/2) 是 Cayley 型离散耗散因子
            T = 0.5 * float(xi_new @ (state.I @ xi_new))
            V = float(self.potential_fn(z_guess, action_guess))
            denom = 1.0 + 0.5 * dt * gamma_curr
            # 使用与 _alpha 一致的 Cayley 型离散化
            alpha_action = self._alpha(gamma_curr, dt, use_cayley=True)
            action_guess = alpha_action * state.action + dt / denom * (T - V)

        m_out = apply_inertia(state.I, xi_new)
        return IntegratorState(
            z_uhp=z_guess,
            xi=xi_new,
            I=state.I,
            m=m_out,
            action=action_guess,
            dt_last=dt,
            gamma_last=gamma_curr,
        )

"""Simulation driver for self-organization on the Poincaré disk."""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal
import time
from tqdm import tqdm

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    cayley_disk_to_uhp,
    cayley_uhp_to_disk,
    hyperbolic_karcher_mean_disk,
    sample_disk_hyperbolic,
)
from .lie import moebius_flow
from .metrics import assign_nearest, pairwise_poincare
from .diamond import aggregate_torque
from .inertia import apply_inertia, compute_kinetic_energy_gradient_hyperboloid
from .geometry import (
    cayley_uhp_to_disk,
    cayley_disk_to_uhp,
    uhp_to_hyperboloid,
)
from .group_integrator import (
    GroupContactIntegrator, 
    GroupIntegratorConfig, 
    GroupIntegratorState,
    create_initial_group_state,
)
from .potential import PotentialOracle, build_baseline_potential, HierarchicalSoftminPotential


@dataclasses.dataclass
class SimulationConfig:
    n_points: int = 50
    steps: int = 1000
    max_rho: float = 3.0
    enable_diamond: bool = True
    initial_xi: tuple[float, float, float] = (0.1, 0.0, 0.0)
    force_clip: float | None = None  # clip magnitude of forces/gradients
    use_hyperbolic_centroid: bool = True
    use_hyperbolic_centroid: bool = True
    eps_dt: float = 0.02  # reduced from 0.1 for stability
    max_dt: float = 0.005 # reduced from 0.02 for stability (stiffness lambda~2500 require dt < 1/sqrt(lambda) ~ 0.02)
    min_dt: float = 1e-5
    disable_dissipation: bool = False
    v_floor: float = 5e-2
    beta: float = 1.0
    beta_eta: float = 0.5
    beta_target_bridge: float = 0.25
    beta_min: float = 0.5
    beta_max: float = 3.0
    use_parent_bias: bool = True
    use_parent_bias: bool = True
    log_interval: int = 20  # Log every N steps to save time

@dataclasses.dataclass
class SimulationLog:
    steps: list[int]  # log step index
    energy: list[float]  # per-point
    potential: list[float]  # per-point
    kinetic: list[float]  # per-point
    z_series: list[float]
    grad_norm: list[float]
    xi_norm: list[float]
    positions_disk: list[NDArray[np.complex128]]
    centroid_disk: list[complex]
    p_proxy: list[float]
    q_proj: list[tuple[float, float]]  # (Re, Im) of centroid on disk
    p_vec: list[NDArray[np.float64]]  # full momentum vector (u,v,w)
    action: list[float]
    residual_contact: list[float]
    residual_momentum: list[float]
    dt_series: list[float]
    gamma_series: list[float]
    gap_median: list[float]
    gap_frac_small: list[float]
    gap_path_frac_1e3: list[float]
    gap_path_frac_1e2: list[float]
    gap_leaf_q: list[tuple[float, float, float, float, float]]  # min,q25,median,q75,max
    gap_path_q: list[tuple[float, float, float, float, float]]
    V_ent: list[float]
    bridge_ratio: list[float]
    ratio_break: list[float]
    beta_series: list[float]
    torque_norm: list[float]
    inertia_eig_min: list[float]
    inertia_eig_max: list[float]
    rigid_speed2: list[float]
    relax_speed2: list[float]
    total_speed2: list[float]


def make_force_fn(potential: PotentialOracle) -> Callable[[ArrayLike, float], NDArray[np.complex128]]:
    def force(z_uhp: ArrayLike, action: float) -> NDArray[np.complex128]:
        return -potential.dV_dz(z_uhp, action)

    return force


def _hyperboloid_tangent_to_uhp_force(
    F_h: NDArray[np.float64],
    h: NDArray[np.float64],
    z_uhp: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    将Hyperboloid切向量转换为UHP复数力
    
    方法：使用数值微分推导坐标变换的Jacobian
    
    对于坐标映射 z = z(h)，力变换为：
        F_z = (∂z/∂h)^T · F_h
    
    参数：
        F_h: Hyperboloid切向量 (..., 3)
        h: Hyperboloid坐标 (..., 3)
        z_uhp: 对应的UHP坐标 (...)
        
    返回：
        UHP复数力 (...)
    """
    from .hyperboloid import hyperboloid_to_disk
    
    F_h_arr = np.asarray(F_h, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    
    if h_arr.ndim == 1:
        h_arr = h_arr.reshape(1, 3)
        F_h_arr = F_h_arr.reshape(1, 3)
    
    # Vectorized Jacobian calculation: F_z = J^T · F_h (Vector pushforward)
    # F_uhp = F_h[0] * dz/dX + F_h[1] * dz/dY + F_h[2] * dz/dT
    
    eps = 1e-6
    
    # Helper to map batch of h to z_uhp
    def map_h_to_z(h_batch):
        z_d = hyperboloid_to_disk(h_batch)
        return cayley_disk_to_uhp(z_d)

    # Compute partial derivatives via central difference
    # X component
    h_plus = h_arr.copy(); h_plus[:, 0] += eps
    h_minus = h_arr.copy(); h_minus[:, 0] -= eps
    dz_dX = (map_h_to_z(h_plus) - map_h_to_z(h_minus)) / (2 * eps)
    
    # Y component
    h_plus = h_arr.copy(); h_plus[:, 1] += eps
    h_minus = h_arr.copy(); h_minus[:, 1] -= eps
    dz_dY = (map_h_to_z(h_plus) - map_h_to_z(h_minus)) / (2 * eps)
    
    # T component
    h_plus = h_arr.copy(); h_plus[:, 2] += eps
    h_minus = h_arr.copy(); h_minus[:, 2] -= eps
    dz_dT = (map_h_to_z(h_plus) - map_h_to_z(h_minus)) / (2 * eps)
    
    # Linear combination
    F_uhp = (F_h_arr[:, 0] * dz_dX + 
             F_h_arr[:, 1] * dz_dY + 
             F_h_arr[:, 2] * dz_dT)
            
    return F_uhp.reshape(z_uhp.shape)

def run_simulation_group(
    potential: PotentialOracle | None = None,
    config: SimulationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> SimulationLog:
    """
    使用群积分器的模拟驱动
    
    核心改进：
    1. 在 SL(2,R) 群上积分，避免边界问题
    2. 在 Hyperboloid 坐标系中评估物理量，避免度量发散
    3. 长周期稳定性显著提升
    
    参数：
        potential: 势能函数
        config: 模拟配置
        rng: 随机数生成器
        
    返回：
        SimulationLog 日志对象
    """
    cfg = config or SimulationConfig()
    rng = np.random.default_rng() if rng is None else rng

    pot = potential
    if pot is None:
        # Check if we should build a hierarchical one or baseline
        # Default fallback is baseline, but if we want to test hierarchy emergence:
        # Actually build_baseline_potential returns GaussianWellsPotential which is simpler.
        # Let's check if the user intended hierarchy.
        # For compatibility with existing tests, we stick to baseline unless specified?
        # But wait, run_simulation_group is generic.
        # Let's update `build_baseline_potential` calls in *tests* primarily.
        # Here we just use the provided potential.
        pot = build_baseline_potential(rng=rng)
    
    if hasattr(pot, "anneal_beta"):
        pot.anneal_beta = 0.0
    
    # 初始化
    z_disk0 = sample_disk_hyperbolic(n=cfg.n_points, max_rho=cfg.max_rho, rng=rng)
    z_uhp0 = cayley_disk_to_uhp(z_disk0)
    
    # 创建群积分器状态
    state = create_initial_group_state(z_uhp0, cfg.initial_xi)

    # 力函数
    if cfg.enable_diamond:
        force_fn = make_force_fn(pot)
    else:
        force_fn = lambda z, action: np.zeros_like(z, dtype=np.complex128)
    
    # 创建群积分器
    integrator = GroupContactIntegrator(
        force_fn=force_fn,
        potential_fn=lambda z, action: pot.potential(z, action),
        gamma_fn=None,  # 使用内置的 Hyperboloid 阻尼
        config=GroupIntegratorConfig(
            eps_disp=cfg.eps_dt,
            max_dt=cfg.max_dt,
            min_dt=cfg.min_dt,
            v_floor=cfg.v_floor,
            use_hyperboloid_gamma=True,
            gamma_scale=2.0 if not cfg.disable_dissipation else 0.0,
        ),
    )

    log = SimulationLog(
        steps=[],
        energy=[],
        potential=[],
        kinetic=[],
        z_series=[],
        grad_norm=[],
        xi_norm=[],
        positions_disk=[],
        centroid_disk=[],
        p_proxy=[],
        q_proj=[],
        p_vec=[],
        action=[],
        residual_contact=[],
        residual_momentum=[],
        dt_series=[],
        gamma_series=[],
        gap_median=[],
        gap_frac_small=[],
        gap_path_frac_1e3=[],
        gap_path_frac_1e2=[],
        gap_leaf_q=[],
        gap_path_q=[],
        V_ent=[],
        bridge_ratio=[],
        ratio_break=[],
        beta_series=[],
        torque_norm=[],
        inertia_eig_min=[],
        inertia_eig_max=[],
        rigid_speed2=[],
        relax_speed2=[],
        total_speed2=[],
    )

    # Initialize performance tracking
    step_times = []
    pbar = tqdm(range(cfg.steps), desc="Running Simulation")
    
    # Track accumulated time for average output
    total_compute_time = 0.0

    # Initialize prev_state for first step
    prev_state = state

    try:
        for step in pbar:
            t_start = time.perf_counter()
            _ = step # keep compatible if variable used
            
            # 惰性求值当前位置
            z_current = state.z_uhp
            h_current = state.h
            I_current = state.I
            
            # --- Periodic Updates (Lambda/Bridge) ---
            # Update lambda potential params periodically if needed (can be every step or decimated, usually every step for accuracy)
            if isinstance(pot, HierarchicalSoftminPotential) and hasattr(pot, "update_lambda"):
                pot.update_lambda(z_current, state.action)

            # --- Core Physics Step ---
            # 计算总有效力 = 势能力 + 几何力
            # 理论依据：理论基础-3.md 引理 3.1.1 和定理 3.2
            # F_total = -∇V_FEP + F_geom
            # 其中 F_geom = -∇_z K(z,ξ) 是动能对位置的梯度
            # ============================================================
            
            # 1. 计算势能力（原有实现）
            grad_V = pot.dV_dz(z_current, state.action)
            if cfg.force_clip is not None:
                mag = np.abs(grad_V)
                scale = np.maximum(1.0, mag / cfg.force_clip)
                grad_V = grad_V / scale
            F_potential = -grad_V
            
            # 2. 计算几何力（Hyperboloid坐标系，避免边界奇异）
            # 在Hyperboloid上计算动能梯度，然后转换为UHP复数力
            F_geom_h = -compute_kinetic_energy_gradient_hyperboloid(
                h_current, state.xi, weights=None
            )
            
            # 将Hyperboloid切向量转换为UHP复数力
            F_geom_uhp = _hyperboloid_tangent_to_uhp_force(
                F_geom_h, h_current, z_current
            )
            
            # 3. 总力 = 势能力 + 几何力
            forces = F_potential + F_geom_uhp
            
            # 速度诊断
            y = np.maximum(np.imag(z_current), 1e-9)
            rigid_vel = moebius_flow(state.xi, z_current)
            relax_eta = getattr(integrator.config, "relax_eta", 0.0) if hasattr(integrator.config, "relax_eta") else 0.0
            relax_vel = relax_eta * forces
            rigid_speed2 = float(np.mean((np.abs(rigid_vel) ** 2) / (y * y)))
            relax_speed2 = float(np.mean((np.abs(relax_vel) ** 2) / (y * y)))
            total_speed2 = float(np.mean((np.abs(rigid_vel + relax_vel) ** 2) / (y * y)))

            torque_now = aggregate_torque(z_current, forces)
            
            # --- Expensive Logging (Decimated) ---
            if step % cfg.log_interval == 0 or step == cfg.steps - 1:
                # 能量计算
                potential_energy = pot.potential(z_current, state.action)
                n_pts = z_current.size
                potential_energy_mean = potential_energy / max(n_pts, 1)
                kinetic_energy = 0.5 * float(state.xi @ (I_current @ state.xi))
                kinetic_energy_mean = kinetic_energy / max(n_pts, 1)
                grad_norm = float(np.linalg.norm(grad_V))
                eigs = np.linalg.eigvalsh(I_current)

                # 记录日志
                log.steps.append(step)
                log.energy.append(kinetic_energy_mean + potential_energy_mean)
                log.potential.append(potential_energy_mean)
                log.kinetic.append(kinetic_energy_mean)
                log.z_series.append(state.action)
                log.grad_norm.append(grad_norm)
                log.xi_norm.append(float(np.linalg.norm(state.xi)))
                log.torque_norm.append(float(np.linalg.norm(torque_now)))
                log.inertia_eig_min.append(float(eigs.min()))
                log.inertia_eig_max.append(float(eigs.max()))
                
                disk_pos = cayley_uhp_to_disk(z_current)
                log.positions_disk.append(disk_pos)
                centroid = (
                    hyperbolic_karcher_mean_disk(disk_pos)
                    if cfg.use_hyperbolic_centroid
                    else complex(disk_pos.mean())
                )
                log.centroid_disk.append(centroid)
                momentum = apply_inertia(I_current, state.xi)
                log.p_proxy.append(float(np.linalg.norm(momentum)))
                log.p_vec.append(momentum.copy())
                log.q_proj.append((float(np.real(disk_pos.mean())), float(np.imag(disk_pos.mean()))))
                log.action.append(state.action)
                
                # Gap/entropy 诊断
                if hasattr(pot, "gap_stats"):
                    gs = pot.gap_stats(z_current, state.action)
                    log.gap_median.append(gs.get("gap_path_min_median", 0.0))
                    log.gap_frac_small.append(gs.get("gap_path_min_frac_small", 0.0))
                    log.gap_path_frac_1e3.append(gs.get("gap_path_frac_1e3", 0.0))
                    log.gap_path_frac_1e2.append(gs.get("gap_path_frac_1e2", 0.0))
                    log.gap_leaf_q.append((
                        float(gs.get("gap_leaf_min", 0.0)),
                        float(gs.get("gap_leaf_q25", 0.0)),
                        float(gs.get("gap_leaf_median", 0.0)),
                        float(gs.get("gap_leaf_q75", 0.0)),
                        float(gs.get("gap_leaf_max", 0.0)),
                    ))
                    log.gap_path_q.append((
                        float(gs.get("gap_path_min", 0.0)),
                        float(gs.get("gap_path_q25", 0.0)),
                        float(gs.get("gap_path_min_median", 0.0)),
                        float(gs.get("gap_path_q75", 0.0)),
                        float(gs.get("gap_path_max", 0.0)),
                    ))
                else:
                    log.gap_median.append(0.0)
                    log.gap_frac_small.append(0.0)
                    log.gap_path_frac_1e3.append(0.0)
                    log.gap_path_frac_1e2.append(0.0)
                    log.gap_leaf_q.append((0, 0, 0, 0, 0))
                    log.gap_path_q.append((0, 0, 0, 0, 0))
                
                if hasattr(pot, "entropy_energy"):
                    ent = pot.entropy_energy(z_current, state.action)
                    log.V_ent.append(ent.get("V_ent", 0.0))
                else:
                    log.V_ent.append(0.0)
                
                # Bridge ratio - Expensive, decimate further (every 5 * log_interval)
                if isinstance(pot, HierarchicalSoftminPotential):
                    if hasattr(pot, "forces_decomposed"):
                        fdec = pot.forces_decomposed(z_current, state.action)
                        grad_tot = fdec.get("grad_total", np.zeros_like(z_current))
                        grad_ent = fdec.get("grad_entropy", np.zeros_like(z_current))
                        norm_tot = float(np.linalg.norm(grad_tot))
                        norm_ent = float(np.linalg.norm(grad_ent))
                        ratio_break = norm_ent / (norm_tot + 1e-12)
                        log.ratio_break.append(ratio_break)
                    else:
                        log.ratio_break.append(0.0)
                    
                    z_disk_bridge = cayley_uhp_to_disk(z_current)
                    centers_disk = cayley_uhp_to_disk(pot.centers)
                    labels_anchor = assign_nearest(z_disk_bridge, centers_disk)
                    D_full = pairwise_poincare(z_disk_bridge)
                    n = D_full.shape[0]
                    edge_total = edge_cross = 0
                    k = 3
                    # Only check top-k neighbors
                    nn = np.argsort(D_full, axis=1)[:, 1 : k + 1]
                    for i_idx in range(n):
                        for j_idx in nn[i_idx]:
                            edge_total += 1
                            if labels_anchor[i_idx] != labels_anchor[j_idx]:
                                edge_cross += 1
                                
                    log.bridge_ratio.append(edge_cross / edge_total if edge_total else 0.0)
                    bridge_val = edge_cross / edge_total if edge_total else 0.0
                    pot.bridge_ema = (1 - cfg.beta_eta) * pot.bridge_ema + cfg.beta_eta * bridge_val
                    pot.beta = float(np.clip(pot.beta + cfg.beta_eta * (pot.bridge_ema - cfg.beta_target_bridge), cfg.beta_min, cfg.beta_max))
                    log.beta_series.append(pot.beta)
                else:
                    log.bridge_ratio.append(0.0)
                    log.ratio_break.append(0.0)
                    log.beta_series.append(0.0)
                        
                log.rigid_speed2.append(rigid_speed2)
                log.relax_speed2.append(relax_speed2)
                log.total_speed2.append(total_speed2)

                # 残差计算
                # Need prev_state, but that's updated every step.
                # Here we only log if we are logging.
                # However, residuals need dt_last from correct step.
                # state has .dt_last.
                
                dt_used = state.dt_last
                gamma_used = state.gamma_last
                alpha_prev = (1.0 - 0.5 * dt_used * gamma_used) / (1.0 + 0.5 * dt_used * gamma_used)
                
                I_new = state.I
                # Note: T_new and V_new should be consistent with state, but expensive V calls.
                # Re-calculate minimal needed or just use current vars?
                # We already calculated kinetic_energy_mean above inside if block.
                # Re-use values?
                
                # Careful: r_contact logic depends on state and prev_state actions.
                # prev_state is updated every step below.
                # So we can compute residuals here for Logging purposes.
                
                # Re-calc just for residual logging
                T_new = 0.5 * float(state.xi @ (I_new @ state.xi))
                V_new = float(pot.potential(state.z_uhp, state.action))
                
                Ld = dt_used * (T_new - V_new) - 0.5 * dt_used * gamma_used * (prev_state.action + state.action)
                r_contact = (state.action - prev_state.action) - Ld
                log.residual_contact.append(float(r_contact))

                m_prev = apply_inertia(prev_state.I, prev_state.xi)
                m_adv = integrator.coadjoint_transport(prev_state.xi, m_prev, dt_used)
                # torque_new already calc'd outside if? NO.
                # torque_now is outside. use it.
                I_prev = prev_state.I
                mom_res_vec = apply_inertia(I_prev, state.xi) - alpha_prev * m_adv - dt_used * torque_now
                log.residual_momentum.append(float(np.linalg.norm(mom_res_vec)))
                log.dt_series.append(dt_used)
                log.gamma_series.append(gamma_used)
            
            # 保存前一状态用于残差计算
            prev_state = state
            
            # 执行一步积分
            state = integrator.step(state)
            
            # Performance Tracking
            t_end = time.perf_counter()
            dt_step = (t_end - t_start) * 1000.0 # ms
            step_times.append(dt_step)
            total_compute_time += dt_step

            # Update progress bar every 10 steps to reduce overhead
            if step % 10 == 0:
                avg_dt = np.mean(step_times[-100:]) if step_times else 0.0
                pbar.set_description(f"Step {step}: H={log.energy[-1]:.2f}, |ξ|={log.xi_norm[-1]:.2f}, {avg_dt:.1f}ms/step") if log.energy else pbar.set_description(f"Step {step}: {avg_dt:.1f}ms/step")

    except KeyboardInterrupt:
        print(f"\nSimulation interrupted manually at step {len(step_times)}")
    
    if step_times:
        avg_total = np.mean(step_times)
        print(f"\nPerformance Summary: Average {avg_total:.2f} ms/step over {len(step_times)} steps.")

    return log

"""Legacy simulation driver using the contact splitting integrator."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..diamond import aggregate_torque
from ..geometry import (
    cayley_disk_to_uhp,
    cayley_uhp_to_disk,
    hyperbolic_karcher_mean_disk,
    sample_disk_hyperbolic,
)
from ..inertia import apply_inertia, locked_inertia_uhp
from ..lie import moebius_flow
from ..metrics import assign_nearest, pairwise_poincare
from ..potential import PotentialOracle, build_baseline_potential, HierarchicalSoftminPotential
from .integrator import ContactSplittingIntegrator, IntegratorConfig, IntegratorState
from ..simulation import SimulationConfig, SimulationLog, make_force_fn


def gamma_critical_inertia(
    z_uhp: ArrayLike,
    I: ArrayLike | None = None,
    scale: float = 2.0,
    mode: str = "adaptive",
) -> NDArray[np.float64] | float:
    """
    几何临界阻尼：根据公理 4.2 实现 γ(q) ∝ √(λ_max(g(q))).
    Retained for the legacy UHP contact integrator.
    """
    if mode == "fixed":
        return scale

    z_arr = np.asarray(z_uhp, dtype=np.complex128)

    if mode == "spectral" and I is not None:
        I_arr = np.asarray(I, dtype=float)
        eigs = np.linalg.eigvalsh(I_arr)
        lambda_max = float(np.max(eigs))
        gamma_spec = scale * np.sqrt(max(lambda_max, 1e-6))
        return min(gamma_spec, 10.0)

    if mode == "adaptive":
        z_disk = cayley_uhp_to_disk(z_arr)
        r = np.abs(z_disk)

        gamma_base = scale * (0.5 + 0.5 * np.tanh(r))

        viscosity = np.zeros_like(r, dtype=float)
        mask = r > 0.95
        if np.any(mask):
            viscosity[mask] = 2.0 * np.exp((r[mask] - 0.95) * 8.0)

        gamma_field = gamma_base + viscosity
        gamma_field = np.clip(gamma_field, 0.3, 10.0)
        return gamma_field

    # mode == "geometric"
    y = np.maximum(np.imag(z_arr), 1e-6)

    gamma_base = np.clip(scale / y, scale * 0.1, scale * 5.0)

    z_disk = cayley_uhp_to_disk(z_arr)
    r = np.abs(z_disk)

    viscosity = np.zeros_like(r, dtype=float)
    mask = r > 0.95
    if np.any(mask):
        viscosity[mask] = 3.0 * np.exp((r[mask] - 0.95) * 10.0)

    gamma_field = gamma_base + viscosity

    gamma_field = np.minimum(gamma_field, 20.0)

    return gamma_field


def run_simulation(
    potential: PotentialOracle | None = None,
    config: SimulationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> SimulationLog:
    """Legacy UHP contact-splitting simulation loop."""
    cfg = config or SimulationConfig()
    rng = np.random.default_rng() if rng is None else rng

    pot = potential or build_baseline_potential(rng=rng)
    # Force-disable annealing to keep wells active throughout simulation
    if hasattr(pot, "anneal_beta"):
        pot.anneal_beta = 0.0
    z_disk0 = sample_disk_hyperbolic(n=cfg.n_points, max_rho=cfg.max_rho, rng=rng)
    z_uhp0 = cayley_disk_to_uhp(z_disk0)

    I = locked_inertia_uhp(z_uhp0)
    xi0 = np.array(cfg.initial_xi, dtype=float)
    m0 = apply_inertia(I, xi0)
    state = IntegratorState(z_uhp=z_uhp0, xi=xi0, I=I, m=m0, action=0.0)

    if cfg.enable_diamond:
        force_fn = make_force_fn(pot)
    else:
        force_fn = lambda z, action: np.zeros_like(z, dtype=np.complex128)  # Diamond off

    # 创建闭包以在 gamma 计算中访问当前惯性张量
    def gamma_fn_with_inertia(z_uhp: ArrayLike) -> NDArray[np.float64] | float:
        if cfg.disable_dissipation:
            return 0.0
        # 使用自适应模式的临界阻尼（推荐，避免过度阻尼）
        return gamma_critical_inertia(z_uhp, I=state.I, mode="adaptive")

    integrator = ContactSplittingIntegrator(
        force_fn=force_fn,
        potential_fn=lambda z, action: pot.potential(z, action),
        gamma_fn=gamma_fn_with_inertia,
        config=IntegratorConfig(eps_disp=cfg.eps_dt, max_dt=cfg.max_dt, min_dt=cfg.min_dt, v_floor=cfg.v_floor),
    )

    log = SimulationLog(
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

    for _ in range(cfg.steps):
        # update inertia endogenously from geometry
        state.I = locked_inertia_uhp(state.z_uhp)
        state.m = apply_inertia(state.I, state.xi)
        if isinstance(pot, HierarchicalSoftminPotential) and hasattr(pot, "update_lambda"):
            pot.update_lambda(state.z_uhp, state.action)

        grad = pot.dV_dz(state.z_uhp, state.action)
        if cfg.force_clip is not None:
            mag = np.abs(grad)
            scale = np.maximum(1.0, mag / cfg.force_clip)
            grad = grad / scale
        forces = -grad
        # Hyperbolic velocity diagnostics (per-point mean squared speed)
        y = np.maximum(np.imag(state.z_uhp), 1e-9)
        rigid_vel = moebius_flow(state.xi, state.z_uhp)
        relax_eta = getattr(integrator.config, "relax_eta", 0.0)
        relax_vel = relax_eta * forces
        rigid_speed2 = float(np.mean((np.abs(rigid_vel) ** 2) / (y * y)))
        relax_speed2 = float(np.mean((np.abs(relax_vel) ** 2) / (y * y)))
        total_speed2 = float(np.mean((np.abs(rigid_vel + relax_vel) ** 2) / (y * y)))

        torque_now = aggregate_torque(state.z_uhp, forces)
        potential_energy = pot.potential(state.z_uhp, state.action)
        n_pts = state.z_uhp.size
        potential_energy_mean = potential_energy / max(n_pts, 1)
        kinetic_energy = 0.5 * float(state.xi @ (state.I @ state.xi))
        kinetic_energy_mean = kinetic_energy / max(n_pts, 1)
        grad_norm = float(np.linalg.norm(grad))
        eigs = np.linalg.eigvalsh(state.I)

        log.energy.append(kinetic_energy_mean + potential_energy_mean)
        log.potential.append(potential_energy_mean)
        log.kinetic.append(kinetic_energy_mean)
        log.z_series.append(state.action)
        log.grad_norm.append(grad_norm)
        log.xi_norm.append(float(np.linalg.norm(state.xi)))
        log.torque_norm.append(float(np.linalg.norm(torque_now)))
        log.inertia_eig_min.append(float(eigs.min()))
        log.inertia_eig_max.append(float(eigs.max()))
        disk_pos = cayley_uhp_to_disk(state.z_uhp)
        log.positions_disk.append(disk_pos)
        centroid = (
            hyperbolic_karcher_mean_disk(disk_pos)
            if cfg.use_hyperbolic_centroid
            else complex(disk_pos.mean())
        )
        log.centroid_disk.append(centroid)
        momentum = apply_inertia(state.I, state.xi)
        log.p_proxy.append(float(np.linalg.norm(momentum)))
        log.p_vec.append(momentum.copy())
        log.q_proj.append((float(np.real(disk_pos.mean())), float(np.imag(disk_pos.mean()))))
        log.action.append(state.action)
        # gap/entropy diagnostics
        if hasattr(pot, "gap_stats"):
            gs = pot.gap_stats(state.z_uhp, state.action)
            log.gap_median.append(gs.get("gap_path_min_median", 0.0))
            log.gap_frac_small.append(gs.get("gap_path_min_frac_small", 0.0))
            log.gap_path_frac_1e3.append(gs.get("gap_path_frac_1e3", 0.0))
            log.gap_path_frac_1e2.append(gs.get("gap_path_frac_1e2", 0.0))
            log.gap_leaf_q.append(
                (
                    float(gs.get("gap_leaf_min", 0.0)),
                    float(gs.get("gap_leaf_q25", 0.0)),
                    float(gs.get("gap_leaf_median", 0.0)),
                    float(gs.get("gap_leaf_q75", 0.0)),
                    float(gs.get("gap_leaf_max", 0.0)),
                )
            )
            log.gap_path_q.append(
                (
                    float(gs.get("gap_path_min", 0.0)),
                    float(gs.get("gap_path_q25", 0.0)),
                    float(gs.get("gap_path_min_median", 0.0)),
                    float(gs.get("gap_path_q75", 0.0)),
                    float(gs.get("gap_path_max", 0.0)),
                )
            )
        else:
            log.gap_median.append(0.0)
            log.gap_frac_small.append(0.0)
            log.gap_path_frac_1e3.append(0.0)
            log.gap_path_frac_1e2.append(0.0)
            log.gap_leaf_q.append((0, 0, 0, 0, 0))
            log.gap_path_q.append((0, 0, 0, 0, 0))
        if hasattr(pot, "entropy_energy"):
            ent = pot.entropy_energy(state.z_uhp, state.action)
            log.V_ent.append(ent.get("V_ent", 0.0))
        else:
            log.V_ent.append(0.0)
        # bridge ratio k=3 using anchor labels if available
        if isinstance(pot, HierarchicalSoftminPotential):
            # forces decomposition
            if hasattr(pot, "forces_decomposed"):
                fdec = pot.forces_decomposed(state.z_uhp, state.action)
                grad_tot = fdec.get("grad_total", np.zeros_like(state.z_uhp))
                grad_ent = fdec.get("grad_entropy", np.zeros_like(state.z_uhp))
                norm_tot = float(np.linalg.norm(grad_tot))
                norm_ent = float(np.linalg.norm(grad_ent))
                ratio_break = norm_ent / (norm_tot + 1e-12)
                log.ratio_break.append(ratio_break)
            else:
                log.ratio_break.append(0.0)
            z_disk = cayley_uhp_to_disk(state.z_uhp)
            centers_disk = cayley_uhp_to_disk(pot.centers)
            labels_anchor = assign_nearest(z_disk, centers_disk)
            D_full = pairwise_poincare(z_disk)
            n = D_full.shape[0]
            edge_total = edge_cross = 0
            k = 3
            for i in range(n):
                nn = np.argsort(D_full[i])[1 : k + 1]
                for j in nn:
                    edge_total += 1
                    if labels_anchor[i] != labels_anchor[j]:
                        edge_cross += 1
            log.bridge_ratio.append(edge_cross / edge_total if edge_total else 0.0)
            # update beta based on bridge via EMA-like multiplicative update
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

        prev_state = state
        state = integrator.step(state)

        # Residuals: contact Herglotz and momentum balance
        dt_used = state.dt_last
        gamma_used = state.gamma_last
        alpha_prev = (1.0 - 0.5 * dt_used * gamma_used) / (1.0 + 0.5 * dt_used * gamma_used)
        T_new = 0.5 * float(state.xi @ (state.I @ state.xi))
        V_new = float(pot.potential(state.z_uhp, state.action))
        Ld = dt_used * (T_new - V_new) - 0.5 * dt_used * gamma_used * (prev_state.action + state.action)
        r_contact = (state.action - prev_state.action) - Ld
        log.residual_contact.append(float(r_contact))

        m_prev = apply_inertia(prev_state.I, prev_state.xi)
        m_adv = integrator.coadjoint_transport(prev_state.xi, m_prev, dt_used)
        torque_new = aggregate_torque(state.z_uhp, force_fn(state.z_uhp, state.action))
        mom_res_vec = apply_inertia(prev_state.I, state.xi) - alpha_prev * m_adv - dt_used * torque_new
        log.residual_momentum.append(float(np.linalg.norm(mom_res_vec)))
        log.dt_series.append(dt_used)
        log.gamma_series.append(gamma_used)

    return log

import numpy as np

from HEI.src.hei.plot_group import plot_group_log
from HEI.src.hei.simulation import SimulationConfig, run_simulation_group
from HEI.src.hei.potential import build_hierarchical_potential


def test_plot_group_log_creates_files(tmp_path) -> None:
    rng = np.random.default_rng(0)
    pot = build_hierarchical_potential(n_points=8, depth=2, branching=2, max_rho=2.0, rng=rng)
    cfg = SimulationConfig(n_points=8, steps=10, max_rho=2.0)
    log = run_simulation_group(potential=pot, config=cfg, rng=rng)

    out_path = tmp_path / "group_plot.png"
    plot_group_log(
        energy=log.energy,
        xi_norm=log.xi_norm,
        potential=log.potential,
        kinetic=log.kinetic,
        z_series=log.z_series,
        grad_norm=log.grad_norm,
        positions_disk=log.positions_disk,
        centroid_disk=log.centroid_disk,
        p_proxy=log.p_proxy,
        q_proj=log.q_proj,
        p_vec=log.p_vec,
        out_path=out_path,
        track_points=3,
        dt_series=log.dt_series,
        gamma_series=log.gamma_series,
        residual_contact=log.residual_contact,
        residual_momentum=log.residual_momentum,
        inertia_eig_min=log.inertia_eig_min,
        inertia_eig_max=log.inertia_eig_max,
        rigid_speed2=log.rigid_speed2,
        relax_speed2=log.relax_speed2,
        total_speed2=log.total_speed2,
    )

    assert out_path.exists()
    assert (tmp_path / "group_plot_diag.png").exists()
    assert (tmp_path / "group_plot_inertia.png").exists()
    assert (tmp_path / "group_plot_speed.png").exists()

"""Visualization helpers focused on the group integrator pipeline.

This module mirrors the legacy ``archive/plot_results.py`` but is streamlined
for the group-based integrator outputs from :func:`hei.simulation.run_simulation_group`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

# Ensure headless-friendly backend for scripts/tests.
matplotlib.use("Agg")  # type: ignore
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .simulation import SimulationConfig, run_simulation_group
from .potential import build_hierarchical_potential, PotentialOracle


def _plot_disk_trajectories(ax: plt.Axes, positions_disk: Iterable[np.ndarray], track_points: int) -> None:
    """Plot trajectories on the Poincaré disk."""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

    if not positions_disk:
        return
    pos_arr = np.stack(list(positions_disk))  # (steps, N)
    n_pts = min(track_points, pos_arr.shape[1])
    cmap = plt.get_cmap("tab10", n_pts)
    for i in range(n_pts):
        traj = pos_arr[:, i]
        ax.plot(traj.real, traj.imag, color=cmap(i))
        ax.scatter(traj.real[-1], traj.imag[-1], color=cmap(i), s=10, alpha=0.8)


def plot_group_log(
    *,
    steps: list[int] | None = None,
    energy: list[float],
    xi_norm: list[float],
    potential: list[float],
    kinetic: list[float],
    z_series: list[float],
    grad_norm: list[float],
    positions_disk: list[np.ndarray],
    centroid_disk: list[complex] | None = None,
    p_proxy: list[float] | None = None,
    q_proj: list[tuple[float, float]] | None = None,
    p_vec: list[np.ndarray] | None = None,
    out_path: Path = Path("outputs/group_simulation.png"),
    track_points: int = 5,
    dt_series: list[float] | None = None,
    gamma_series: list[float] | None = None,
    residual_contact: list[float] | None = None,
    residual_momentum: list[float] | None = None,
    inertia_eig_min: list[float] | None = None,
    inertia_eig_max: list[float] | None = None,
    rigid_speed2: list[float] | None = None,
    relax_speed2: list[float] | None = None,
    total_speed2: list[float] | None = None,
) -> None:
    """Plot core diagnostics for the group integrator."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    x_axis = steps if steps is not None else np.arange(len(energy))

    axes[0, 0].plot(x_axis, energy, label="Energy")
    axes[0, 0].set_title("Total energy (per point)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_axis, xi_norm, color="tab:orange", label="||xi||")
    axes[0, 1].set_title("xi norm")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Norm")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(x_axis, grad_norm, color="tab:green", label="||grad||")
    axes[0, 2].set_title("Gradient norm")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("Norm")
    axes[0, 2].grid(True, alpha=0.3)

    # 绘制势能（左侧y轴）
    ax_left = axes[1, 0]
    ax_left.plot(x_axis, potential, label="Potential", color="tab:blue")
    ax_left.set_xlabel("Step")
    ax_left.set_ylabel("Potential energy", color="tab:blue")
    ax_left.tick_params(axis='y', labelcolor="tab:blue")
    ax_left.grid(True, alpha=0.3)
    
    # 创建右侧y轴绘制动能
    ax_right = ax_left.twinx()
    ax_right.plot(x_axis, kinetic, label="Kinetic", color="tab:orange")
    ax_right.set_ylabel("Kinetic energy", color="tab:orange")
    ax_right.tick_params(axis='y', labelcolor="tab:orange")
    
    # 合并图例
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right')
    ax_left.set_title("Energy components (per point)")

    axes[1, 1].plot(x_axis, z_series, label="Cumulative z", color="tab:red")
    axes[1, 1].set_title("Contact action Z")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Z")
    axes[1, 1].grid(True, alpha=0.3)

    _plot_disk_trajectories(axes[1, 2], positions_disk, track_points)
    axes[1, 2].set_title(f"Trajectories on disk (first {track_points})")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Optional diagnostics in separate panels to keep main figure clean
    if dt_series or gamma_series or residual_contact or residual_momentum:
        fig_diag, ax_diag = plt.subplots(2, 2, figsize=(10, 6))
        if dt_series:
            x_diag = steps if (steps is not None and len(steps) == len(dt_series)) else np.arange(len(dt_series))
            ax_diag[0, 0].plot(x_diag, dt_series, label="dt")
            ax_diag[0, 0].set_title("Step size dt")
            ax_diag[0, 0].grid(True, alpha=0.3)
        if gamma_series:
            x_diag = steps if (steps is not None and len(steps) == len(gamma_series)) else np.arange(len(gamma_series))
            ax_diag[0, 1].plot(x_diag, gamma_series, label="gamma", color="tab:purple")
            ax_diag[0, 1].set_title("Gamma")
            ax_diag[0, 1].grid(True, alpha=0.3)
        if residual_contact:
            x_diag = steps if (steps is not None and len(steps) == len(residual_contact)) else np.arange(len(residual_contact))
            ax_diag[1, 0].plot(x_diag, residual_contact, label="contact residual", color="tab:red")
            ax_diag[1, 0].set_title("Contact residual")
            ax_diag[1, 0].grid(True, alpha=0.3)
        if residual_momentum:
            x_diag = steps if (steps is not None and len(steps) == len(residual_momentum)) else np.arange(len(residual_momentum))
            ax_diag[1, 1].plot(x_diag, residual_momentum, label="momentum residual", color="tab:brown")
            ax_diag[1, 1].set_title("Momentum residual")
            ax_diag[1, 1].grid(True, alpha=0.3)
        fig_diag.tight_layout()
        diag_path = out_path.with_name(out_path.stem + "_diag.png")
        fig_diag.savefig(diag_path, dpi=200, bbox_inches="tight")
        plt.close(fig_diag)

    if inertia_eig_min and inertia_eig_max:
        fig_I, ax_I = plt.subplots(figsize=(6, 3))
        ax_I.plot(inertia_eig_min, label="lambda_min")
        ax_I.plot(inertia_eig_max, label="lambda_max")
        ax_I.set_title("Inertia eigenvalues")
        ax_I.set_xlabel("Step")
        ax_I.set_ylabel("Eigenvalue")
        ax_I.legend()
        ax_I.grid(True, alpha=0.3)
        I_path = out_path.with_name(out_path.stem + "_inertia.png")
        fig_I.tight_layout()
        fig_I.savefig(I_path, dpi=200, bbox_inches="tight")
        plt.close(fig_I)

    if rigid_speed2 and relax_speed2 and total_speed2:
        fig_v, ax_v = plt.subplots(figsize=(6, 3))
        ax_v.plot(rigid_speed2, label="rigid")
        ax_v.plot(relax_speed2, label="relax")
        ax_v.plot(total_speed2, label="total")
        ax_v.set_title("Hyperbolic speeds (mean squared)")
        ax_v.set_xlabel("Step")
        ax_v.set_ylabel("Speed^2")
        ax_v.legend()
        ax_v.grid(True, alpha=0.3)
        v_path = out_path.with_name(out_path.stem + "_speed.png")
        fig_v.tight_layout()
        fig_v.savefig(v_path, dpi=200, bbox_inches="tight")
        plt.close(fig_v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run group integrator simulation and plot diagnostics.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--n-points", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--max-rho", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/group/group_integrator.png"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pot: PotentialOracle = build_hierarchical_potential(
        n_points=args.n_points, depth=args.depth, branching=args.branching, max_rho=args.max_rho, rng=rng
    )
    cfg = SimulationConfig(n_points=args.n_points, steps=args.steps, max_rho=args.max_rho)
    log = run_simulation_group(potential=pot, config=cfg, rng=rng)

    plot_group_log(
        steps=log.steps,
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
        out_path=args.output,
        track_points=5,
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
    print(f"Saved plots to {args.output.parent.resolve()}")


if __name__ == "__main__":
    main()

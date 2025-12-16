"""Run baseline and ablation CCD simulations and plot comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .simulation import SimulationConfig, run_simulation


def run_suite(
    steps: int,
    seed: int,
    eps_dt: float,
    max_dt: float,
    min_dt: float,
) -> Dict[str, dict]:
    rng = np.random.default_rng(seed)
    configs = {
        "baseline": SimulationConfig(steps=steps, eps_dt=eps_dt, max_dt=max_dt, min_dt=min_dt),
        "no_diss": SimulationConfig(steps=steps, eps_dt=eps_dt, max_dt=max_dt, min_dt=min_dt, disable_dissipation=True),
        "no_diamond": SimulationConfig(
            steps=steps,
            eps_dt=eps_dt,
            max_dt=max_dt,
            min_dt=min_dt,
            enable_diamond=False,
            initial_xi=(0.2, 0.0, 0.0),  # small seed so motion exists
        ),
    }
    logs = {}
    for name, cfg in configs.items():
        logs[name] = run_simulation(config=cfg, rng=rng)
    return logs


def plot_comparisons(logs: Dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for name, log in logs.items():
        axes[0, 0].plot(log.energy, label=name)
    axes[0, 0].set_title("Total energy")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for name, log in logs.items():
        axes[0, 1].plot(log.xi_norm, label=name)
    axes[0, 1].set_title("xi norm")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Norm")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for name, log in logs.items():
        axes[1, 0].plot(log.z_series, label=name)
    axes[1, 0].set_title("Contact action Z")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Z")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_title("Baseline trajectories (first 5)")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    base_log = logs.get("baseline")
    if base_log and base_log.positions_disk:
        pos_arr = np.stack(base_log.positions_disk)
        n_pts = min(5, pos_arr.shape[1])
        cmap = plt.get_cmap("tab10", n_pts)
        for i in range(n_pts):
            traj = pos_arr[:, i]
            ax.plot(traj.real, traj.imag, color=cmap(i))
            ax.scatter(traj.real[-1], traj.imag[-1], color=cmap(i), s=10, alpha=0.8)
    ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCD ablation suite and plot.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eps-dt", type=float, default=1e-2, dest="eps_dt")
    parser.add_argument("--max-dt", type=float, default=5e-2, dest="max_dt")
    parser.add_argument("--min-dt", type=float, default=1e-4, dest="min_dt")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/ablation.png"), help="Output image path"
    )
    parser.add_argument("--hier", action="store_true", help="Use hierarchical potential")
    args = parser.parse_args()

    potential = None
    if args.hier:
        from .potential import build_hierarchical_potential

        potential = build_hierarchical_potential(rng=np.random.default_rng(args.seed))

    logs = run_suite(
        steps=args.steps,
        seed=args.seed,
        eps_dt=args.eps_dt,
        max_dt=args.max_dt,
        min_dt=args.min_dt,
    )
    if potential is not None:
        # override potentials for each log by re-running with seeded RNGs for reproducibility
        new_logs = {}
        rng_seed = args.seed
        for name, cfg in {
            "baseline": SimulationConfig(steps=args.steps, eps_dt=args.eps_dt, max_dt=args.max_dt, min_dt=args.min_dt),
            "no_diss": SimulationConfig(
                steps=args.steps, eps_dt=args.eps_dt, max_dt=args.max_dt, min_dt=args.min_dt, disable_dissipation=True
            ),
            "no_diamond": SimulationConfig(
                steps=args.steps,
                eps_dt=args.eps_dt,
                max_dt=args.max_dt,
                min_dt=args.min_dt,
                enable_diamond=False,
                initial_xi=(0.2, 0.0, 0.0),
            ),
        }.items():
            new_logs[name] = run_simulation(
                potential=potential,
                config=cfg,
                rng=np.random.default_rng(rng_seed),
            )
            rng_seed += 1
        logs = new_logs
    plot_comparisons(logs, args.out)


if __name__ == "__main__":
    main()

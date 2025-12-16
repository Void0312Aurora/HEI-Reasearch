"""Sweep adaptive step tolerance eps_dt to probe stability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .simulation import SimulationConfig, run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive step tolerance sweep for CCD.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--eps-list", type=float, nargs="+", default=[1e-3, 5e-3, 1e-2, 2e-2]
    )
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/eps_sweep.png"), help="Output plot"
    )
    args = parser.parse_args()

    results = []
    rng = np.random.default_rng(123)
    for eps_dt in args.eps_list:
        energies = []
        z_final = []
        for seed in range(args.seeds):
            cfg = SimulationConfig(steps=args.steps, eps_dt=eps_dt)
            log = run_simulation(config=cfg, rng=rng)
            energies.append(log.energy[-1])
            z_final.append(log.z_series[-1])
        results.append(
            {
                "eps_dt": eps_dt,
                "E_final": float(np.mean(energies)),
                "z_final": float(np.mean(z_final)),
            }
        )

    fig, ax1 = plt.subplots(figsize=(6, 4))
    eps_vals = [r["eps_dt"] for r in results]
    E_vals = [r["E_final"] for r in results]
    z_vals = [r["z_final"] for r in results]

    ax1.plot(eps_vals, E_vals, marker="o", label="Final energy")
    ax1.set_xlabel("eps_dt")
    ax1.set_ylabel("Final energy")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eps_vals, z_vals, marker="s", color="tab:red", label="Final z")
    ax2.set_ylabel("Final z")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved eps_dt sweep plot to {args.out}")


if __name__ == "__main__":
    main()

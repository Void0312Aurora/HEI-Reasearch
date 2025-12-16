"""Residual diagnostics for DCVI: Herglotz and momentum balance."""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

from .simulation import SimulationConfig, run_simulation


def summarize_residuals(res: np.ndarray) -> Tuple[float, float, float]:
    """Return (max, mean, std) of residual sequence."""
    return float(np.max(res)), float(np.mean(res)), float(np.std(res))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run residual diagnostics for DCVI.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eps-dt", type=float, default=1e-2, dest="eps_dt")
    parser.add_argument("--max-dt", type=float, default=5e-2, dest="max_dt")
    parser.add_argument("--min-dt", type=float, default=1e-4, dest="min_dt")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = SimulationConfig(steps=args.steps, eps_dt=args.eps_dt, max_dt=args.max_dt, min_dt=args.min_dt)
    log = run_simulation(config=cfg, rng=np.random.default_rng(args.seed))

    rc = np.array(log.residual_contact)
    rm = np.array(log.residual_momentum)
    rc_max, rc_mean, rc_std = summarize_residuals(rc)
    rm_max, rm_mean, rm_std = summarize_residuals(rm)

    print("Residual diagnostics (|rk|, ||F||):")
    print(f"Contact Herglotz rk -> max {rc_max:.2e}, mean {rc_mean:.2e}, std {rc_std:.2e}")
    print(f"Momentum balance ||F|| -> max {rm_max:.2e}, mean {rm_mean:.2e}, std {rm_std:.2e}")


if __name__ == "__main__":
    main()

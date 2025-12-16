"""Quick demo run for CCD hyperbolic clustering scaffold."""

from __future__ import annotations

import numpy as np

from .simulation import SimulationConfig, run_simulation


def main() -> None:
    cfg = SimulationConfig(steps=200, eps_dt=1e-2, max_rho=3.0)
    log = run_simulation(config=cfg, rng=np.random.default_rng(42))
    print(f"Recorded {len(log.energy)} steps")
    print(f"Energy: start {log.energy[0]:.4f} -> end {log.energy[-1]:.4f}")
    print(f"xi norm end: {log.xi_norm[-1]:.4f}")
    print("Positions_disk sample:", log.positions_disk[-1][:3])


if __name__ == "__main__":
    main()

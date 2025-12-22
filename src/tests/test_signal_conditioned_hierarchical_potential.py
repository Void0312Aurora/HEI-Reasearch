import numpy as np

from HEI.src.hei.potential import HierarchicalSoftminPotential, build_hierarchical_potential
from HEI.src.hei.simulation import SimulationConfig
from HEI.src.hei.archive.legacy_simulation import run_simulation


def test_build_hierarchical_potential_includes_signal_bias() -> None:
    rng = np.random.default_rng(0)
    pot = build_hierarchical_potential(n_points=10, depth=3, branching=3, max_rho=3.0, rng=rng)

    assert isinstance(pot, HierarchicalSoftminPotential)
    assert pot.point_logit_bias is not None
    assert pot.point_leaf_ids is not None
    assert pot.point_logit_bias.shape[1] == 10
    assert pot.point_leaf_ids.shape == (10,)


def test_signal_conditioned_potential_has_nonzero_gradient() -> None:
    rng = np.random.default_rng(1)
    pot = build_hierarchical_potential(n_points=8, depth=3, branching=3, max_rho=3.0, rng=rng)

    z = rng.uniform(-0.5, 0.5, size=8) + 1j * rng.uniform(1.0, 2.0, size=8)
    grad = pot.gradient(z)

    assert float(np.linalg.norm(grad)) > 1e-8


def test_signal_conditioned_potential_drives_motion() -> None:
    seed = 2
    rng = np.random.default_rng(seed)
    cfg = SimulationConfig(n_points=12, steps=20, max_rho=3.0)
    pot = build_hierarchical_potential(n_points=cfg.n_points, depth=3, branching=3, max_rho=cfg.max_rho, rng=rng)
    log = run_simulation(potential=pot, config=cfg, rng=np.random.default_rng(seed))

    assert max(log.xi_norm) > 1e-6

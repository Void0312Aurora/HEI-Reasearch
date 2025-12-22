
import numpy as np
import matplotlib.pyplot as plt
from hei.simulation import SimulationConfig, run_simulation_group
from hei.potential import build_baseline_potential

def test_energy_beta():
    # Use config similar to user
    cfg = SimulationConfig(
        n_points=50,
        steps=500, # Short run
        max_dt=0.005,
        log_interval=10
    )
    rng = np.random.default_rng(42)
    
    # Create potential (ensure hierarchy to enable beta annealing)
    # build_baseline_potential usually returns GaussianWellsPotential (no hierarchy/anneal?)
    # We need HierarchicalSoftminPotential
    from hei.potential import HierarchicalSoftminPotential, build_hierarchical_potential
    pot = build_hierarchical_potential(n_points=50, depth=2, branching=3, rng=rng)
    pot.anneal_beta = 0.0 # From simulation.py default, but let's check
    
    # simulation.py overrides anneal_beta=0 in run_simulation_group?
    # Yes: "if hasattr(pot, 'anneal_beta'): pot.anneal_beta = 0.0"
    # But Pot.beta is annealed in the loop.
    
    log = run_simulation_group(potential=pot, config=cfg, rng=rng)
    
    # Print correlation
    betas = np.array(log.beta_series)
    energies = np.array(log.energy)
    steps = np.array(log.steps)
    
    # Handle decimated beta logs (log.beta_series might be 0 if not hierarchical?)
    # Check if we have valid beta
    print(f"Beta range: {betas.min()} -> {betas.max()}")
    print(f"Energy range: {energies.min()} -> {energies.max()}")
    
    # Plot or print ratio
    # Avoid div by zero
    valid = betas > 0.01
    if np.any(valid):
        ratio = energies[valid] / betas[valid]
        print(f"Energy/Beta Ratio Start: {ratio[0]:.4f}")
        print(f"Energy/Beta Ratio End: {ratio[-1]:.4f}")
        print(f"Ratio Change: {(ratio[-1]-ratio[0])/ratio[0]*100:.2f}%")
        
    # Also Check Kinetic Energy trend
    kin = np.array(log.kinetic)
    print(f"Kinetic Start: {kin[0]:.4f}, End: {kin[-1]:.4f}")

if __name__ == "__main__":
    test_energy_beta()


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# Ensure we can import from hei
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hei.group_integrator import GroupContactIntegrator, GroupIntegratorState, GroupIntegratorConfig
from hei.inertia import riemannian_inertia
from hei.data_potential import DataDrivenStressPotential
from hei.geometry import uhp_to_hyperboloid
from experiment_tunneling import build_tree_distances, initialize_twisted_trap

def run_sweep_trial(stiffness: float, gamma: float, mass: float = 1.0, steps: int = 5000) -> Tuple[float, List[float]]:
    """Runs a single trial with given parameters."""
    
    # Setup
    dists = build_tree_distances()
    pot = DataDrivenStressPotential(target_dists=dists, stiffness=stiffness)
    z_init = initialize_twisted_trap()
    
    # Config
    cfg = GroupIntegratorConfig(
        max_dt=2e-2,
        min_dt=1e-6,
        gamma_scale=gamma,
        gamma_mode="constant",
        # Unclipped physics
        torque_clip=10000.0,
        xi_clip=1000.0,
        use_riemannian_inertia=True
    )
    
    # Init State
    h_init = uhp_to_hyperboloid(z_init)
    n_points = z_init.shape[0]
    
    state = GroupIntegratorState(
        G=np.eye(2),
        z0_uhp=z_init,
        xi=np.zeros((n_points, 3)),
        m=None,
        action=0.0
    )
    
    force_fn = lambda z, a: -pot.gradient(z, a)
    integrator = GroupContactIntegrator(force_fn, pot.potential, cfg)
    
    energies = []
    
    print(f"Running Trial: K={stiffness}, Gamma={gamma}, M={mass}")
    
    for i in range(steps):
        state = integrator.step(state)
        # We can optimize by not computing full potential every step if needed,
        # but for 5000 steps it's fine.
        if i % 100 == 0:
            e = pot.potential(state.z_uhp)
            energies.append(e)
            
            # Early exit if exploded
            if e > 1e6:
                print("  -> Exploded!")
                return e, energies
                
    final_e = pot.potential(state.z_uhp)
    energies.append(final_e)
    print(f"  -> Final E: {final_e:.4f}")
    
    return final_e, energies

def main():
    stiffnesses = [1.0] # Focus on soft potential where sim is fast
    gammas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]
    masses = [1.0] # Mass 1.0 is stable
    
    results = []
    
    best_e = float('inf')
    best_params = None
    
    plt.figure(figsize=(10, 6))
    
    for k in stiffnesses:
        for g in gammas:
            for m in masses:
                final_e, history = run_sweep_trial(k, g, m)
                label = f"K={k}, G={g}, M={m}"
                plt.plot(np.linspace(0, 5000, len(history)), history, label=label + f" ({final_e:.3f})")
                
                if final_e < best_e:
                    best_e = final_e
                    best_params = (k, g, m)
                    
    print("\nSweep Complete.")
    print(f"Best Parameters: K={best_params[0]}, Gamma={best_params[1]}, Mass={best_params[2]}")
    print(f"Best Energy: {best_e:.4f}")
    
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.title('Parameter Sweep for Tunneling')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('sweep_results.png')

if __name__ == "__main__":
    main()

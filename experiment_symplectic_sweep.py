"""
Symplectic Parameter Sweep Experiment
=====================================

Grid Search for SymplecticLieEuler on Tree Reconstruction Grid:
- dt: [0.005, 0.01, 0.02]
- gamma: [1.0, 5.0, 10.0]
- clip_norm: [1.0, 5.0] (Maybe 0.5 too?)

Goal: Find stable region for Fixed Step optimization.
"""

import sys
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hei.symplectic import SymplecticLieEuler, SymplecticConfig
from hei.group_integrator import GroupIntegratorState
from experiment_tree import SyntheticTree, TreePotential, init_big_bang, calculate_distortion

def run_single_cfg(tree, pot, z0, dt, gamma, clip, steps=5000):
    cfg = SymplecticConfig(
        dt=dt,
        gamma=gamma,
        mass=1.0,
        clip_norm=clip
    )
    
    # Init state
    G_init = np.array([np.eye(2) for _ in range(tree.num_nodes)])
    xi_init = np.zeros((tree.num_nodes, 3))
    state = GroupIntegratorState(G=G_init, z0_uhp=z0.copy(), xi=xi_init, m=None)
    force_fn = lambda z, a: -pot.gradient(z, a)
    
    integrator = SymplecticLieEuler(force_fn, cfg)
    
    dist_hist = []
    
    try:
        for i in range(steps):
            state = integrator.step(state)
            if i % 100 == 0:
                d = calculate_distortion(state.z_uhp, tree)
                dist_hist.append(d)
                # Early exit on explosion
                if d > 100.0:
                    break
    except Exception as e:
        print(f"Run crashed: {e}")
        return 999.9, []
        
    final_dist = dist_hist[-1] if dist_hist else 999.9
    return final_dist, dist_hist

def run_sweep():
    print("Setting up Symplectic Sweep...")
    depth = 3
    tree = SyntheticTree(depth=depth)
    pot = TreePotential(tree=tree, L_target=1.7, k_att=2.0, A_rep=1.0, sigma_rep=0.5)
    
    np.random.seed(42)
    z0 = init_big_bang(tree.num_nodes, sigma=0.01)
    
    # Grid
    dts = [0.005, 0.01, 0.02]
    gammas = [1.0, 2.0, 5.0, 10.0]
    clips = [0.5, 1.0, 5.0]
    
    results = []
    
    total_runs = len(dts) * len(gammas) * len(clips)
    count = 0
    
    print(f"Starting {total_runs} runs...")
    
    best_dist = 999.9
    best_cfg = None
    
    for dt, gamma, clip in itertools.product(dts, gammas, clips):
        count += 1
        print(f"[{count}/{total_runs}] Running dt={dt}, gamma={gamma}, clip={clip}...", end="", flush=True)
        
        final_d, hist = run_single_cfg(tree, pot, z0, dt, gamma, clip)
        
        print(f" Dist={final_d:.4f}")
        
        results.append({
            'dt': dt,
            'gamma': gamma,
            'clip': clip,
            'dist': final_d
        })
        
        if final_d < best_dist:
            best_dist = final_d
            best_cfg = (dt, gamma, clip)
            
    print("\n=== Sweep Complete ===")
    print(f"Best Distortion: {best_dist:.4f}")
    print(f"Best Config: dt={best_cfg[0]}, gamma={best_cfg[1]}, clip={best_cfg[2]}")
    
    # Save results to CSV (or just print table)
    df = pd.DataFrame(results)
    df.to_csv('symplectic_sweep_results.csv', index=False)
    print("Saved to symplectic_sweep_results.csv")
    
    # Pivot table for visualization
    # Average over clips?
    print("\nPivot (Gamma vs Dt, averaged over Clip):")
    piv = df.pivot_table(index='gamma', columns='dt', values='dist', aggfunc='min')
    print(piv)

if __name__ == "__main__":
    run_sweep()

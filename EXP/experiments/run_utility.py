import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_path)

from EXP.experiments.run_entity import run_experiment_entity

def run_utility_comparison():
    print("Running E3-Utility Comparison: Plastic Entity vs Contact Baseline...")
    
    # Common Config
    base_config = {
        'dim_q': 2,
        'dt': 0.1,
        'omega': 1.0,
        'eta': 0.05,
        'env_type': 'limit_cycle',
        'drive_amp': 1.0, 
        'env_gamma': 0.1,
        'online_steps': 500,
        'offline_steps': 0,
        'max_steps': 500,
        'output_dir': 'EXP/utility_test',
        'force_u_self_zero': False
    }
    
    # 1. Plastic Entity
    cfg_plastic = base_config.copy()
    cfg_plastic['kernel_type'] = 'plastic'
    cfg_plastic['seed'] = 42
    print("Running Plastic Entity...")
    d1_p, _, d3_p, a1_p, log_p = run_experiment_entity(cfg_plastic)
    
    # Extract Weights
    dim_q = cfg_plastic['dim_q']
    x_int_all = np.array(log_p['x_int']).squeeze(1) # [T, d_int]
    
    # Weights are at the end
    w_start = 4 * dim_q + 1
    weights = x_int_all[:, w_start:] # [T, d*d]
    w_norm = np.linalg.norm(weights, axis=1)
    
    final_norm = w_norm[-1]
    
    print("\nResults:")
    print(f"Final Weight Norm: {final_norm:.4f}")
    
    # Plot Trajectories
    out_dir = os.path.join(base_path, base_config['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(w_norm, label='Weight Norm (Plasticity)')
    plt.title(f"E3 Utility: Plasticity Convergence (Norm={final_norm:.2f})")
    plt.xlabel("Step")
    plt.ylabel("|W|")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "utility_learning.png"))
    print(f"Plot saved to {out_dir}/utility_learning.png")
    
    if final_norm > 0.1:
        print("SUCCESS: Entity demonstrates Learning (Weights Acquired).")
    else:
        print("WARNING: Entity failed to learn (Weights low).")

if __name__ == "__main__":
    run_utility_comparison()

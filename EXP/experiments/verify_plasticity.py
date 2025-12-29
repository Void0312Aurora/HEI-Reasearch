import sys
import os
import yaml
import torch
import numpy as np
import argparse
import pandas as pd
from scipy import stats

# Add path to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from experiments.run import run_experiment

def run_plasticity_verification(config_path, num_seeds=10):
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    conditions = [
        {"name": "Replay (Ordered)", "mode": "replay", "shuffle": False, "reverse": False, "block": False},
        {"name": "Mismatch (Reverse)", "mode": "replay", "shuffle": False, "reverse": True, "block": False},
        {"name": "Mismatch (BlockShuffle)", "mode": "replay", "shuffle": False, "reverse": False, "block": True}
    ]
    
    # Storage
    raw_data = {c["name"]: {"weight_norm": []} for c in conditions}
    
    # Use Plastic Kernel + Limit Cycle Env (Pure Decay Mode)
    base_config['kernel_type'] = 'plastic'
    base_config['env_type'] = 'limit_cycle' 
    
    # E2 Critical Setup: Pure Decay (Transient)
    # Forward: High -> Low (Damping)
    # Reverse: Low -> High (Anti-Damping)
    # This asymmetry is guaranteed if noise is low.
    base_config['drive_amp'] = 0.0 # No drive
    base_config['env_noise'] = 0.0 # No noise
    base_config['env_gamma'] = 0.1 # Slow decay
    
    base_config['omega'] = 1.0 
    base_config['eta'] = 0.05 
    dim_q = base_config['dim_q']
    
    base_seed = base_config.get('seed', 42)
    
    print(f"Running Plasticity Verification (N={num_seeds}) with Kernel: {base_config['kernel_type']}")
    
    for i in range(num_seeds):
        curr_seed = base_seed + i
        print(f"  > Seed {i+1}/{num_seeds} (seed={curr_seed})...")
        
        for cond in conditions:
            cfg = base_config.copy()
            cfg['seed'] = curr_seed
            cfg['offline_u_source'] = cond['mode']
            if cond['shuffle']: cfg['replay_shuffle'] = True
            if cond['reverse']: cfg['replay_reverse'] = True
            if cond['block']: cfg['replay_block_shuffle'] = True
            cfg['force_u_self_zero'] = False
            
            d1, d2, d3, log = run_experiment(cfg)
            
            # Extract Weights W from end of trajectory
            # State: q, p, s, r, vr, w
            # Start of W: 4*dim + 1
            w_start = 4*dim_q + 1
            w_dim = dim_q * dim_q
            
            x_int = np.array(log['x_int'])
            if x_int.ndim == 3: x_int = x_int.squeeze(1)
            
            # Get final state weight
            # Or mean weight norm over offline phase?
            # Final weight represents total learning.
            w_final = x_int[-1, w_start:w_start+w_dim]
            w_norm = np.linalg.norm(w_final)
            
            raw_data[cond["name"]]["weight_norm"].append(w_norm)

    # Stats
    report_rows = []
    baseline_name = "Replay (Ordered)"
    base_vals = raw_data[baseline_name]["weight_norm"]
    
    for cond_name in raw_data:
        vals = raw_data[cond_name]["weight_norm"]
        
        mean, std = np.mean(vals), np.std(vals)
        ci = 1.96 * std / np.sqrt(num_seeds)
        
        p_val_valid = False
        p_val = 1.0
        if cond_name != baseline_name:
            t_stat, p_val = stats.ttest_ind(base_vals, vals, equal_var=False)
            p_val_valid = True
            
        row = {
            "Condition": cond_name,
            "WeightNorm (Mean)": f"{mean:.4f}",
            "WeightNorm (Std)": f"{std:.4f}",
            "CI95": f"±{ci:.4f}",
            "p-value": f"{p_val:.5f}" if p_val_valid else "-"
        }
        report_rows.append(row)
        
    df = pd.DataFrame(report_rows)
    print("\n=== Plasticity Verification Results (Iteration 1.4 - Pure Decay) ===\n")
    print(df.to_string(index=False))
    
    out_file = os.path.join(base_config['output_dir'], "verification_iter_1_4.md")
    with open(out_file, 'w') as f:
        f.write(f"# Iteration 1.4 Structure Verification (Pure Decay, N={num_seeds})\n")
        f.write("## Hypothesis: Decay (Forward) vs Growth (Reverse) -> Significant Weight Difference\n\n```\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n")
    print(f"\nSaved report to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    
    run_plasticity_verification(args.config, args.seeds)

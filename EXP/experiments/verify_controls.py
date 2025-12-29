import sys
import os
import yaml
import torch
import numpy as np
import argparse
import pandas as pd

# Add path to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from experiments.run import run_experiment
from diag.compute.metrics import compute_d3_port_loop

def run_verification(config_path):
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    results = []
    
    # 1. Baseline Run
    print("Running Baseline...")
    config_base = base_config.copy()
    config_base['force_u_self_zero'] = False
    d1_base, d2_base, d3_base, log_base = run_experiment(config_base)
    
    res_base = {"Condition": "Baseline"}
    res_base.update(d1_base)
    res_base.update(d2_base)
    res_base.update(d3_base)
    results.append(res_base)
    
    # 2. Ablation Run (u_self = 0)
    print("Running Ablation (u_self=0)...")
    config_abl = base_config.copy()
    config_abl['force_u_self_zero'] = True
    d1_abl, d2_abl, d3_abl, log_abl = run_experiment(config_abl)
    
    res_abl = {"Condition": "Ablation (Zero U)"}
    # D1 might fail/change
    res_abl.update(d1_abl)
    res_abl.update(d2_abl)
    # D3 should be low/zero
    res_abl.update(d3_abl)
    results.append(res_abl)
    
    # 3. Mismatch Check (Shuffle u_self from Baseline)
    print("Checking Mismatch (Shuffle U)...")
    # Recover traj from Baseline Log
    # log_base['x_int'] -> T x 1 x Dim
    traj_x_int = torch.tensor(np.array(log_base["x_int"])).squeeze(1)
    dim_q = base_config['dim_q']
    
    # Extract u_self from x_blanket
    traj_x_blanket = torch.tensor(np.array(log_base["x_blanket"])).squeeze(1)
    # x_blanket = [u_env, u_self]
    # Assuming u_env is dim_q, u_self is dim_q. 
    # Check shape logic in run.py: x_blanket = torch.cat([u_env, u_self], dim=1)
    u_self_traj = traj_x_blanket[:, dim_q:]
    
    # Shuffle u_self in time
    permcheck = torch.randperm(u_self_traj.shape[0])
    u_self_shuffled = u_self_traj[permcheck]
    
    # Recompute D3
    d3_mismatch = compute_d3_port_loop(traj_x_int[:, :dim_q].unsqueeze(1), u_self_shuffled.unsqueeze(1))
    
    res_mis = {"Condition": "Mismatch (Shuffle U)"}
    # D1 is same as Baseline (dynamics didn't change, just metric input)
    # Copy D1 from Base
    for k, v in d1_base.items():
        res_mis[k] = v
    # Update D3
    res_mis.update(d3_mismatch)
    results.append(res_mis)
    
    # 4. Fast-Slow Epsilon Scan (E2 Evidence)
    print("Running Fast-Slow Epsilon Scan...")
    epsilons = [1.0, 0.1, 0.01]
    for eps in epsilons:
        print(f"  Testing Epsilon = {eps}...")
        config_fs = base_config.copy()
        config_fs['kernel_type'] = 'fast_slow'
        config_fs['epsilon'] = eps
        # Use baseline control settings
        config_fs['force_u_self_zero'] = False 
        
        # d1, d2, d3, log = run_experiment(config_fs)
        # We need to catch if kernel is unknown? run_experiment handles it.
        d1_fs, d2_fs, d3_fs, _ = run_experiment(config_fs)
        
        res_fs = {"Condition": f"FastSlow (e={eps})"}
        res_fs.update(d1_fs)
        res_fs.update(d2_fs)
        res_fs.update(d3_fs)
        results.append(res_fs)
    df = pd.DataFrame(results)
    
    # Reorder cols
    cols = ["Condition"] + [c for c in df.columns if c.startswith("D2") or c.startswith("d2")] + \
           [c for c in df.columns if c.startswith("d3")] + \
           [c for c in df.columns if c.startswith("d1") and c not in ["Condition"]]
    df = df[cols]
    
    print("\n=== Verification Results (Iteration 0.1) ===\n")
    print(df.to_string(index=False))
    
    # Save to file
    out_file = os.path.join(base_config['output_dir'], "verification_iter_0_1.md")
    with open(out_file, 'w') as f:
        f.write("# Iteration 0.1 Verification Results\n\n```\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n")
    print(f"\nSaved simplified report to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    run_verification(args.config)

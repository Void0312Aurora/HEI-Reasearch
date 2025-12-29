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

def run_verification(config_path):
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    results = []
    
    # 1. Internal Baseline (Pure Self-Loop)
    print("Running Internal Baseline (u_source=internal)...")
    config_int = base_config.copy()
    config_int['offline_u_source'] = 'internal'
    config_int['force_u_self_zero'] = False
    d1_int, d2_int, d3_int, _ = run_experiment(config_int)
    
    res_int = {"Condition": "Internal (Baseline)"}
    res_int.update(d1_int)
    res_int.update(d2_int)
    res_int.update(d3_int)
    results.append(res_int)
    
    # 2. Replay (Experience Driving)
    print("Running Experience Replay (u_source=replay)...")
    config_rep = base_config.copy()
    config_rep['offline_u_source'] = 'replay'
    config_rep['force_u_self_zero'] = False
    d1_rep, d2_rep, d3_rep, _ = run_experiment(config_rep)
    
    res_rep = {"Condition": "Replay (Experience)"}
    res_rep.update(d1_rep)
    res_rep.update(d2_rep)
    res_rep.update(d3_rep)
    results.append(res_rep)
    
    # 3. Mismatch Replay (Shuffled Experience)
    print("Running Mismatch Replay (u_source=replay, shuffled)...")
    config_mis = base_config.copy()
    config_mis['offline_u_source'] = 'replay'
    config_mis['replay_shuffle'] = True
    config_mis['force_u_self_zero'] = False
    d1_mis, d2_mis, d3_mis, _ = run_experiment(config_mis)
    
    res_mis = {"Condition": "Mismatch (Shuffled Exp)"}
    res_mis.update(d1_mis)
    res_mis.update(d2_mis)
    res_mis.update(d3_mis)
    results.append(res_mis)
    
    # Output Table
    df = pd.DataFrame(results)
    
    # Select critical columns
    # D1: Variance, Max Excursion
    # D2: Max Eig, SVD Gap, Gap23, Top1/2/3
    # D3: Correlation
    
    cols = ["Condition", 
            "d1_variance", "d1_max_excursion",
            "d2_max_eig", "d2_svd_gap", "d2_gap23", 
            "d2_top1", "d2_top2", "d2_top3",
            "d3_correlation"]
            
    # Filter valid cols
    valid_cols = [c for c in cols if c in df.columns]
    df = df[valid_cols]
    
    print("\n=== Experience Verification Results (Iteration 1.0) ===\n")
    print(df.to_string(index=False))
    
    # Save to file
    out_file = os.path.join(base_config['output_dir'], "verification_iter_1_0.md")
    with open(out_file, 'w') as f:
        f.write("# Iteration 1.0 Experience Verification\n\n```\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n")
    print(f"\nSaved simplified report to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    run_verification(args.config)

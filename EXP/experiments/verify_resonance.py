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
    
    # Use Resonant Kernel for this test
    # We want to see if Resonator energy distinguishes Replay from Mismatch
    base_config['kernel_type'] = 'resonant'
    base_config['omega'] = 1.0 # Tune to match expected dynamics? Or generic.
    
    print(f"Using Kernel: {base_config['kernel_type']} (omega={base_config.get('omega')})")
    
    # 1. Replay (Experience)
    print("Running Replay (Ordered)...")
    config_rep = base_config.copy()
    config_rep['offline_u_source'] = 'replay'
    config_rep['force_u_self_zero'] = False
    d1_rep, d2_rep, d3_rep, log_rep = run_experiment(config_rep)
    
    # Calculate Resonator Energy (r^2 + v_r^2)
    # x_int: [q, p, s, r, v_r]
    # dim_q = 2. indices: q:0-2, p:2-4, s:4-5, r:5-7, vr:7-9
    x_int_rep = np.array(log_rep['x_int'])
    if x_int_rep.ndim == 3:
        x_int_rep = x_int_rep.squeeze(1)
        
    dim_q = base_config['dim_q']
    # r is at [5, 6], vr at [7, 8] if dim=2
    # General: q(d), p(d), s(1), r(d), vr(d).
    # Start of r: 2*d + 1
    start_r = 2*dim_q + 1
    r_rep = x_int_rep[:, start_r:start_r+dim_q]
    vr_rep = x_int_rep[:, start_r+dim_q:start_r+2*dim_q]
    energy_res_rep = np.mean(r_rep**2 + vr_rep**2)
    
    res_rep = {"Condition": "Replay (Ordered)"}
    res_rep.update(d1_rep)
    res_rep['resonator_energy'] = energy_res_rep
    results.append(res_rep)
    
    # 2. Time-Reverse Mismatch
    # We need to hack run.py or buffer to do this.
    # run.py doesn't natively support "reverse".
    # We can pre-record online buffer, reverse it, and feed it?
    # Or just add a 'replay_mode' to config and run.py?
    # Let's add 'replay_mode' = 'reverse' support to run.py OR
    # Temporarily, we can just edit the u_env_online_buffer in memory if we could hook in... but we can't easily.
    # EASIEST: Expect run.py to support 'replay_reverse' flag.
    # I will modify run.py to support 'replay_reverse' and 'replay_block_shuffle'.
    
    print("Running Time-Reverse...")
    config_rev = base_config.copy()
    config_rev['offline_u_source'] = 'replay'
    config_rev['replay_reverse'] = True
    config_rev['force_u_self_zero'] = False
    d1_rev, d2_rev, d3_rev, log_rev = run_experiment(config_rev)
    
    x_int_rev = np.array(log_rev['x_int'])
    if x_int_rev.ndim == 3:
        x_int_rev = x_int_rev.squeeze(1)
        
    r_rev = x_int_rev[:, start_r:start_r+dim_q]
    vr_rev = x_int_rev[:, start_r+dim_q:start_r+2*dim_q]
    energy_res_rev = np.mean(r_rev**2 + vr_rev**2)
    
    res_rev = {"Condition": "Mismatch (Reverse)"}
    res_rev.update(d1_rev)
    res_rev['resonator_energy'] = energy_res_rev
    results.append(res_rev)
    
    # 3. Block-Shuffle Mismatch
    print("Running Block-Shuffle...")
    config_blk = base_config.copy()
    config_blk['offline_u_source'] = 'replay'
    config_blk['replay_block_shuffle'] = True
    config_blk['force_u_self_zero'] = False
    d1_blk, d2_blk, d3_blk, log_blk = run_experiment(config_blk)
    
    x_int_blk = np.array(log_blk['x_int'])
    if x_int_blk.ndim == 3:
        x_int_blk = x_int_blk.squeeze(1)
        
    r_blk = x_int_blk[:, start_r:start_r+dim_q]
    vr_blk = x_int_blk[:, start_r+dim_q:start_r+2*dim_q]
    energy_res_blk = np.mean(r_blk**2 + vr_blk**2)
    
    res_blk = {"Condition": "Mismatch (BlockShuffle)"}
    res_blk.update(d1_blk)
    res_blk['resonator_energy'] = energy_res_blk
    results.append(res_blk)
    
    # Output Table
    df = pd.DataFrame(results)
    
    cols = ["Condition", "d1_variance", "resonator_energy", "d1_max_excursion"]
    valid_cols = [c for c in cols if c in df.columns]
    df = df[valid_cols]
    
    print("\n=== Temporal Resonance Verification (Iteration 1.1) ===\n")
    print(df.to_string(index=False))
    
    out_file = os.path.join(base_config['output_dir'], "verification_iter_1_1.md")
    with open(out_file, 'w') as f:
        f.write("# Iteration 1.1 Temporal Resonance\n\n```\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n")
    print(f"\nSaved report to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    run_verification(args.config)

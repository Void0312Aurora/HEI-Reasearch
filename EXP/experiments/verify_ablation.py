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

def run_ablation_verification(config_path, num_seeds=10):
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    # Standard Setup for E2 LimitCycle
    base_config['env_type'] = 'limit_cycle'
    base_config['drive_amp'] = 0.0
    base_config['env_noise'] = 0.0
    base_config['env_gamma'] = 0.1
    base_config['omega'] = 1.0
    base_config['eta'] = 0.05
    dim_q = base_config['dim_q']

    # Conditions to test
    # We want to test Discrimination: Replay vs BlockShuffle
    # For two kernels: Plastic, Resonant
    
    kernels = ['plastic', 'resonant']
    modes = ['Replay', 'BlockShuffle']
    
    results = []
    
    print(f"Running Ablation Verification (N={num_seeds}) on LimitCycle (Decay)...")
    
    for k_type in kernels:
        print(f"\n--- Testing Kernel: {k_type} ---")
        
        # Storage for stats
        metrics = {m: [] for m in modes}
        energy_checks = {m: {"mean_u2": [], "var_u": []} for m in modes}
        
        for i in range(num_seeds):
            curr_seed = base_config['seed'] + i
            
            for mode in modes:
                cfg = base_config.copy()
                cfg['seed'] = curr_seed
                cfg['kernel_type'] = k_type
                cfg['offline_u_source'] = 'replay'
                cfg['force_u_self_zero'] = False
                
                if mode == 'BlockShuffle':
                    cfg['replay_block_shuffle'] = True
                    
                d1, d2, d3, log = run_experiment(cfg)
                
                # 1. Metric Extraction
                x_int = np.array(log['x_int'])
                if x_int.ndim == 3: x_int = x_int.squeeze(1)
                
                # Filter Offline
                phase = np.array([m['phase'] for m in log['step_meta']])
                mask = phase == 'offline'
                if not np.any(mask): continue
                
                x_off = x_int[mask]
                
                if k_type == 'plastic':
                    # Metric: Weight Norm
                    w_start = 4*dim_q + 1
                    w_dim = dim_q * dim_q
                    # Take final weight
                    w_final = x_off[-1, w_start:w_start+w_dim]
                    val = np.linalg.norm(w_final)
                else: 
                    # Metric: Resonator Energy (Mean over trajectory)
                    start_r = 2*dim_q + 1
                    r = x_off[:, start_r:start_r+dim_q]
                    vr = x_off[:, start_r+dim_q:start_r+2*dim_q]
                    val = np.mean(r**2 + vr**2)
                    
                metrics[mode].append(val)
                
                # 2. Distribution Check (sanity)
                u_t = np.array(log['u_t']) # This includes u_self + u_env?
                # run.py: log u_t is u_combined.
                # In Replay mode, u_combined = u_env_replay - 0.05*p (damping).
                # We want to check u_env consistency.
                # run.py doesn't log isolated u_env easily in the final array unless we parse x_blanket?
                # x_blanket = [u_env, u_self]
                x_blanket = np.array(log['x_blanket'])
                if x_blanket.ndim == 3: x_blanket = x_blanket.squeeze(1)
                u_env_log = x_blanket[mask, :dim_q]
                
                # Compute stats on u_env
                mean_sq = np.mean(u_env_log**2)
                var = np.var(u_env_log)
                energy_checks[mode]["mean_u2"].append(mean_sq)
                energy_checks[mode]["var_u"].append(var)

        # Statistical Analysis per Kernel
        replay_vals = metrics['Replay']
        shuffle_vals = metrics['BlockShuffle']
        
        t_stat, p_val = stats.ttest_ind(replay_vals, shuffle_vals, equal_var=False)
        
        # Energy Check Stats
        u2_replay = np.mean(energy_checks['Replay']['mean_u2'])
        u2_shuffle = np.mean(energy_checks['BlockShuffle']['mean_u2'])
        
        res = {
            "Kernel": k_type,
            "Replay (Mean)": np.mean(replay_vals),
            "Shuffle (Mean)": np.mean(shuffle_vals),
            "p-value": p_val,
            "DistCheck (Replay u^2)": u2_replay,
            "DistCheck (Shuffle u^2)": u2_shuffle,
            "Shuffle Energy Ratio": u2_shuffle / u2_replay if u2_replay > 0 else 0
        }
        results.append(res)
        
    df = pd.DataFrame(results)
    
    print("\n=== E2 Strengthening & Ablation Results (Iteration 1.5) ===\n")
    print(df.to_string(index=False))
    
    out_file = os.path.join(base_config['output_dir'], "verification_iter_1_5.md")
    with open(out_file, 'w') as f:
        f.write("# Iteration 1.5 Ablation & Consistency Audit\n")
        f.write(f"Env: LimitCycle (Pure Decay), Seeds: {num_seeds}\n\n")
        f.write("## 1. Discrimination Power (Replay vs Shuffle)\n")
        f.write("- Plastic Metric: Weight Norm\n")
        f.write("- Resonant Metric: Resonator Energy\n\n```\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n")
        f.write("\n## 2. Distribution Consistency Check\n")
        f.write("If 'Shuffle Energy Ratio' is close to 1.0, the shuffle is chemically pure (energy preserving).\n")
        
    print(f"\nSaved ablation report to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    
    run_ablation_verification(args.config, args.seeds)

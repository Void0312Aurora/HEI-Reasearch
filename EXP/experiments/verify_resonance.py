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

def run_multiseed_verification(config_path, num_seeds=10):
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    conditions = [
        {"name": "Replay (Ordered)", "mode": "replay", "shuffle": False, "reverse": False, "block": False},
        {"name": "Mismatch (Reverse)", "mode": "replay", "shuffle": False, "reverse": True, "block": False},
        {"name": "Mismatch (BlockShuffle)", "mode": "replay", "shuffle": False, "reverse": False, "block": True}
    ]
    
    # Storage for stats
    # condition -> list of {d1_var, res_energy}
    raw_data = {c["name"]: {"d1_variance": [], "resonator_energy": []} for c in conditions}
    
    # Use Resonant Kernel
    base_config['kernel_type'] = 'resonant'
    base_config['omega'] = 1.0 
    dim_q = base_config['dim_q']
    
    # Loop seeds
    base_seed = base_config.get('seed', 42)
    
    print(f"Running Robust Verification (N={num_seeds}) with Kernel: {base_config['kernel_type']}")
    
    for i in range(num_seeds):
        curr_seed = base_seed + i
        print(f"  > Seed {i+1}/{num_seeds} (seed={curr_seed})...")
        
        for cond in conditions:
            # Prepare config
            cfg = base_config.copy()
            cfg['seed'] = curr_seed
            cfg['offline_u_source'] = cond['mode']
            if cond['shuffle']: cfg['replay_shuffle'] = True
            if cond['reverse']: cfg['replay_reverse'] = True
            if cond['block']: cfg['replay_block_shuffle'] = True
            cfg['force_u_self_zero'] = False
            
            # Run
            d1, d2, d3, log = run_experiment(cfg)
            
            # Extract Resonator Energy (Offline Only!)
            x_int = np.array(log['x_int'])
            if x_int.ndim == 3: x_int = x_int.squeeze(1)
            
            # Mask offline phase
            step_metas = log["step_meta"]
            phases = [m["phase"] for m in step_metas]
            offline_mask = np.array(phases) == "offline"
            
            # x_int indices: q(d), p(d), s(1), r(d), vr(d)
            start_r = 2*dim_q + 1
            
            if np.any(offline_mask):
                x_off = x_int[offline_mask]
                r = x_off[:, start_r:start_r+dim_q]
                vr = x_off[:, start_r+dim_q:start_r+2*dim_q]
                energy = np.mean(r**2 + vr**2)
            else:
                energy = 0.0
                
            raw_data[cond["name"]]["d1_variance"].append(d1.get("d1_variance", np.nan))
            raw_data[cond["name"]]["resonator_energy"].append(energy)

    # Statistical Analysis
    report_rows = []
    
    baseline_name = "Replay (Ordered)"
    base_energies = raw_data[baseline_name]["resonator_energy"]
    base_vars = raw_data[baseline_name]["d1_variance"]
    
    for cond_name in raw_data:
        energies = raw_data[cond_name]["resonator_energy"]
        vars = raw_data[cond_name]["d1_variance"]
        
        # Stats
        e_mean, e_std = np.mean(energies), np.std(energies)
        v_mean, v_std = np.mean(vars), np.std(vars)
        e_ci = 1.96 * e_std / np.sqrt(num_seeds)
        
        # Significance (Welch's t-test) vs Baseline
        p_val_valid = False
        p_val = 1.0
        
        if cond_name != baseline_name:
            t_stat, p_val = stats.ttest_ind(base_energies, energies, equal_var=False)
            p_val_valid = True
            
        row = {
            "Condition": cond_name,
            "ResEnergy (Mean)": f"{e_mean:.3f}",
            "ResEnergy (Std)": f"{e_std:.3f}",
            "ResEnergy (CI95)": f"±{e_ci:.3f}",
            "D1 Var (Mean)": f"{v_mean:.3f}",
            "p-value (vs Replay)": f"{p_val:.5f}" if p_val_valid else "-"
        }
        report_rows.append(row)
        
    df = pd.DataFrame(report_rows)
    
    print("\n=== Robustness Verification Results (Iteration 1.2) ===\n")
    print(df.to_string(index=False))
    
    out_file = os.path.join(base_config['output_dir'], "verification_iter_1_2.md")
    with open(out_file, 'w') as f:
        f.write(f"# Iteration 1.2 Robustness Verification (N={num_seeds})\n")
        f.write("## Hypothesis: Ordered Replay should have significantly higher Resonator Energy than Mismatched.\n\n")
        f.write("```\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n")
        
        f.write("\n## Raw Data Summary\n")
        for k, v in raw_data.items():
            f.write(f"### {k}\n")
            f.write(f"- Energies: {v['resonator_energy']}\n")
            f.write(f"- Variances: {v['d1_variance']}\n")

    print(f"\nSaved statistical report to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    
    run_multiseed_verification(args.config, args.seeds)

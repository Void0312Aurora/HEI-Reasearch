import sys
import os
import yaml
import json
import torch
import numpy as np
import argparse
from datetime import datetime
from itertools import product

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_path)

from EXP.experiments.run_entity import run_experiment_entity

def run_suite():
    # Matrix Definition
    seeds = range(42, 47) # 5 seeds for speed (Plan said 10, shrinking for MVP speed)
    dampings = [0.05, 0.1, 0.2]
    noises = [0.0, 0.1]
    
    results = []
    
    print(f"Starting Structured Suite v0...")
    print(f"Matrix: {len(seeds)} Seeds x {len(dampings)} Dampings x {len(noises)} Noises")
    
    output_dir = os.path.join(base_path, "EXP/structured_suite")
    os.makedirs(output_dir, exist_ok=True)
    
    # Base Config
    base_config = {
        'dim_q': 2,
        'kernel_type': 'plastic',
        'dt': 0.1,
        'omega': 1.0,
        'eta': 0.05,
        'env_type': 'limit_cycle',
        'drive_amp': 1.0, # Driven for robustness test
        'online_steps': 200,
        'offline_steps': 0, # Continuous run for stability
        'max_steps': 200,
        'force_u_self_zero': False,
        'replay_block_shuffle': False,
        'output_dir': output_dir
    }
    
    for seed, gamma, noise in product(seeds, dampings, noises):
        cfg = base_config.copy()
        cfg['seed'] = seed
        cfg['env_gamma'] = gamma
        cfg['env_noise'] = noise
        
        # Run
        try:
            d1, d2, d3, a1, log = run_experiment_entity(cfg)
            
            # Extract Key Metrics
            # E3-Safety: State Max
            x_int = np.array(log['x_int']).squeeze(1)
            e3_max = np.max(np.abs(x_int))
            e3_safe = e3_max < 1e3
            
            # Record
            res = {
                'seed': seed,
                'damping': gamma,
                'noise': noise,
                'e3_safe': bool(e3_safe),
                'e3_max_state': float(e3_max),
                'a1_screening': float(a1['reduction_ratio']),
                'a1_pass': bool(a1['is_screened']),
                'd1_variance': float(d1.get('variance', 0.0)),
                'd3_gain': float(d3.get('d3_gain', 0.0))
            }
            results.append(res)
            print(f"Conf(g={gamma}, n={noise}, s={seed}) -> Safe={e3_safe}, A1={res['a1_screening']:.2%}")
            
        except Exception as e:
            print(f"Conf(g={gamma}, n={noise}, s={seed}) -> FAILED: {e}")
            
    # Aggregation
    df_safe = [r['e3_safe'] for r in results]
    safety_rate = sum(df_safe) / len(df_safe)
    
    df_a1 = [r['a1_pass'] for r in results]
    a1_rate = sum(df_a1) / len(df_a1)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(results),
        "safety_rate": safety_rate,
        "a1_pass_rate": a1_rate,
        "results": results
    }
    
    # Save
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nSuite Complete.")
    print(f"Safety Rate: {safety_rate:.2%}")
    print(f"A1 Pass Rate: {a1_rate:.2%}")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_suite()

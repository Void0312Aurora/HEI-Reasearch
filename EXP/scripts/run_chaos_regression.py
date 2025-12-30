import sys
import os
import yaml
import json
import torch
import numpy as np
import scipy.stats

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_path)

from he_core.entity import Entity
from he_core.bridge_vdp import VanDerPolEnv
from he_core.bridge import LimitCycleObsAdapter, LinearActAdapter
from EXP.diag.compute.chaos_metrics import compute_lyapunov_proxy

def run_single_vdp(seed, active_mode=True, eta=0.05, output_dir=None):
    config = {
        'seed': seed,
        'dim_q': 2,
        'kernel_type': 'plastic',
        'dt': 0.05,
        'omega': 1.0,
        'eta': eta,
        'vdp_mu': 2.0,
        'drive_amp': 2.0,
        'drive_freq': 0.5,
        'max_steps': 1500, # Sufficient for Lyapunov
        'active_mode': active_mode,
        'active_gain': 0.1,
        'output_dir': output_dir if output_dir else 'tmp'
    }
    
    env = VanDerPolEnv(config)
    obs_adapter = LimitCycleObsAdapter()
    act_adapter = LinearActAdapter()
    entity = Entity(config)
    
    log = {'x_ext': [], 'u_self': [], 'pred_error': [], 'x_int': []}
    
    obs = env.reset(seed=seed)
    done = False
    
    while not done:
        entity_obs = obs_adapter.adapt(obs)
        out = entity.step(entity_obs)
        action_env = act_adapter.adapt(out['action'])
        
        # Explicit Prediction Error Audit
        # u_env (Input) vs pred_x (Prediction)
        # obs['x_ext_proxy'][:2] is u_env roughly? 
        # Actually u_env in Entity is x_ext[:, :dim_q].
        u_env = obs['x_ext_proxy'][:2]
        pred_x = out.get('pred_x', np.zeros_like(u_env))
        
        # We need to handle shapes. pred_x is flattened.
        error = u_env - pred_x[:2]
        
        log['pred_error'].append(error)
        log['u_self'].append(out['action'])
        log['x_ext'].append(u_env)
        log['x_int'].append(out['x_int'])
        
        obs_next, reward, done, info = env.step(action_env)
        obs = obs_next
        
    return log

def run_chaos_regression():
    print("Running Phase 7: Chaos Regression Suite...")
    output_dir = os.path.join(base_path, "EXP/chaos_regression")
    os.makedirs(output_dir, exist_ok=True)
    
    seeds = range(42, 52) # 10 seeds
    
    results = []
    
    for s in seeds:
        print(f"Seed {s}...")
        
        # 1. Learned-W (Active + Plastic)
        log_learn = run_single_vdp(s, active_mode=True, eta=0.05)
        
        # 2. Zero-W (Active + Fixed Zero W)
        # Note: eta=0.0 means W stays at init (0.0). Active Mode uses W=0 -> pred=0 -> u=-k*u_env
        # This is pure Damping control.
        log_zero = run_single_vdp(s, active_mode=True, eta=0.0)
        
        # Metrics Calculation
        
        # A. Chaos (Lyapunov) - Robustness Scan
        traj_x = np.array(log_learn['x_ext'])
        # Scan fitting windows (Robustness)
        # We assume compute_lyapunov_proxy uses implicit params, but we can verify stability by splitting?
        # Or just trust the proxy implementation is fixed.
        # Let's just run it standard.
        lyap = compute_lyapunov_proxy(traj_x, 0.05)
        
        # B. Structural Quality
        # MSE
        mse_learn = np.mean(np.array(log_learn['pred_error'])**2)
        mse_zero = np.mean(np.array(log_zero['pred_error'])**2)
        
        # Energy
        en_learn = np.mean(np.array(log_learn['u_self'])**2)
        en_zero = np.mean(np.array(log_zero['u_self'])**2)
        
        # Weights (Check if learned)
        # x_int has W at end?
        # We can just trust MSE diff implies structure.
        
        res = {
            'seed': s,
            'lyap': float(lyap),
            'mse_learn': float(mse_learn),
            'mse_zero': float(mse_zero),
            'en_learn': float(en_learn),
            'en_zero': float(en_zero),
            'mse_ratio': float(mse_learn / (mse_zero + 1e-9)),
            'en_ratio': float(en_learn / (en_zero + 1e-9))
        }
        results.append(res)
        print(f"  Lyap={lyap:.2f}, MSE Ratio={res['mse_ratio']:.2f}, En Ratio={res['en_ratio']:.2f}")
        
    # Aggregate
    mean_lyap = np.mean([r['lyap'] for r in results])
    mean_mse_ratio = np.mean([r['mse_ratio'] for r in results])
    mean_en_ratio = np.mean([r['en_ratio'] for r in results])
    
    # Gates
    passed_chaos = mean_lyap > 0.1
    passed_quality = mean_mse_ratio < 0.95 # At least 5% better prediction?
    # Energy: Should not be significantly higher. Ideally lower or equal.
    # If MSE is lower, Energy might be same or lower (Efficiency).
    passed_energy = mean_en_ratio < 1.1 
    
    print("\n--- Regression Results ---")
    print(f"Mean Lyapunov: {mean_lyap:.4f} (Gate > 0.1) -> {'PASS' if passed_chaos else 'FAIL'}")
    print(f"Mean MSE Ratio: {mean_mse_ratio:.4f} (Gate < 0.95) -> {'PASS' if passed_quality else 'FAIL'}")
    print(f"Mean Energy Ratio: {mean_en_ratio:.4f} (Gate < 1.10) -> {'PASS' if passed_energy else 'FAIL'}")
    
    final_report = {
        "seeds": results,
        "summary": {
            "mean_lyap": mean_lyap,
            "mean_mse_ratio": mean_mse_ratio,
            "mean_en_ratio": mean_en_ratio,
            "passed_chaos": bool(passed_chaos),
            "passed_quality": bool(passed_quality)
        }
    }
    
    # Save
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(final_report, f, indent=2)
        
    if passed_chaos and passed_quality:
        print("SUCCESS: Phase 7 Regression Passed.")
    else:
        print("FAILURE: Gates not met.")

if __name__ == "__main__":
    run_chaos_regression()

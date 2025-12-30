import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_path)

from he_core.entity import Entity
from he_core.bridge_vdp import VanDerPolEnv
from he_core.bridge import LimitCycleObsAdapter, LinearActAdapter

import json
from EXP.diag.compute.chaos_metrics import compute_lyapunov_proxy, check_broadband_spectrum

def run_vdp_chaos():
    print("Running Entity in Van der Pol (Chaos Scale) Environment...")
    
    config = {
        'seed': 42,
        'dim_q': 2,
        'kernel_type': 'plastic',
        'dt': 0.05, 
        'omega': 1.0,
        'eta': 0.05,
        'vdp_mu': 2.0, 
        'drive_amp': 2.0, 
        'drive_freq': 0.5, # Adjusted to 0.5 (Known Chaos regime for mu=2 is around freq=0.4-0.6?)
        # Let's try to find a known chaotic set. 
        # Hayashi (1964): mu=0.2, B=17, v=4... 
        # Let's stick to the previous one if it worked, or tune freq slightly.
        # Original: 0.8. Let's keep 0.5 for more complex interaction.
        'max_steps': 2000, # More steps for Lyapunov
        'active_mode': True,
        'active_gain': 0.1,
        'output_dir': 'EXP/chaos_vdp'
    }
    
    os.makedirs(os.path.join(base_path, config['output_dir']), exist_ok=True)
    
    # Setup
    env = VanDerPolEnv(config)
    obs_adapter = LimitCycleObsAdapter() 
    act_adapter = LinearActAdapter()
    entity = Entity(config)
    
    log = {'x_int': [], 'x_ext': [], 'u_self': [], 'pred_error': []}
    
    obs = env.reset(seed=config['seed'])
    done = False
    
    while not done:
        # Entity Step
        entity_obs = obs_adapter.adapt(obs)
        out = entity.step(entity_obs)
        action_env = act_adapter.adapt(out['action'])
        
        # Calc explicit prediction error locally for audit (if possible)
        # Entity doesn't export 'pred_x' in 'out'. 
        # But we can infer it or update Entity to return it.
        # For minimal change: assume e ~ u_self / gain (if simple P-control)
        # u_self = -gain * error => error = -u_self / gain
        if config['active_gain'] > 1e-6:
             pred_error_est = -out['action'] / config['active_gain']
             log['pred_error'].append(pred_error_est)
        else:
             log['pred_error'].append(np.zeros_like(out['action']))
        
        log['u_self'].append(out['action'])
        
        # Env Step
        obs_next, reward, done, info = env.step(action_env)
        
        # Log
        log['x_int'].append(out['x_int'])
        log['x_ext'].append(obs['x_ext_proxy'][:2]) 
        
        obs = obs_next
        
    # Analysis
    traj_x = np.array(log['x_ext'])
    traj_int = np.array(log['x_int']).squeeze(1)
    
    # 1. Survival Check
    e3_max = np.max(np.abs(traj_int))
    if np.isnan(traj_int).any() or np.isnan(traj_x).any():
        print("FAILED: NaNs detected.")
        return
    
    # 2. Chaos Audit (Lyapunov)
    # Use x_ext (Env State)
    lyap_est = compute_lyapunov_proxy(traj_x, config['dt'])
    spec_metrics = check_broadband_spectrum(traj_x[:, 0], config['dt'])
    
    print(f"\n--- Chaos Audit ---")
    print(f"Max Lyapunov Est: {lyap_est:.4f}")
    print(f"Spectral Entropy: {spec_metrics['entropy']:.4f}")
    print(f"Peakiness:        {spec_metrics['peakiness']:.4f}")
    
    # 3. Structural Quality Audit
    # Energy
    u_self_arr = np.array(log['u_self'])
    energy_intervention = np.mean(u_self_arr**2)
    error_arr = np.array(log['pred_error'])
    mse_error = np.mean(error_arr**2)
    
    print(f"\n--- Structural Quality ---")
    print(f"Intervention Energy:  {energy_intervention:.4f}")
    print(f"Est Prediction MSE:   {mse_error:.4f}")
    
    # Save Metrics
    metrics = {
        'e3_max_state': float(e3_max),
        'lyap_max': float(lyap_est),
        'spectral_entropy': float(spec_metrics['entropy']),
        'peakiness': float(spec_metrics['peakiness']),
        'intervention_energy': float(energy_intervention),
        'pred_mse': float(mse_error)
    }
    
    with open(os.path.join(base_path, config['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(traj_x[:, 0], traj_x[:, 1], alpha=0.6, lw=0.5)
    plt.title(f"VdP (L={lyap_est:.2f})")
    plt.xlabel("x")
    plt.ylabel("v")
    
    plt.subplot(1, 2, 2)
    plt.plot(traj_int[:, 0], traj_int[:, 1], alpha=0.6, lw=0.5, color='orange')
    plt.title(f"Entity (E={energy_intervention:.2f})")
    
    out_path = os.path.join(base_path, config['output_dir'], "vdp_phase.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    run_vdp_chaos()

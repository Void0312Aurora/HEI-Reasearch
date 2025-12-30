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

def run_vdp_chaos():
    print("Running Entity in Van der Pol (Chaos Scale) Environment...")
    
    config = {
        'seed': 42,
        'dim_q': 2,
        'kernel_type': 'plastic',
        'dt': 0.05, # Smaller DT for VdP stability
        'omega': 1.0,
        'eta': 0.05,
        'vdp_mu': 2.0, # Moderately nonlinear
        'drive_amp': 2.0, # Strong drive
        'drive_freq': 0.8, # Detuned from omega=1.0 for quasi-chaos
        'max_steps': 1000,
        'active_mode': True,
        'active_gain': 0.1,
        'output_dir': 'EXP/chaos_vdp'
    }
    
    os.makedirs(os.path.join(base_path, config['output_dir']), exist_ok=True)
    
    # Setup
    env = VanDerPolEnv(config)
    obs_adapter = LimitCycleObsAdapter() # Compatible interface (4D proxy)
    act_adapter = LinearActAdapter()
    entity = Entity(config)
    
    log = {'x_int': [], 'x_ext': []}
    
    obs = env.reset(seed=config['seed'])
    done = False
    
    while not done:
        # Entity Step
        entity_obs = obs_adapter.adapt(obs)
        out = entity.step(entity_obs)
        action_env = act_adapter.adapt(out['action'])
        
        # Env Step
        obs_next, reward, done, info = env.step(action_env)
        
        # Log
        log['x_int'].append(out['x_int'])
        log['x_ext'].append(obs['x_ext_proxy'][:2]) # Actual VdP State
        
        obs = obs_next
        
    # Analysis
    traj_x = np.array(log['x_ext'])
    traj_int = np.array(log['x_int']).squeeze(1)
    
    # E3 Safety Check
    if np.isnan(traj_int).any() or np.isnan(traj_x).any():
        print("FAILED: NaNs detected in Chaos Run.")
        return
        
    e3_max = np.max(np.abs(traj_int))
    print(f"E3 Max State: {e3_max:.4f}")
    if e3_max > 1e4:
        print("WARNING: State Explosion detected.")
    else:
        print("SUCCESS: Entity survived Chaos (Stable).")
        
    # Plot Phase Space (Traj)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(traj_x[:, 0], traj_x[:, 1], alpha=0.6)
    plt.title("VdP Phase Space (External)")
    plt.xlabel("x")
    plt.ylabel("v")
    
    plt.subplot(1, 2, 2)
    plt.plot(traj_int[:, 0], traj_int[:, 1], alpha=0.6, color='orange')
    plt.title("Entity Internal Space (q0 vs q1)")
    
    out_path = os.path.join(base_path, config['output_dir'], "vdp_phase.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    run_vdp_chaos()

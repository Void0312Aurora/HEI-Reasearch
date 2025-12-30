import sys
import os
import yaml
import torch
import numpy as np
import argparse
from datetime import datetime

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(base_path)

from he_core.entity import Entity
from he_core.bridge import LimitCycleGymEnv, LimitCycleObsAdapter, LinearActAdapter
from EXP.diag.compute.metrics import compute_d1_offline_non_degenerate, compute_d3_port_loop

def run_episode(config):
    # 1. Setup
    seed = config.get('seed', 42)
    
    # Init Env
    env = LimitCycleGymEnv(config)
    obs_adapter = LimitCycleObsAdapter()
    act_adapter = LinearActAdapter(scale=config.get('action_scale', 1.0))
    
    # Init Entity
    entity = Entity(config)
    
    # Logging
    log = {
        'x_int': [], 'x_ext': [], 'x_blanket': [], 'action': [], 'reward': []
    }
    
    # 2. Loop
    obs = env.reset(seed=seed)
    done = False
    step = 0
    
    print(f"Starting Episode (Max Steps: {env.max_steps})...")
    
    while not done:
        # Adapt Obs
        entity_obs = obs_adapter.adapt(obs)
        
        # Entity Step
        entity_out = entity.step(entity_obs)
        action_raw = entity_out['action']
        
        # Adapt Action
        action_env = act_adapter.adapt(action_raw)
        
        # Env Step
        obs_next, reward, done, info = env.step(action_env)
        
        # Log
        log['x_int'].append(entity_out['x_int'])
        log['x_blanket'].append(entity_out['x_blanket'])
        log['action'].append(action_env)
        log['reward'].append(reward)
        if 'x_ext_proxy' in obs:
            log['x_ext'].append(obs['x_ext_proxy'])
            
        obs = obs_next
        step += 1
        
    print(f"Episode Finished after {step} steps.")
    
    # 3. Analysis (Quick Diagnostics)
    traj_x_int = np.array(log['x_int']).squeeze(1) # [T, Dim]
    traj_x_ext = np.vstack(log['x_ext']) # [T, 4]
    
    print("Computing D1 (Variance)...")
    var = np.var(traj_x_int, axis=0).mean()
    print(f"Internal Variance: {var:.4f}")
    
    # 4. Save
    run_name = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(base_path, 'EXP/embodiment_runs', run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    np.savez(os.path.join(save_dir, 'traj.npz'), **log)
    print(f"Trajectory saved to {save_dir}")
    
if __name__ == "__main__":
    # Default Config
    config = {
        'seed': 42,
        'dim_q': 2,
        'kernel_type': 'plastic',
        'env_gamma': 0.1,
        'dt': 0.1,
        'max_steps': 500,
        'action_scale': 1.0,
        'online_steps': 500, # Full Online
        'offline_steps': 0
    }
    run_episode(config)

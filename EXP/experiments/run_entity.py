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
from EXP.diag.compute.metrics import compute_d1_offline_non_degenerate, compute_d3_port_loop, compute_d2_spectral, compute_a1_integrity, compute_gaussian_mi

def run_experiment_entity(config):
    # Setup
    seed = config.get('seed', 42)
    env = LimitCycleGymEnv(config)
    obs_adapter = LimitCycleObsAdapter()
    act_adapter = LinearActAdapter()
    entity = Entity(config)
    
    log = {
        'x_int': [], 'x_ext_proxy': [], 'x_blanket': [], 'u_t': [], 'step_meta': []
    }
    
    obs = env.reset(seed=seed)
    done = False
    
    # Run loop
    while not done:
        # Adapt Obs
        entity_obs = obs_adapter.adapt(obs)
        if 'x_ext_proxy' in obs:
            x_ext_proxy = obs['x_ext_proxy']
        else:
            x_ext_proxy = np.zeros(4)
            
        # Step Entity
        out = entity.step(entity_obs)
        action_env = act_adapter.adapt(out['action'])
        
        # Step Env
        obs_next, reward, done, info = env.step(action_env)
        
        # Log (Match run.py format)
        log['x_int'].append(out['x_int']) # [1, Dim]
        log['x_ext_proxy'].append(x_ext_proxy) # [4]
        log['x_blanket'].append(out['x_blanket']) # [1, 2*Dim]
        log['u_t'].append(out['u_t'])
        log['step_meta'].append(out['meta'])
        
        obs = obs_next
        
    # Analysis (Reused from metrics)
    # Analysis (Reused from metrics)
    print("Computing metrics...")
    traj_x_int = torch.tensor(np.array(log['x_int'])).squeeze(1) # [T, Dim]
    
    # D1 (Approx Velocity via Difference)
    if traj_x_int.shape[0] > 1:
        traj_v = traj_x_int[1:] - traj_x_int[:-1]
        # Pad last to match T
        traj_v = torch.cat([traj_v, traj_v[-1:]], dim=0)
    else:
        traj_v = torch.zeros_like(traj_x_int)
        
    # Metrics expect [T, B, Dim]
    d1 = compute_d1_offline_non_degenerate(traj_x_int.unsqueeze(1), traj_v.unsqueeze(1))
    
    # D3 (Approx)
    # Recover u_self? Entity state doesn't explicit log u_self in dict, but it's in x_blanket
    dim_q = config['dim_q']
    traj_blk = torch.tensor(np.array(log['x_blanket'])).squeeze(1)
    u_self = traj_blk[:, dim_q:]
    d3 = compute_d3_port_loop(traj_x_int[:, :dim_q].unsqueeze(1), u_self.unsqueeze(1))
    
    # A1
    traj_int_q0 = traj_x_int[:, 0].detach().numpy()
    traj_ext_stack = np.vstack(log["x_ext_proxy"])
    traj_ext_q0 = traj_ext_stack[:, 0]
    traj_bln_q0 = traj_blk[:, 0].detach().numpy()
    
    a1_metrics = compute_a1_integrity(traj_int_q0, traj_ext_q0, traj_bln_q0)
    
    return d1, {}, d3, a1_metrics, log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    d1, d2, d3, a1, log = run_experiment_entity(config)
    print("A1 Metrics:", a1)

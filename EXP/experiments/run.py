import sys
import os
import yaml
import torch
import numpy as np
import argparse
from datetime import datetime

# Add path to sys.path to find proto/diag modules
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
print(f"DEBUG: sys.path augmented with: {base_path}")
print(f"DEBUG: Available dirs in base_path: {os.listdir(base_path) if os.path.exists(base_path) else 'Path not found'}")

from proto.env.point_mass import PointMassEnv
from proto.kernel.kernels import SymplecticKernel, ContactKernel, FastSlowKernel
from proto.scheduler.scheduler import Scheduler
from diag.compute.metrics import compute_d1_offline_non_degenerate, compute_d3_port_loop, compute_d2_spectral

def run_experiment(config):
    # Setup
    if isinstance(config, str):
         with open(config, 'r') as f:
            config = yaml.safe_load(f)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Init Env
    env = PointMassEnv(dt=0.1)
    obs = env.reset(seed=config['seed'])
    
    # Init Kernel
    dim_q = config['dim_q']
    k_type = config['kernel_type']
    if k_type == 'symplectic':
        kernel = SymplecticKernel(dim_q)
    elif k_type == 'contact':
        kernel = ContactKernel(dim_q, damping=config['damping'])
    elif k_type == 'fast_slow':
        kernel = FastSlowKernel(dim_q, damping=config['damping'], epsilon=config['epsilon'])
    else:
        raise ValueError(f"Unknown kernel type: {k_type}")
        
    x_int = kernel.init_state(batch_size=1)
    
    # Init Scheduler
    scheduler = Scheduler(config)
    
    # Logs
    log = {
        "x_int": [],
        "x_blanket": [],
        "x_ext_proxy": [],
        "u_t": [],
        "step_meta": []
    }
    
    # Loop
    total_steps = config['total_steps']
    u_self = torch.zeros(1, dim_q) # Initial write-back
    
    # Ablation control
    force_u_self_zero = config.get("force_u_self_zero", False)
    
    for t in range(total_steps):
        # 1. Scheduler
        sched_info = scheduler.step()
        phase = sched_info['phase']
        
        # 2. Env Step (Get x_ext_proxy)
        env_action = np.random.uniform(-0.1, 0.1, size=2) # Random world drift
        obs, _, _, _ = env.step(env_action)
        x_ext_proxy = torch.tensor(obs['x_ext_proxy'], dtype=torch.float32).unsqueeze(0) # [1, 4]
        
        # 3. Input Processing
        # u_env from x_ext_proxy (Projection)
        u_env = x_ext_proxy[:, :dim_q] 
        
        # u_self generation (Mock Policy/Readout)
        if hasattr(kernel, 'dim_q'):
             # x_int has q, p, maybe s
             p_curr = x_int[:, dim_q:2*dim_q]
             # Iteration 0.2 Stabilization: Change positive feedback to negative feedback (damping)
             # Old: u_self = 0.5 * p_curr
             # New: u_self = -0.05 * p_curr (ensure net dissipation if damping=0.1)
             u_self = -0.05 * p_curr
             
        if force_u_self_zero:
            u_self = torch.zeros_like(u_self)
        
        u_t_tensor = scheduler.process_input(u_env, u_self, sched_info)
        
        # Derived x_blanket (v0 definition: u_env + u_self + phase_bit?)
        # For v0 simplicity: just concat components
        x_blanket = torch.cat([u_env, u_self], dim=1)
        
        # 4. Kernel Step
        x_int_next = kernel(x_int, u_t_tensor)
        
        # 5. Logging
        log["x_int"].append(x_int.detach().numpy()) 
        log["x_ext_proxy"].append(x_ext_proxy.detach().numpy())
        log["x_blanket"].append(x_blanket.detach().numpy())
        log["u_t"].append(u_t_tensor.detach().numpy())
        
        # step_meta structure
        log["step_meta"].append({
            "phase": phase,
            "u_source": sched_info["u_source"],
            "step": t
        })
        
        x_int = x_int_next
        
    # Process Logs
    traj_x_int = torch.tensor(np.array(log["x_int"])).squeeze(1) # T x Dim
    # Recover u_self for D3? 
    # Valid question. D3 needs u_self. Logic: u_self is part of x_blanket [u_env, u_self].
    # dim_q = config['dim_q']
    traj_x_blanket = torch.tensor(np.array(log["x_blanket"])).squeeze(1)
    traj_u_self = traj_x_blanket[:, dim_q:] 

    # Extract Offline Segment
    step_metas = log["step_meta"]
    phases = [m["phase"] for m in step_metas]
    offline_mask = np.array(phases) == "offline"
    
    # If no offline, warn
    if not np.any(offline_mask):
        print("Warning: No offline phase found.")
        d1 = {}
    else:
        q_off = traj_x_int[offline_mask, :dim_q]
        p_off = traj_x_int[offline_mask, dim_q:2*dim_q]
        d1 = compute_d1_offline_non_degenerate(q_off.unsqueeze(1), p_off.unsqueeze(1))
        
        # D2: Spectral Diagnostics (Iter 0.3 Enhanced)
        # Sample multiple points from offline phase to get robust stats
        # step_metas has {phase, u_source, step}
        # offline_mask is boolean array matching log entries
        
        offline_indices = np.where(offline_mask)[0]
        if len(offline_indices) > 0:
            # Sample up to 10 points uniformly
            num_samples = 10
            indices = np.linspace(0, len(offline_indices)-1, num_samples, dtype=int)
            sampled_indices = offline_indices[indices]
            
            d2_list = []
            for idx in sampled_indices:
                 x_curr = torch.tensor(log["x_int"][idx]).clone().detach()
                 u_curr = torch.tensor(log["u_t"][idx]).clone().detach()
                 d2_i = compute_d2_spectral(kernel, x_curr, u_curr)
                 d2_list.append(d2_i)
            
            # Average metrics
            d2 = {}
            for k in d2_list[0].keys():
                vals = [d[k] for d in d2_list]
                d2[k] = float(np.mean(vals))
            # Also log std? For now just mean.
        else:
            d2 = {"d2_max_eig": 0.0, "d2_gap": 0.0, "d2_ratio": 0.0}
        
    # D3: Whole trajectory
    d3 = compute_d3_port_loop(traj_x_int[:, :dim_q].unsqueeze(1), traj_u_self.unsqueeze(1))
    
    # Return metrics and log for external verification scripts
    return d1, d2, d3, log

def save_report(config, d1, d2, d3):
    report_dir = config['output_dir']
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(os.path.join(report_dir, run_name), exist_ok=True)
    
    report_path = os.path.join(report_dir, run_name, "report.md")
    with open(report_path, 'w') as f:
        f.write(f"# Experiment Report: {run_name}\n")
        f.write(f"Config: {config}\n\n")
        f.write("## D1: Offline Non-degenerate\n")
        for k, v in d1.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\n## D2: Spectral Diagnostics\n")
        for k, v in d2.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\n## D3: Port Loop Amplification\n")
        for k, v in d3.items():
            f.write(f"- {k}: {v:.4f}\n")
            
    print(f"Run completed. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    d1, d2, d3, _ = run_experiment(config)
    save_report(config, d1, d2, d3)

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
from proto.kernel.kernels import SymplecticKernel, ContactKernel, FastSlowKernel, ResonantKernel
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
    elif k_type == 'resonant':
        # Default omega=1.0 unless in config
        omega = config.get('omega', 1.0)
        kernel = ResonantKernel(dim_q, damping=config['damping'], omega=omega)
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
    
    # Replay Buffer (for u_env)
    u_env_online_buffer = []
    
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
        
        if phase == "online":
             u_env_online_buffer.append(u_env.clone())
        
        # Replay / Mismatch Control (Iter 1.0)
        # If offline and u_source is replay, we need to override u_env
        if phase == "offline" and sched_info["u_source"] == "replay":
             # replay_index calculation
             # For simplicity, map t-T_online to recorded buffer
             # We assume online phase length is fixed or tracked.
             # sched_info doesn't tell us "step within phase".
             # Hack: use generic modulo or tracking.
             # Better: buffer length.
             replay_idx = (t) % len(u_env_online_buffer) if len(u_env_online_buffer) > 0 else 0
             
             if config.get("replay_shuffle", False):
                 # Full Mismatch: random frame
                 replay_idx = np.random.randint(0, len(u_env_online_buffer))
             elif config.get("replay_reverse", False):
                 # Time-Reverse
                 # Map t to len-t-1 (modulo length)
                 # effective_t = t % len
                 # reverse_t = len - effective_t - 1
                 eff_t = t % len(u_env_online_buffer)
                 replay_idx = len(u_env_online_buffer) - eff_t - 1
             elif config.get("replay_block_shuffle", False):
                 # Block Shuffle (approx implementation)
                 # Let's say block size = 10% of length
                 L = len(u_env_online_buffer)
                 block_size = max(1, int(L * 0.1))
                 num_blocks = L // block_size
                 # Has to remain consistent for the run?
                 # If we do it stateless here, it's hard. 
                 # run.py is stateless per step.
                 # We need to pre-compute the index map if we want consistent block shuffle.
                 # HACK: Use a seeded permutation based on block index.
                 eff_t = t % L
                 block_idx = eff_t // block_size
                 # deterministic shuffle of blocks based on seed
                 np.random.seed(config['seed'] + 123) # consistent shuffle
                 perm = np.random.permutation(num_blocks + 1) # simple
                 # But we can't seed every step or it resets?
                 # We are inside the loop. 
                 # OK, for now, simple random block offset?
                 # Better: Simple Block Swap. 
                 # Map block i to block (i + N/2) % N? (Cyclic shift)
                 # Map block i to block N-i? (Block reverse)
                 # Let's do Block Reverse: Blocks are ordered, but time within block is reversed? No.
                 # Let's do Block Permutation:
                 # block_idx -> perm[block_idx]
                 # We need `perm` to be static.
                 # Reconstruct perm every step is fine if deterministic.
                 perm = np.random.permutation(num_blocks if num_blocks > 0 else 1)
                 new_block_idx = perm[block_idx % len(perm)]
                 offset = eff_t % block_size
                 replay_idx = new_block_idx * block_size + offset
                 if replay_idx >= L: replay_idx = L - 1
             
             u_env_replay = u_env_online_buffer[replay_idx]
             
             # process_input usually ignores u_env in offline, UNLESS we pass it explicitly as "override"
             # But scheduler.process_input has logic:
             # if meta["u_source"] == "replay": u_combined = u_self
             # Wait, scheduler logic I wrote in Step 182 was:
             # if meta["u_source"] == "replay": u_combined = u_self
             # That means it ignored u_env!
             # We need to fix Scheduler logic to actually USE u_env if replay.
             # Or we do it here:
             u_t_tensor = u_self + u_env_replay # Add experience to self-loop?
             # SPEC says: Replay means driving by memory.
             # If "experience modulated", maybe it's u_self + u_replay.
             # Let's assume input is additive.
             
             pass # continue to normal process_input, but we need to pass this replay u_env
             # But process_input signature is (u_env, u_self, meta)
             # So we just pass u_env = u_env_replay
             u_env = u_env_replay

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

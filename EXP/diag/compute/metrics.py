import torch
import numpy as np
from typing import Dict, List

def compute_d1_offline_non_degenerate(traj_x: torch.Tensor, traj_v: torch.Tensor) -> Dict[str, float]:
    """
    D1: Check if offline trajectory is non-degenerate.
    Returns:
        is_fixed_point: 1.0 if velocity is near zero.
        is_periodic: >0 if high autocorrelation.
        variance: state variance (should be > 0).
    """
    # traj_x: Time x Batch x Dim
    # We take the mean across batch for simplified metric
    
    # 1. Variance Check
    var = torch.var(traj_x, dim=0).mean().item()
    
    # 2. Fixed Point Check (Velocity Magnitude)
    # v is already passed or can be diff of x
    avg_speed = torch.norm(traj_v, dim=-1).mean().item()
    is_fixed_point = 1.0 if avg_speed < 1e-3 else 0.0
    
    # 3. Periodicity (Autocorrelation of speed)
    # Simple lag-1 autocorrelation of speed
    speed = torch.norm(traj_v, dim=-1).mean(dim=1) # Time
    if speed.shape[0] > 10 and speed.std() > 1e-5:
        speed_centered = speed - speed.mean()
        # Lag 5
        lag = 5
        ac = (speed_centered[lag:] * speed_centered[:-lag]).mean() / (speed.var() + 1e-8)
        periodicity = ac.item()
    else:
        periodicity = 0.0
        
    return {
        "d1_variance": var,
        "d1_speed": avg_speed,
        "d1_fixed_point": is_fixed_point,
        "d1_periodicity": periodicity
    }

def compute_d3_port_loop(traj_x: torch.Tensor, u_self_traj: torch.Tensor) -> Dict[str, float]:
    """
    D3: Quantify Port Closed-loop Amplification.
    Correlation between u_self(t) and dx(t+1) or x(t+1).
    """
    # Simple correlation magnitude
    # u_self: T x B x Dim
    # x: T x B x Dim
    
    if u_self_traj is None or u_self_traj.shape[0] == 0:
        return {"d3_amplification": 0.0}
        
    # Align time: u_self[t] affects x[t+1]
    u = u_self_traj[:-1]
    dx = traj_x[1:] - traj_x[:-1]
    
    # Flatten
    u_flat = u.reshape(-1)
    dx_flat = dx.reshape(-1)
    
    if u_flat.std() < 1e-5:
        return {"d3_amplification": 0.0}
        
    corr = torch.corrcoef(torch.stack([u_flat, dx_flat]))[0, 1].item()
    
    # Amplification gain (Energy ratio)
    gain = (dx.norm() / (u.norm() + 1e-8)).item()
    
    return {
        "d3_correlation": corr,
        "d3_gain": gain
    }

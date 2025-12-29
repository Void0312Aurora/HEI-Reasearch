import torch
import numpy as np
from typing import Dict, List

def compute_d1_offline_non_degenerate(traj_q: torch.Tensor, traj_v: torch.Tensor) -> Dict[str, float]:
    """
    D1: Check if offline trajectory is non-degenerate.
    traj_q: State
    traj_v: Velocity/Momentum
    Returns:
        is_fixed_point: 1.0 if velocity is near zero.
        is_periodic: >0 if high autocorrelation.
        variance: state variance (should be > 0).
    """
    # traj_x (q): Time x Batch x Dim
    # We take the mean across batch for simplified metric
    
    # 1. Variance Check
    var = torch.var(traj_q, dim=0).mean().item()
    
    # 2. Fixed Point Check (Velocity Magnitude)
    # v is already passed or can be diff of x
    avg_speed = torch.norm(traj_v, dim=-1).mean().item()
    is_fixed_point = 1.0 if avg_speed < 1e-3 else 0.0
    
    # 4. Periodicity (Autocorrelation of speed)
    # Simple lag-1 autocorrelation of speed
    speed = torch.norm(traj_v, dim=-1).mean(dim=1) # Time
    if speed.shape[0] > 10 and speed.std() > 1e-5:
        speed_centered = speed - speed.mean()
        # Lag 1 for basic trend
        lag1 = 1
        ac1 = (speed_centered[lag1:] * speed_centered[:-lag1]).mean() / (speed.var() + 1e-8)
        periodicity = ac1.item()
        
        # Lag 5 for short-period check
        lag5 = 5
        if speed.shape[0] > lag5:
            ac5 = (speed_centered[lag5:] * speed_centered[:-lag5]).mean() / (speed.var() + 1e-8)
            lag_corr = ac5.item()
        else:
            lag_corr = 0.0
    else:
        periodicity = 0.0
        lag_corr = 0.0
        
    # 5. Growth Check (Log Norm Slope or Max Excursion)
    # Check if state explodes
    max_excursion = torch.norm(traj_q, dim=-1).max().item() # Max displacement
        
    return {
        "d1_variance": var,
        "d1_speed": avg_speed,
        "d1_fixed_point": is_fixed_point,
        "d1_periodicity": periodicity,
        "d1_lag5_corr": lag_corr,
        "d1_max_excursion": max_excursion
    }

def compute_d2_spectral(kernel, x_int: torch.Tensor, u_t: torch.Tensor) -> Dict[str, float]:
    """
    D2: Spectral Diagnostics (Jacobian).
    Computes spectral gap and max eigenvalue of J = dx_{t+1}/dx_t.
    """
    # x_int: [B, Dim] (Need gradient)
    x_int = x_int.detach().clone().requires_grad_(True)
    u_t = u_t.detach().clone() # Constant input for linearization check
    
    # One-step map
    x_next = kernel(x_int, u_t)
    
    # Compute Jacobian J: [Dim, Dim]
    dim = x_int.shape[1]
    J = torch.zeros(dim, dim)
    
    for i in range(dim):
        # We need retain_graph=True because we backprop multiple times (once per output dim)
        grad = torch.autograd.grad(x_next[:, i].sum(), x_int, create_graph=False, retain_graph=True)[0]
        J[i] = grad[0]
        
    # Eigenvalues
    try:
        J_np = J.detach().numpy()
        eigvals = np.linalg.eigvals(J_np)
        sorted_indices = np.argsort(np.abs(eigvals))[::-1] # Sort desc
        sorted_eigs = eigvals[sorted_indices]
        magnitudes = np.abs(sorted_eigs)
        
        # Top-K (K=4)
        K = 4
        top_k = [float(m) for m in magnitudes[:K]]
        while len(top_k) < K:
            top_k.append(0.0)
            
        lambda_1 = magnitudes[0]
        lambda_2 = magnitudes[1] if dim > 1 else 0.0
        lambda_3 = magnitudes[2] if dim > 2 else 0.0
        
        # Stability (Max Eig)
        stability = lambda_1
        
        # Spectral Gaps
        gap12 = lambda_1 - lambda_2
        gap23 = lambda_2 - lambda_3
        ratio = lambda_1 / (lambda_2 + 1e-8)
        
        # SVD (Singular Values)
        # U, S, Vh = svd(J)
        U_svd, S_svd, Vh_svd = np.linalg.svd(J_np)
        sigma_1 = S_svd[0]
        sigma_2 = S_svd[1] if dim > 1 else 0.0
        gap_svd = sigma_1 - sigma_2
        
    except Exception as e:
        print(f"Eig/SVD decomp failed: {e}")
        stability, gap12, gap23, ratio, gap_svd, sigma_1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        top_k = [0.0]*4
        
    return {
        "d2_max_eig": stability, # |lambda_1|
        "d2_gap12": gap12,
        "d2_gap23": gap23,
        "d2_ratio": ratio,
        "d2_svd_max": sigma_1,
        "d2_svd_gap": gap_svd,
        "d2_top1": top_k[0],
        "d2_top2": top_k[1],
        "d2_top3": top_k[2]
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

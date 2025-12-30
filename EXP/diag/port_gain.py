import torch
import numpy as np
from typing import Callable, List

def compute_port_gain(entity, input_seq: torch.Tensor, steps: int = 10) -> float:
    """
    Protocol 5 Proxy: Small Gain Property.
    Measure sensitivity: ||dx_out / du_in||.
    If > 1.0 heavily, we have expansion (instability).
    """
    # 1. Perturb Input
    # input_seq (B, seq_len, dim_ext)
    # Take one step input
    u_base = input_seq[:, 0, :]
    u_pert = u_base + torch.randn_like(u_base) * 0.01
    
    # 2. Step Entity
    # We need to access entity state. Assuming entity reset.
    state_0 = entity.state.clone()
    
    out_base = entity.step({'x_ext_proxy': u_base.numpy().flatten()})['active']
    
    entity.state = state_0 # Reset
    out_pert = entity.step({'x_ext_proxy': u_pert.numpy().flatten()})['active']
    
    # 3. Ratio
    # ||out_diff|| / ||in_diff||
    
    diff_in = np.linalg.norm(u_pert.numpy() - u_base.numpy())
    diff_out = np.linalg.norm(out_pert - out_base)
    
    gain = diff_out / (diff_in + 1e-9)
    print(f"[Protocol 5] Port Gain: {gain:.4f}")
    return gain

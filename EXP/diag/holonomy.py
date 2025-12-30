import torch
import numpy as np
from typing import Dict, Any, List

def compute_holonomy(connection_module, loop_points: List[torch.Tensor], v0: torch.Tensor) -> Dict[str, float]:
    """
    Computes Holonomy of the connection along a discrete loop.
    Loop: q_0 -> q_1 -> ... -> q_N -> q_0.
    v_new = Transport(q_i, q_{i+1}, v_old).
    
    Return ||v_final - v0||.
    """
    v_curr = v0.clone()
    
    # Traverse
    path = loop_points + [loop_points[0]] # Close loop
    
    for i in range(len(path) - 1):
        q_from = path[i]
        q_to = path[i+1]
        
        # Connection requires Batch dim
        if q_from.dim() == 1: q_from = q_from.unsqueeze(0)
        if q_to.dim() == 1: q_to = q_to.unsqueeze(0)
        if v_curr.dim() == 1: v_curr = v_curr.unsqueeze(0)
        
        v_curr = connection_module(q_from, q_to, v_curr)
        
    v_final = v_curr
    
    diff = (v_final - v0.unsqueeze(0)).norm().item()
    ratio = diff / (v0.norm().item() + 1e-9)
    
    return {
        "holonomy_error": float(diff),
        "holonomy_ratio": float(ratio),
        "is_flat": ratio < 0.01
    }

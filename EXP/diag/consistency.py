import torch
import numpy as np
from typing import Dict, Any

def compute_consistency_drift(trajectory_stats: Dict[str, Any]) -> Dict[str, float]:
    """
    Protocol A3: Consistency Drift Analysis.
    trajectory_stats: List of dict steps, containing 'H_val' or 's_val'.
    
    Returns:
        drift_mean: Average change per step (should be <= 0 for dissipative).
        drift_std: Volatility of drift.
        boundedness: Check if V stays within range.
    """
    # Extract scalar metric (Hamiltonian H or Contact S)
    # Usually H decreases in dissipative system? 
    # Or S increases? 
    # In Contact Eq: dot_s = p dH/dp - H.
    # If H = 0.5p^2 + V + alpha*s.
    # Metric: Let's track 's_val' (Action Budget) or 'p_norm' (Activity).
    
    # We use 'H_val' if available, else 's_val' proxy.
    
    keys = ['H_val', 's_val', 'p_norm']
    target_key = None
    for k in keys:
        if k in trajectory_stats[0]:
            target_key = k
            break
            
    if target_key is None:
        return {"error": "No scalar metric found"}
        
    values = np.array([step[target_key] for step in trajectory_stats])
    
    # 1. Compute Differences
    deltas = np.diff(values)
    
    drift_mean = np.mean(deltas)
    drift_std = np.std(deltas)
    final_val = values[-1]
    
    # 2. Check "Self-Supervised" (Directed)
    # If drift_mean is significantly negative (dissipating) OR 
    # approx 0 with low std (stable).
    # Bad: Positive drift (runaway) or Zero with high std (Random Walk).
    
    status = "UNKNOWN"
    if drift_mean < -1e-5:
        status = "DISSIPATIVE"
    elif abs(drift_mean) < 1e-5:
        if drift_std < 0.1:
            status = "STABLE"
        else:
            status = "RANDOM_WALK"
    else:
        status = "RUNAWAY"
        
    return {
        "metric": target_key,
        "drift_mean": float(drift_mean),
        "drift_std": float(drift_std),
        "status": status,
        "final_val": float(final_val)
    }

def report_consistency(metrics: Dict[str, Any]) -> bool:
    if "error" in metrics:
        print(f"[Protocol A3] Error: {metrics['error']}")
        return False
        
    print(f"[Protocol A3] Status: {metrics['status']}")
    print(f"  Metric: {metrics['metric']}")
    print(f"  Drift Mean: {metrics['drift_mean']:.6f}")
    
    # Pass if Dissipative or Stable (Structured)
    # Fail if Random Walk or Runaway
    if metrics['status'] in ["DISSIPATIVE", "STABLE"]:
        print("  Result: PASS (Structured Drive)")
        return True
    else:
        print("  Result: FAIL (Unstructured/Unstable)")
        return False

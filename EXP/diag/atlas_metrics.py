import numpy as np
from typing import Dict, Any, List

def compute_atlas_metrics(trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Protocol Atlas: Coverage & Consistency.
    trajectory_data: List of steps. Must contain 'active_charts' (weights) and 'consistency_loss'.
    """
    if not trajectory_data:
        return {"error": "Empty trajectory"}
        
    # check keys
    if 'chart_weights' not in trajectory_data[0]:
        return {"error": "No chart_weights in data"}
        
    # 1. Coverage
    # Sum weights over time
    all_weights = np.array([step['chart_weights'] for step in trajectory_data]) # (T, K)
    
    total_usage = np.sum(all_weights, axis=0) # (K,)
    coverage_dist = total_usage / (np.sum(total_usage) + 1e-9)
    
    # Entropy of coverage (Balance check)
    coverage_entropy = -np.sum(coverage_dist * np.log(coverage_dist + 1e-9))
    
    # 2. Overlap Consistency
    # We want average consistency loss conditioned on Overlap > 0
    # or just mean consistency loss.
    
    consist_losses = [step.get('consistency_loss', 0.0) for step in trajectory_data]
    mean_consist_loss = np.mean(consist_losses)
    
    # 3. Switching Frequency
    # argmax change count
    dominant_charts = np.argmax(all_weights, axis=1)
    switches = np.sum(dominant_charts[1:] != dominant_charts[:-1])
    
    return {
        "coverage_distribution": coverage_dist.tolist(),
        "coverage_entropy": float(coverage_entropy),
        "mean_consistency_loss": float(mean_consist_loss),
        "switch_count": int(switches),
        "num_charts": all_weights.shape[1]
    }

def report_atlas_metrics(metrics: Dict[str, Any]) -> bool:
    if "error" in metrics:
        print(f"[Atlas] Error: {metrics['error']}")
        return False
        
    print(f"[Atlas] Coverage: {metrics['coverage_distribution']}")
    print(f"[Atlas] Consistency Loss: {metrics['mean_consistency_loss']:.4f}")
    print(f"[Atlas] Switches: {metrics['switch_count']}")
    
    # Gate: Coverage not collapsed (entropy > 0 if K>1)
    # Gate: Consistency Low (< Threshold)
    
    passed = True
    if metrics['mean_consistency_loss'] > 0.1:
         print("  FAIL: High Consistency Loss")
         passed = False
         
    if len(metrics['coverage_distribution']) > 1 and metrics['coverage_entropy'] < 0.1:
         print("  WARNING: Low Coverage Entropy (Collapse)")
         # Not necessarily fail if task requires 1 chart
         
    return passed

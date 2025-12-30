"""
Verify L2 Transfer Generators (Holonomy Loop).

This script verifies the geometric consistency of chart transitions (Parallel Transport).
It constructs a loop sequence of transitions and measures the closure error (Holonomy).

Metric:
Holonomy Error = || x_A - Phi_BA( Phi_AB(x_A) ) ||

If the connection is flat (curvature=0) and maps are consistent, error should be ~0.
If curvature exists, error reflects the curvature (Holonomy group).

This diagnostic is crucial for "Self-Supervision":
We can train transitions by minimizing this Holonomy Error.
"""
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.state import ContactState

def measure_holonomy_error(entity, chart_cycle: List[int], num_samples: int = 20) -> Tuple[float, float, List[float]]:
    """
    Measure Holonomy error for a given cycle of charts [c1, c2, ..., c1].
    Transitions must be defined in entity.atlas.
    """
    dim_q = entity.dim_q
    
    # Generate random samples in starting chart
    start_chart = chart_cycle[0]
    states = []
    for _ in range(num_samples):
        s = ContactState(dim_q)
        s.q = torch.randn(1, dim_q)
        s.p = torch.randn(1, dim_q)
        states.append(s)
        
    errors = []
    
    for k in range(num_samples):
        curr_state = states[k]
        original_flat = curr_state.flat.clone()
        
        # Traverse cycle
        valid_path = True
        for i in range(len(chart_cycle) - 1):
            src = chart_cycle[i]
            dst = chart_cycle[i+1]
            key = f"{src}_{dst}"
            
            if key not in entity.atlas.transitions:
                print(f"  Warning: Transition {src}->{dst} not defined.")
                valid_path = False
                break
                
            map_ij = entity.atlas.transitions[key]
            
            # Apply map
            new_flat = map_ij(curr_state)
            
            # Update state wrapper
            curr_state = ContactState(dim_q, 1, device=new_flat.device, flat_tensor=new_flat)
            
        if not valid_path:
            errors.append(float('nan'))
            continue
            
        # Final state vs Original
        final_flat = curr_state.flat
        error = (final_flat - original_flat).norm().item()
        errors.append(error)
        
    # Stats
    errors = [e for e in errors if not np.isnan(e)]
    if not errors:
        return float('nan'), float('nan'), []
        
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    return mean_error, max_error, errors

def setup_consistent_transitions(entity):
    """
    Manually inject consistent (inverse) transitions for testing.
    Map 0->1: Rotation + shift
    Map 1->0: Inverse Rotation - shift
    """
    dim_full = 2 * entity.dim_q + 1
    
    # 0->1
    entity.atlas.add_transition(0, 1)
    # 1->0
    entity.atlas.add_transition(1, 0)
    
    with torch.no_grad():
        # Define random orthogonal matrix Q
        Q_matrix = torch.eye(dim_full)
        # Apply explicit rotation block? For now just near identity but consistent.
        # Let's say Map_01(x) = x + d
        # Map_10(x) = x - d
        
        shift = torch.ones(dim_full) * 0.5
        
        # Linear layer: y = xA^T + b. 
        # Map 0->1: Identity, Bias = shift
        entity.atlas.transitions["0_1"].linear.weight.copy_(torch.eye(dim_full))
        entity.atlas.transitions["0_1"].linear.bias.copy_(shift)
        
        # Map 1->0: Identity, Bias = -shift
        entity.atlas.transitions["1_0"].linear.weight.copy_(torch.eye(dim_full))
        entity.atlas.transitions["1_0"].linear.bias.copy_(-shift)
        
def run_diagnostics():
    print("=== Phase 16.2: L2 Transfer Diagnostics (Holonomy) ===")
    
    config = {
        'dim_q': 2,
        'learnable_coupling': False, 
        'num_charts': 2,
        'damping': 1.0,
        'use_port_interface': True
    }
    
    entity = UnifiedGeometricEntity(config)
    
    # Test A: Random Initialization (Expect Fail/High Error)
    print("\n--- Test A: Random Transitions (Untrained) ---")
    entity.atlas.add_transition(0, 1)
    entity.atlas.add_transition(1, 0)
    
    mean_err, max_err, _ = measure_holonomy_error(entity, [0, 1, 0])
    print(f"  Mean Error: {mean_err:.4f}")
    if mean_err > 0.1:
        print("  Result: HIGH ERROR (Expected for untrained)")
    else:
        print("  Result: LOW ERROR (Unexpected)")
        
    # Test B: Consistent Transitions
    print("\n--- Test B: Consistent Transitions (Injected) ---")
    setup_consistent_transitions(entity)
    
    mean_err, max_err, _ = measure_holonomy_error(entity, [0, 1, 0])
    print(f"  Mean Error: {mean_err:.6f}")
    
    if mean_err < 1e-5:
        print("  Result: PASS (Consistent loop verified)")
    else:
        print("  Result: FAIL (Consistency setup failed)")

if __name__ == "__main__":
    run_diagnostics()


"""
Phase 21.2: Verify Holonomy (Logic Interface).
Measures the cyclic displacement (A -> -A) of the Fundamental Adaptive Entity.
"""

import os
import torch
import torch.nn as nn
from he_core.entity_v5 import UnifiedGeometricEntityV5 as UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.holonomy import HolonomyAnalyzer
from he_core.state import ContactState

def verify_holonomy():
    print("=== Phase 21.2 (v5): Holonomy Audit ===")
    
    # 1. Setup Entity (v5)
    dim_q = 64
    config = {
        'dim_q': dim_q,
        'dim_u': 32,
        'num_charts': 1,
        'learnable_coupling': True,
        'use_port_interface': False,
        'beta_kl': 0.01
    }
    entity = UnifiedGeometricEntity(config)
    
    # Re-inject Adaptive
    net_V = nn.Sequential(nn.Linear(dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(dim_q, net_V=net_V)
    
    # Crucial: Match the Training Script Structure
    # The training script sets both .generator and .internal_gen
    entity.generator = PortCoupledGenerator(adaptive, 32, True, 1)
    entity.internal_gen = adaptive # This ensures the shared reference matches loading
    
    # Try loading weights
    try:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(cur_dir, 'phase21_entity.pth')
        entity.load_state_dict(torch.load(model_path))
        print(f"Loaded trained weights from {model_path}.")
    except:
        print("Warning: Training weights not found. Running with baseline structure (Random).")
        
    entity.eval()
    
    # 2. Test Cases
    batch_size = 16
    
    # Init Start State: q=0 (Ground State)
    start_flat = torch.zeros(batch_size, 2*dim_q+1)
    start_state = ContactState(dim_q, batch_size, torch.device('cpu'), start_flat)
    
    analyzer = HolonomyAnalyzer()
    
    # Scenario A: Zero Drive (Check Drift)
    print("\n--- Test 1: Zero Drive (Drift Check) ---")
    seq_zero = [torch.zeros(batch_size, 32) for _ in range(100)] # 100 steps * 0.01 = 1.0s
    res_zero = analyzer.measure_cycle_displacement(
        entity, start_state, seq_zero, dt=0.01
    )
    print(f"Zero Drift: q={res_zero['q_disp']:.4f}, p={res_zero['p_disp']:.4f}")
    
    # Scenario B: A -> -A Cycle (Hysteresis Check)
    print("\n--- Test 2: A -> -A Cycle (Hysteresis/Memory) ---")
    seq_A = analyzer.generate_cyclic_sequence(32, steps=200, magnitude=1.0, batch_size=batch_size, style='A_minus_A')
    res_A = analyzer.measure_cycle_displacement(
        entity, start_state, seq_A, dt=0.01
    )
    print(f"Cycle Type: A (100 steps) -> -A (100 steps)")
    print(f"Holonomy Error: q={res_A['q_disp']:.4f}, p={res_A['p_disp']:.4f}")
    
    # Interpretation
    # If q_disp > 0, the system has Memory (State depended on path).
    # If q_disp ~ 0, the system is Elastic/Conservative.
    
    if res_A['q_disp'] > 1e-3:
        print("Result: System exhibits HYSTERESIS (Memory/Logic Potential).")
    else:
        print("Result: System is REVERSIBLE (Elastic).")
        
    return res_A

if __name__ == "__main__":
    verify_holonomy()

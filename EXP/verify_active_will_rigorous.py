"""
Phase 25.4: Rigorous Will Verification.
Addresses Critique: "Is Energy-Action correlation just Scale confounding?"
Method: Partial Correlation r(H, |a| | ||q||).
If H drives Action beyond simple scaling, this should be positive.
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def partial_corr(x, y, z):
    """
    Calculate partial correlation r_xy.z
    """
    xy = np.corrcoef(x, y)[0, 1]
    xz = np.corrcoef(x, z)[0, 1]
    yz = np.corrcoef(y, z)[0, 1]
    
    numerator = xy - xz * yz
    denominator = np.sqrt((1 - xz**2) * (1 - yz**2))
    
    return numerator / denominator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    args = parser.parse_args()
    
    # 1. Setup Entity
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1)
    entity.internal_gen = adaptive
    entity.to(DEVICE)
    
    # Load Trained Weights
    try:
        entity.load_state_dict(torch.load('EXP/active_agent.pth'))
        print("Loaded Trained Agent.")
    except:
        print("Warning: Could not load agent. Cannot verify rigorous will.")
        return

    # 2. Sample Data
    energies = []
    actions = []
    norms = []
    
    print("Sampling states...")
    for _ in range(200):
        # Create random state with varying scale
        scale = torch.rand(1).item() * 5.0
        q = torch.randn(1, args.dim_q, device=DEVICE) * scale
        p = torch.randn(1, args.dim_q, device=DEVICE) * scale
        s = torch.zeros(1, 1, device=DEVICE)
        
        flat = torch.cat([q, p, s], dim=1)
        # Use explicit construction
        from he_core.state import ContactState
        state_obj = ContactState(args.dim_q, 1, DEVICE, flat)
        
        with torch.no_grad():
            H = entity.internal_gen(state_obj).item()
            a = entity.get_action(flat, 'default').abs().mean().item()
            nm = q.norm().item()
            
        energies.append(H)
        actions.append(a)
        norms.append(nm)
        
    # 3. Analysis
    r_ha = np.corrcoef(energies, actions)[0, 1]
    r_hn = np.corrcoef(energies, norms)[0, 1]
    r_an = np.corrcoef(actions, norms)[0, 1]
    
    numerator = r_ha - r_hn * r_an
    denominator = np.sqrt((1 - r_hn**2) * (1 - r_an**2))
    r_partial = numerator / denominator
    
    print(f"Weights q norm: {entity.generator.ports['default'].W_action_q.norm().item():.4f}")
    print(f"Weights p norm: {entity.generator.ports['default'].W_action_p.norm().item():.4f}")
    
    print(f"r(H, |a|): {r_ha:.4f}")
    print(f"r(H, ||q||): {r_hn:.4f}")
    print(f"r(|a|, ||q||): {r_an:.4f}")
    print(f"Partial r: {r_partial:.4f}")
    
    if r_partial > 0.2:
        print(">> SUCCESS: Will Verified (Action driven by Energy beyond Scale).")
    else:
        print(">> FAILURE: Correlation is purely structural/scale-based.")

if __name__ == "__main__":
    main()

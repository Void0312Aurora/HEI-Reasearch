"""
Phase 25.3: "Will" Verification (Energy-Action Correlation).
Verifies that Action Magnitude is correlated with Internal Potential (Energy).
Hypothesis: High internal energy (High 'Stress') leads to vigorous action.
"""

import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    args = parser.parse_args()
    
    # 1. Setup Untrained Entity (Random) OR Trained
    # Even untrained, the correlation should strictly exist due to physics structure
    # if a depends on q.
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
        print("Warning: Could not load agent. Using random weights.")
    
    # 2. Sweep Energy Levels
    energies = []
    actions = []
    
    print("Sampling states...")
    for _ in range(100):
        # Create random state with varying scale
        scale = torch.rand(1).item() * 5.0
        q = torch.randn(1, args.dim_q, device=DEVICE) * scale
        p = torch.randn(1, args.dim_q, device=DEVICE) * scale
        s = torch.zeros(1, 1, device=DEVICE)
        
        flat = torch.cat([q, p, s], dim=1)
        from he_core.state import ContactState
        # Need batchdim support in ContactState constructor
        # flat is (1, dim)
        state_obj = ContactState(args.dim_q, 1, DEVICE, flat)
        
        # Calculate Energy H
        # H = K + V + S_term
        # We focus on K+V (Hamiltonian)
        with torch.no_grad():
            H = entity.internal_gen(state_obj).item()
            
            # Calculate Action
            a = entity.get_action(flat, 'default').abs().mean().item()
            
        energies.append(H)
        actions.append(a)
        
    # Plot
    plt.scatter(energies, actions, alpha=0.6)
    plt.xlabel("Internal Energy H")
    plt.ylabel("Action Magnitude |a|")
    plt.title("A3/A4: Will (Energy-Action Correlation)")
    plt.savefig('EXP/will_verification.png')
    
    # Correlation Check
    import numpy as np
    corr = np.corrcoef(energies, actions)[0, 1]
    print(f"Energy-Action Correlation: {corr:.4f}")
    
    if corr > 0.3:
        print(">> SUCCESS: Action intensity is driven by Internal Energy.")
    else:
        print(">> FAILURE: Action is decoupled from Energy.")

if __name__ == "__main__":
    main()

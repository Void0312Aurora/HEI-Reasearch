"""
Phase 24.3: Offline Cognition (A2).
Verifies that the internal soul dynamics (u=0) are modulated by previous experience.
1. Perform Task A (XOR=0).
2. Perform Task B (XOR=1).
3. Observe offline dynamics (u=0) in both cases.
4. Verify non-triviality and separability.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt

from he_core.entity_v5 import UnifiedGeometricEntityV5 as UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_fixed_xor(args):
    config = {
        'dim_q': args.dim_q, 
        'dim_u': args.dim_q, 
        'dim_z': 8, # L3 Context
        'num_charts': 1, 
        'learnable_coupling': True,
        'beta_kl': 0.01,
        'gamma_pred': 1.0,
        'use_adaptive_generator': True # Ensure we use the z-aware generator
    }
    entity = UnifiedGeometricEntity(config)
    
    # We no longer manually replace internal_gen here if config flag works.
    # But for safety and explicit audit, let's verify or reinject.
    # Actually v5 init now handles it if 'use_adaptive_generator' is True.
    # We just need to ensure the optimizer sees the right parameters.
    
    entity.to(DEVICE)
    
    classifier = nn.Linear(args.dim_q, 2).to(DEVICE)
    opt = optim.Adam(list(entity.parameters()) + list(classifier.parameters()), lr=1e-2)
    
    print("  Training Phase (100 epochs)...")
    for ep in range(100):
        # XOR data
        A = torch.randint(0, 2, (args.batch_size,)).float() * 2 - 1
        B = torch.randint(0, 2, (args.batch_size,)).float() * 2 - 1
        y = (A * B < 0).long()
        u = torch.zeros(args.batch_size, 25, args.dim_q, device=DEVICE)
        u[:, 5, 0] = A * 10.0
        u[:, 15, 0] = B * 10.0
        
        curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)
        for t in range(25):
            out = entity.forward_tensor(curr, {'default': u[:, t, :]}, args.dt)
            curr = out['next_state_flat']
        
        loss = nn.functional.cross_entropy(classifier(curr[:, :args.dim_q]), y.to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        
    return entity, classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    
    print("=== Phase 24.3 (v5): Offline Cognition Audit ===")
    print("Training reference model...")
    entity, classifier = train_fixed_xor(args)
    entity.eval()
    
    print("\nRecording Offline Trajectories (u=0) and F-functional...")
    
    trajs = {}
    f_vals = {}
    
    for name, a_b in [('Logic-0', (1, 1)), ('Logic-1', (1, -1))]:
        u = torch.zeros(args.batch_size, 25, args.dim_q, device=DEVICE)
        u[:, 5, 0] = a_b[0] * 10.0
        u[:, 15, 0] = a_b[1] * 10.0
        
        # 1. Online Segment
        curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)
        for t in range(25):
            with torch.no_grad():
                out = entity.forward_tensor(curr, {'default': u[:, t, :]}, args.dt)
                curr = out['next_state_flat']
        
        # 2. Offline Segment (50 steps of u=0)
        history = []
        fs = []
        for t in range(50):
            with torch.no_grad():
                out = entity.forward_tensor(curr, {}, args.dt)
                curr = out['next_state_flat']
                history.append(curr[0, :args.dim_q].cpu().numpy())
                fs.append(out['free_energy'].item())
        
        trajs[name] = np.array(history)
        f_vals[name] = np.array(fs)

    # Visualization & Analysis
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Trajectories
    plt.subplot(1, 2, 1)
    for name, data in trajs.items():
        plt.plot(data[:, 0], label=f"{name} (q[0])")
    plt.title("A2: Experience-conditioned q(t)")
    plt.xlabel("Time (steps)")
    plt.ylabel("Internal State q[0]")
    plt.legend()
    
    # Subplot 2: Free Energy
    plt.subplot(1, 2, 2)
    for name, data in f_vals.items():
        plt.plot(data, label=f"{name} (Free Energy)")
    plt.title("A3: Unified Functional F(t)")
    plt.xlabel("Time (steps)")
    plt.ylabel("F = V(q) + KL(z)")
    plt.legend()
    
    plt.tight_layout()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(cur_dir, 'offline_cognition_v5.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Check separability (A2)
    dist = np.linalg.norm(trajs['Logic-0'] - trajs['Logic-1'], axis=1)
    mean_dist = dist.mean()
    
    # Check F trend (A3)
    f_dec_0 = f_vals['Logic-0'][-1] < f_vals['Logic-0'][0]
    f_dec_1 = f_vals['Logic-1'][-1] < f_vals['Logic-1'][0]
    
    print(f"\nAudit Results:")
    print(f"  > A2 (Separability): Mean Dist = {mean_dist:.4f}")
    print(f"  > A3 (F-Descent Logic-0): {'SUCCESS' if f_dec_0 else 'FAILURE'}")
    print(f"  > A3 (F-Descent Logic-1): {'SUCCESS' if f_dec_1 else 'FAILURE'}")
    
    if mean_dist > 0.1 and f_dec_0 and f_dec_1:
        print("\n>> OVERALL RESULT: Hardened v5 satisfies A2/A3.")
    else:
        print("\n>> OVERALL RESULT: Partial Failure or Insufficient Convergence.")
        
    # --- New Test: L3 Binding Check ---
    print("\n--- Test: L3 Binding (Does z modulate V?) ---")
    # Take a state
    s0 = torch.zeros(1, args.dim_q, device=DEVICE)
    z_base = entity.z.detach().clone().expand(1, -1)
    z_pert = z_base + torch.randn_like(z_base)
    
    # Compute H (or V)
    # Access internal generator
    v_base = entity.net_V(torch.cat([s0, z_base], dim=1)).item()
    v_pert = entity.net_V(torch.cat([s0, z_pert], dim=1)).item()
    
    print(f"V(q, z_base) = {v_base:.4f}")
    print(f"V(q, z_pert) = {v_pert:.4f}")
    
    if abs(v_base - v_pert) > 1e-4:
        print(">> SUCCESS: V is modulated by z.")
    else:
        print(">> FAILURE: V ignores z (L3 disconnected).")

if __name__ == "__main__":
    main()

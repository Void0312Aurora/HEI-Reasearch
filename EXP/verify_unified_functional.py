"""
Phase 24.4: Unified Potential (A3).
Demonstrates the existence of a single functional F that decreases during evolution.
1. Define F as the System Energy (H).
2. Monitor F during learning.
3. Verify F decreases during offline periods (Self-organization).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    
    # 1. Setup Entity
    config = {'dim_q': args.dim_q, 'dim_u': args.dim_q, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, args.dim_q, True, 1)
    entity.internal_gen = adaptive
    entity.to(DEVICE)
    
    # Optimizer
    opt = optim.Adam(entity.parameters(), lr=1e-2)
    
    # Logic task to provide learning signal
    def get_batch():
        A = torch.randint(0, 2, (128,), device=DEVICE).float() * 2 - 1
        B = torch.randint(0, 2, (128,), device=DEVICE).float() * 2 - 1
        y = (A * B < 0).long().to(DEVICE)
        u = torch.zeros(128, 25, args.dim_q, device=DEVICE)
        u[:, 5, 0] = A * 10.0
        u[:, 15, 0] = B * 10.0
        return u, y
    
    classifier = nn.Linear(args.dim_q, 2).to(DEVICE)
    opt_total = optim.Adam(list(entity.parameters()) + list(classifier.parameters()), lr=1e-2)
    
    f_history = []
    print("Monitoring Potential F (Hamiltonian H) during training...")
    
    for ep in range(50):
        # 1. Online Training
        u, y = get_batch()
        curr = torch.zeros(128, 2*args.dim_q + 1, device=DEVICE)
        
        ep_f = 0.0
        for t in range(25):
            # Capture H
            with torch.no_grad():
                # We need to manually calculate H from generator for monitoring
                from he_core.state import ContactState
                s = ContactState(args.dim_q, 128, DEVICE, curr)
                H = entity.internal_gen(s).mean().item()
                ep_f += H
                
            out = entity.forward_tensor(curr, {'default': u[:, t, :]}, args.dt)
            curr = out['next_state_flat']
            
        loss = nn.functional.cross_entropy(classifier(curr[:, :args.dim_q]), y)
        opt_total.zero_grad(); loss.backward(); opt_total.step()
        
        f_history.append(ep_f / 25.0)
        if ep % 10 == 0:
            print(f"  Ep {ep}: Mean F = {f_history[-1]:.4f}")

    # 2. Offline Descent Check
    print("\nVerifying Offline F-Descent (Self-organization)...")
    entity.eval()
    curr_off = torch.zeros(1, 2*args.dim_q + 1, device=DEVICE)
    # Give some initial momentum
    curr_off[0, args.dim_q:2*args.dim_q] = torch.randn(args.dim_q, device=DEVICE)
    
    off_f = []
    for t in range(100):
        from he_core.state import ContactState
        s = ContactState(args.dim_q, 1, DEVICE, curr_off)
        with torch.no_grad():
            H = entity.internal_gen(s).item()
            off_f.append(H)
            
        out = entity.forward_tensor(curr_off, {}, args.dt)
        curr_off = out['next_state_flat'].detach()
        
    plt.plot(off_f)
    plt.title("A3: Unified Potential F (Hamiltonian) Offline Descent")
    plt.xlabel("Offline Time")
    plt.ylabel("F (H)")
    plt.savefig('EXP/unified_potential.png')
    
    start_f = off_f[0]
    end_f = off_f[-1]
    print(f"Offline F: Start={start_f:.4f}, End={end_f:.4f}")
    
    if end_f < start_f:
        print(">> SUCCESS: Unified Potential F decreases during offline evolution (A3).")
    else:
        print(">> FAILURE: F is increasing or static (Check damping/generator).")

if __name__ == "__main__":
    main()

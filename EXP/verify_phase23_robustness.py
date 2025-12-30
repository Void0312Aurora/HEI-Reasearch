"""
Phase 23.4: Robustness Radius Curves.
Generates Accuracy vs Amplitude and Energy vs Amplitude curves for the XOR task.
Addresses the 'Dissipative Capture Zone' hypothesis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_xor_model(args):
    dim_u = args.dim_q
    def get_batch():
        A = torch.randint(0, 2, (args.batch_size,)).float() * 2 - 1
        B = torch.randint(0, 2, (args.batch_size,)).float() * 2 - 1
        y = (A * B < 0).long()
        inp = torch.zeros(args.batch_size, 25, dim_u, device=DEVICE)
        inp[:, 5, 0] = A * 10.0
        inp[:, 15, 0] = B * 10.0
        return inp, y.to(DEVICE)

    config = {'dim_q': args.dim_q, 'dim_u': dim_u, 'num_charts': 1, 'learnable_coupling': True, 'use_port_interface': False}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    for p in net_V.parameters(): nn.init.normal_(p, std=1e-5)
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, dim_u, True, 1)
    entity.internal_gen = adaptive
    entity.to(DEVICE)
    
    classifier = nn.Linear(args.dim_q, 2).to(DEVICE)
    params = list(entity.parameters()) + list(classifier.parameters())
    opt = optim.Adam(params, lr=1e-2)
    
    # Increase training to 300 epochs to ensure solid model
    print(f"  Training reference model (LR=1e-2)...")
    for ep in range(300): 
        u, y = get_batch()
        curr = torch.zeros(u.shape[0], 2*args.dim_q + 1, device=DEVICE)
        for t in range(u.shape[1]):
            out = entity.forward_tensor(curr, u[:, t, :], args.dt)
            curr = out['next_state_flat']
        
        logits = classifier(curr[:, :args.dim_q])
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if ep % 50 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"    Ep {ep}: Loss={loss.item():.4f}, Acc={acc:.2f}")
            if acc > 0.999: break
            
    return entity, classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    
    entity, classifier = train_xor_model(args)
    entity.eval(); classifier.eval()
    
    mags = np.linspace(0.0, 20.0, 41)
    accs = []
    energies = []
    
    print("Scanning magnitudes...")
    for m in mags:
        # Test 00, 01, 10, 11
        val_acc = 0.0
        avg_energy = 0.0
        
        A_vals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        y_vals = [0, 1, 1, 0]
        
        with torch.no_grad():
            hits = 0
            e_sum = 0
            for (a, b), y in zip(A_vals, y_vals):
                u = torch.zeros(100, 25, args.dim_q, device=DEVICE)
                u[:, 5, 0] = a * m
                u[:, 15, 0] = b * m
                
                curr = torch.zeros(100, 2*args.dim_q + 1, device=DEVICE)
                for t in range(u.shape[1]):
                    # We can't use no_grad here due to ContactDynamics internal grad
                    # But we can detach the state
                    with torch.enable_grad():
                        # Set requires_grad if needed (ContactIntegrator handles it)
                        out = entity.forward_tensor(curr, u[:, t, :], args.dt)
                        curr = out['next_state_flat'].detach()
                
                logits = classifier(curr[:, :args.dim_q])
                hits += (logits.argmax(1) == y).sum().item()
                
                # Kinetic Energy Example: 0.5 * p^2
                p = curr[:, args.dim_q:2*args.dim_q]
                e_sum += (0.5 * p.pow(2).sum(dim=1)).mean().item()
            
            accs.append(hits / 400.0)
            energies.append(e_sum / 4.0)
            
        print(f"  Mag {m:.1f}: Acc={accs[-1]:.2f}, Energy={energies[-1]:.4e}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mags, accs, 'o-')
    plt.axvline(10.0, color='r', linestyle='--', label='Train Mag')
    plt.xlabel('Pulse Magnitude')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy vs Magnitude')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mags, energies, 's-')
    plt.xlabel('Pulse Magnitude')
    plt.ylabel('Mean Kinetic Energy')
    plt.title('System Energy vs Magnitude')
    
    plt.tight_layout()
    plt.savefig('EXP/robustness_curves.png')
    print("Saved results to EXP/robustness_curves.png")

if __name__ == "__main__":
    main()

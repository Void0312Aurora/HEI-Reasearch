"""
Phase 23.6: True Cyclic Holonomy (A,B,-A,-B Loop Closure).
Verifies if Logic Gates correspond to non-vanishing geometric phase.
Net drive = sum(u) = 0.
Metric: Loop Closure Error |q_final - q_start|.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.holonomy import HolonomyAnalyzer

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
    
    for ep in range(150):
        u, y = get_batch()
        curr = torch.zeros(u.shape[0], 2*args.dim_q + 1, device=DEVICE)
        for t in range(u.shape[1]):
            out = entity.forward_tensor(curr, u[:, t, :], args.dt)
            curr = out['next_state_flat']
        loss = nn.functional.cross_entropy(classifier(curr[:, :args.dim_q]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
            
    return entity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    
    print("Training reference model...")
    entity = train_xor_model(args)
    entity.eval()
    
    print("\nMeasuring Cyclic Holonomy (A -> B -> -A -> -B)...")
    # Sequence:
    # Pulse A (t=0), Pulse B (t=10), Pulse -A (t=20), Pulse -B (t=30)
    # Each pulse has duration 1 (single dt).
    
    cases = {
        'OR-Case (1, 0, -1, 0)': (1, 0, -1, 0),
        'AND-Case (1, 1, -1, -1)': (1, 1, -1, -1),
        'XOR-Logic (1, 1)': (1, 1, -1, -1) # Loop closure error should correspond to Class
    }
    
    results = {}
    
    with torch.enable_grad(): # Ensure autograd enabled for Dynamics
        for name, drivers in cases.items():
            u = torch.zeros(100, 50, args.dim_q, device=DEVICE)
            mag = 10.0
            u[:, 5, 0] = drivers[0] * mag
            u[:, 15, 0] = drivers[1] * mag
            u[:, 25, 0] = drivers[2] * mag
            u[:, 35, 0] = drivers[3] * mag
            
            curr = torch.zeros(100, 2*args.dim_q + 1, device=DEVICE)
            curr.requires_grad_(True)
            
            q0 = curr[:, :args.dim_q].clone().detach()
            
            for t in range(u.shape[1]):
                out = entity.forward_tensor(curr, u[:, t, :], args.dt)
                curr = out['next_state_flat']
                # curr.detach_() # Keep graph or not? For displacement |qT-q0|, detach is safer per step
                curr = curr.detach().requires_grad_(True)
                
            qT = curr[:, :args.dim_q]
            error = (qT - q0).norm(dim=1).mean().item()
            results[name] = error
            print(f"  {name}: Closure Error = {error:.4f}")

    print("\nConclusion:")
    if results['AND-Case (1, 1, -1, -1)'] > 0.5:
        print("  >> Non-vanishing Holonomy confirmed for logic loop.")
    else:
        print("  >> Loop closed. Logic may be encoded in endpoint, not cyclic phase.")

if __name__ == "__main__":
    main()

"""
Phase 23.3: Holonomy-Logic Correlation Verification.
Hypothesis: Logical classes correspond to distinct regions/clusters in the entity's Holonomy space (Geometric Phase).
Protocol:
1. Train Entity on XOR.
2. Generate test data for 4 XOR cases: (0,0), (0,1), (1,0), (1,1).
3. Measure 'Logic Holonomy' (Displacement |q_T - q_0|) for each.
4. Verify:
   - Same-class inputs have similar Holonomy.
   - Diff-class inputs have distinct Holonomy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.holonomy import HolonomyAnalyzer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== DATA & MODEL ==========
def generate_xor_case(batch_size, dim_u, A_val, B_val, t1=5, t2=15, seq_len=25):
    inputs = torch.zeros(batch_size, seq_len, dim_u)
    inputs[:, t1, 0] = A_val * 10.0
    inputs[:, t2, 0] = B_val * 10.0
    return inputs.to(DEVICE)

def create_trained_xor_model(args):
    # Quick train
    dim_u = args.dim_q
    
    # Data
    def get_batch():
        A = torch.randint(0, 2, (args.batch_size,)).float() * 2 - 1
        B = torch.randint(0, 2, (args.batch_size,)).float() * 2 - 1
        y = (A * B < 0).long()
        inp = torch.zeros(args.batch_size, 25, dim_u)
        inp[:, 5, 0] = A * 10.0
        inp[:, 15, 0] = B * 10.0
        return inp.to(DEVICE), y.to(DEVICE)

    # Model
    config = {'dim_q': args.dim_q, 'dim_u': dim_u, 'num_charts': 1, 'learnable_coupling': True, 'use_port_interface': False}
    entity = UnifiedGeometricEntity(config) # CPU initially
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    for p in net_V.parameters(): nn.init.normal_(p, std=1e-5)
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, dim_u, True, 1)
    entity.internal_gen = adaptive
    
    # Move EVERYTHING to device at once
    entity.to(DEVICE)
    classifier = nn.Linear(args.dim_q, 2).to(DEVICE)
    params = list(entity.parameters()) + list(classifier.parameters())
    opt = optim.Adam(params, lr=1e-2)
    
    print("  Training XOR Model...")
    for ep in range(5):
        u, y = get_batch()
        curr = torch.zeros(u.shape[0], 2*args.dim_q + 1, device=DEVICE)
        for t in range(u.shape[1]):
            out = entity.forward_tensor(curr, u[:, t, :], args.dt)
            curr = out['next_state_flat']
        loss = nn.functional.cross_entropy(classifier(curr[:, :args.dim_q]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 20 == 0:
            acc = (classifier(curr[:, :args.dim_q]).argmax(1) == y).float().mean()
            print(f"    Ep {ep}: Loss={loss.item():.4f}, Acc={acc:.2f}")
            
    return entity

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256) # Safe batch size
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 23.3: Holonomy-Logic Correlation (XOR)")
    print("=" * 60)
    
    entity = create_trained_xor_model(args)
    
    # Measure Holonomy for 4 cases
    print("\n[MEASURING HOLONOMY]")
    cases = {
        '00 (-1, -1)': (-1, -1),
        '01 (-1, +1)': (-1, 1),
        '10 (+1, -1)': (1, -1),
        '11 (+1, +1)': (1, 1)
    }
    
    results = {}
    center_states = {} # Store mean final q for clustering analysis
    
    entity.eval()
    
    # Do NOT use torch.no_grad() because ContactDynamics requires autograd for dH/dx
    # even during inference.
    
    for name, (a, b) in cases.items():
        u = generate_xor_case(100, args.dim_q, a, b)
        
        # 1. Scalar Holonomy (Displacement)
        disp = HolonomyAnalyzer.measure_logic_holonomy(entity, u, args.dt)
        mean_disp = disp.mean().item()
        std_disp = disp.std().item()
        results[name] = mean_disp
        
        print(f"  Case {name}: Displacement = {mean_disp:.4f} +/- {std_disp:.4f}")
            
    # Analysis
    print("\n[ANALYSIS]")
    # Logic Class 0: (00, 11) -> Displacements D00, D11
    # Logic Class 1: (01, 10) -> Displacements D01, D10
    
    d00 = results['00 (-1, -1)']
    d11 = results['11 (+1, +1)']
    d01 = results['01 (-1, +1)']
    d10 = results['10 (+1, -1)']
    
    print(f"  Class 0 Mean Disp: {(d00+d11)/2:.4f}")
    print(f"  Class 1 Mean Disp: {(d01+d10)/2:.4f}")
    
    diff_inter = abs((d00+d11)/2 - (d01+d10)/2)
    diff_intra_0 = abs(d00 - d11)
    diff_intra_1 = abs(d01 - d10)
    
    print(f"  Inter-Class Diff: {diff_inter:.4f}")
    print(f"  Intra-Class Diff (0): {diff_intra_0:.4f}")
    print(f"  Intra-Class Diff (1): {diff_intra_1:.4f}")
    
    if diff_inter > 2 * max(diff_intra_0, diff_intra_1):
        print("  >> PASS: Strong Correlation (Inter >> Intra)")
    else:
        print("  >> WEAK: Logic classes not clearly separated by displacement magnitude.")
        print("     (Note: Logic might be encoded in direction, not just magnitude)")

if __name__ == "__main__":
    main()

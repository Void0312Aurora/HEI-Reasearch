"""
PMI-Only Toy Experiment.
========================

Goal: Verify if PMI potential alone drives two nodes closer in isolation.
If d(A, B) does not decrease, the PMI potential implementation is flawed.

Setup:
- 2 Nodes: A at origin, B at d=1.0.
- 1 Edge: A-B with weak PMI weight.
- No Skeleton, No Repulsion, No Thermostat.
- Run 1000 steps.
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.integrator_torch import ContactIntegratorTorch, ContactConfigTorch, ContactStateTorch, IdentityInertiaTorch
from hei_n.potentials_torch import SpringAttractionTorch, SparseEdgePotentialTorch, CompositePotentialTorch
from hei_n.torch_core import minkowski_metric_torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # 1. Setup 2 Nodes
    # Both at Radius R=1.5 (cosh=2.35, sinh=2.12)
    # Node 0: (cosh(R), sinh(R), 0, ...)
    # Node 1: (cosh(R), 0, sinh(R), ...) -> Orthogonal spatial vectors
    # Distance: -ch*ch + 0 + 0 = -ch^2. d = acosh(ch^2).
    # This separates them angularly.
    
    R = 1.5
    ch = np.cosh(R)
    sh = np.sinh(R)
    
    dim = 6
    N = 2
    
    G_np = np.zeros((N, dim, dim))
    for i in range(N):
        G_np[i] = np.eye(dim)
        
    # Node 0
    G_np[0, 0, 0] = ch
    G_np[0, 1, 0] = sh
    # Frame orthog
    G_np[0, 0, 1] = sh
    G_np[0, 1, 1] = ch
    
    # Node 1 (Rotate spatial: x -> y)
    G_np[1, 0, 0] = ch
    G_np[1, 2, 0] = sh # y-axis
    # Frame orthog (col 0 and 2 interact)
    G_np[1, 0, 2] = sh
    G_np[1, 2, 2] = ch
    
    G = torch.tensor(G_np, device=device, dtype=torch.float32)
    M = torch.zeros_like(G)
    z = torch.tensor(0.0, device=device)
    state = ContactStateTorch(G=G, M=M, z=z)
    
    # Calculate Dist Init
    # inner = -ch*ch
    # d = acosh(ch^2)
    dist_init = np.arccosh(ch**2)
    
    # 2. Setup PMI Potential Only
    # Edge 0-1
    edges = torch.tensor([[0, 1]], device=device, dtype=torch.long)
    k_pmi = SpringAttractionTorch(k=1.0)
    pot = SparseEdgePotentialTorch(edges, k_pmi)
    
    oracle = CompositePotentialTorch([pot])
    inertia = IdentityInertiaTorch()
    
    # 3. Integrator Config
    config = ContactConfigTorch(
        dt=0.01,
        gamma=2.0, # Damping to settle
        fixed_point_iters=3,
        adaptive=False,
        freeze_radius=True, # Test with Freeze Radius enabled
        device=device
    )
    
    integrator = ContactIntegratorTorch(oracle, inertia, config)
    
    # 4. Run loop
    print("\nStarting PMI-Only Toy Test...")
    print(f"Initial Distance: {dist_init:.4f}")
    
    distances = []
    
    for step in range(100):
        state = integrator.step(state)
        
        # Calc distance
        x = state.x
        # d = acosh(-<x0, x1>)
        inner = -x[0,0]*x[1,0] + torch.sum(x[0,1:]*x[1,1:])
        d = torch.acosh(torch.clamp(-inner, min=1.0)).item()
        distances.append(d)
        
        if step % 10 == 0:
            print(f"Step {step}: d={d:.4f}")
            
    final_dist = distances[-1]
    print(f"Final Distance: {final_dist:.4f}")
    
    if final_dist < dist_init * 0.9:
        print(">>> RESULT: PASS (PMI effectively pulls nodes closer)")
    else:
        print(">>> RESULT: FAIL (PMI failed to reduce distance)")

if __name__ == "__main__":
    main()

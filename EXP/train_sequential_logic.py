
"""
Phase 22: Sequential Logic Emergence (Temporal XOR).
Tests if the Adaptive Geometric Entity can solve a temporal XOR task using its internal hysteresis.

Task:
- T=0: Init
- T=10: Input A (+1 or -1)
- T=20: Input B (+1 or -1)
- T=30: Classify XOR(A, B) -> 1 if A!=B, 0 if A==B.

The system must maintain state between T=10 and T=20 (Memory) and mix them non-linearly (Logic).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import numpy as np

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

def generate_xor_data(batch_size, dim_u, t_pulse1=5, t_pulse2=15, seq_len=25):
    """
    Generates (Inputs, Targets).
    Inputs: (B, SeqLen, dim_u)
    Targets: (B,) Long Tensor (0 or 1)
    """
    # Random A and B: {-1, 1}
    # We use a specific channel for the logic signal (e.g. channel 0)
    # The rest are noise or zero.
    
    A = torch.randint(0, 2, (batch_size,)).float() * 2 - 1 # -1 or 1
    B = torch.randint(0, 2, (batch_size,)).float() * 2 - 1
    
    # XOR Target: 1 if A != B, 0 if A == B
    # A != B is equivalent to A*B == -1
    labels = (A * B < 0).long()
    
    inputs = torch.zeros(batch_size, seq_len, dim_u)
    
    # Pulse 1 (Magnitude 10 for stronger signal)
    inputs[:, t_pulse1, 0] = A * 10.0
    # Pulse 2
    inputs[:, t_pulse2, 0] = B * 10.0
    
    return inputs, labels

def train_logic():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.1) # Larger dt for more dynamics
    args = parser.parse_args()
    
    print(f"=== Phase 22: Sequential XOR Training ===")
    
    # 1. Model Setup (Same Fundamental Config)
    # CRITICAL: dim_u = dim_q for valid Port Coupling (Identity Init)
    dim_u = args.dim_q
    
    config = {
        'dim_q': args.dim_q,
        'dim_u': dim_u,
        'num_charts': 1,
        'learnable_coupling': True, # Important: Learn to read input
        'use_port_interface': False
    }
    
    entity = UnifiedGeometricEntity(config)
    
    # Inject Adaptive
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    # Stable Init
    for p in net_V.parameters(): nn.init.normal_(p, mean=0.0, std=1e-5)
    
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    
    entity.generator = PortCoupledGenerator(adaptive, dim_u, True, 1)
    entity.internal_gen = adaptive
    
    # Initialize Coupling to Read Input Channel 0
    # B(q) = W q. H_port = u^T W q.
    # We need W such that u[0] affects q.
    # Default is random. Sufficient.
    
    # Classifier (Readout) - MLP for XOR non-linearity
    classifier = nn.Sequential(
        nn.Linear(args.dim_q, 64),
        nn.Tanh(),
        nn.Linear(64, 2)
    )
    
    params = list(entity.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    
    # 2. Training Loop
    history = []
    
    for epoch in range(args.epochs):
        # Generate Data (using correct dim_u)
        batch_u, batch_y = generate_xor_data(args.batch_size, dim_u)
        
        optimizer.zero_grad()
        
        # Init State (Clean)
        curr_flat = torch.zeros(args.batch_size, 2*args.dim_q + 1)
        
        # Rollout
        for t in range(batch_u.shape[1]):
            u_t = batch_u[:, t, :]
            out = entity.forward_tensor(curr_flat, u_t, args.dt)
            curr_flat = out['next_state_flat']
            
        # Readout at end
        final_q = curr_flat[:, :args.dim_q]
        logits = classifier(final_q)
        
        loss = nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        
        # Clip
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        # Metrics
        acc = (logits.argmax(dim=1) == batch_y).float().mean()
        
        if epoch % 50 == 0:
            q_norm = final_q.norm(dim=1).mean().item()
            print(f"Epoch {epoch}: Loss={loss.item():.4f} Acc={acc.item():.2f} |q|={q_norm:.4f}")
            
        if acc > 0.99:
            print(f"Converged at Epoch {epoch}! Acc=1.00")
            break
            
    # Save
    torch.save(entity.state_dict(), 'phase22_logic_entity.pth')
    if acc > 0.9:
        print("SUCCESS: Logic Gate Learned.")
    else:
        print("FAILURE: Logic Gate Failed.")

if __name__ == "__main__":
    train_logic()

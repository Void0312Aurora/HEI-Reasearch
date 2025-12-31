"""
Phase 27: Logic Selection (Self-Programming).
Task: Sequential Logic (A, B) -> Output.
Context: [1, 0] -> XOR. [0, 1] -> AND.
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
    parser.add_argument('--dim_q', type=int, default=128) # Higher dim for logic
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()
    
    # 1. Setup Entity
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 256), nn.Tanh(), nn.Linear(256, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1)
    entity.internal_gen = adaptive
    
    # Context Port (2D)
    entity.generator.add_port('context', dim_u=2)
    # Init small
    with torch.no_grad():
        entity.generator.ports['context'].W_stack.data.normal_(0, 0.01)

    # Readout (Linear)
    # We use q -> Linear -> y
    readout = nn.Linear(args.dim_q, 1, bias=True).to(DEVICE)
    entity.to(DEVICE)
    
    opt = optim.Adam(list(entity.parameters()) + list(readout.parameters()), lr=1e-3)
    
    print("Task: Logic Selection (XOR/AND).")
    
    for ep in range(args.epochs):
        # Generate Data
        # A at t=10, B at t=30.
        # 4 cases: 00, 01, 10, 11.
        # Context: 0 (XOR), 1 (AND).
        # Total 8 combos.
        
        # Vectorized batch generation
        # shape: (B, 1)
        labels_A = torch.randint(0, 2, (args.batch_size, 1), device=DEVICE).float()
        labels_B = torch.randint(0, 2, (args.batch_size, 1), device=DEVICE).float()
        
        # Context types: 0 or 1
        ctx_type = torch.randint(0, 2, (args.batch_size, 1), device=DEVICE).float() # 0 or 1
        
        # Compute Targets
        # XOR: A != B
        # AND: A * B
        target_xor = (labels_A != labels_B).float()
        target_and = (labels_A * labels_B).float()
        
        target = ctx_type * target_and + (1 - ctx_type) * target_xor
        
        # Context Vector: [1, 0] if XOR (type 0), [0, 1] if AND (type 1)
        # Type 0 -> [1, 0]. Type 1 -> [0, 1].
        ctx_vec = torch.cat([1 - ctx_type, ctx_type], dim=1) 
        
        # Run Trajectory
        batch_loss = 0.0
        curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)
        
        # Capture readout at t=50 (End)
        # Pulse width = 5 steps?
        
        for t in range(60):
            # Input u
            u_val = torch.zeros(args.batch_size, 1, device=DEVICE)
            # Pulse A (10-15)
            if 10 <= t < 15:
                u_val = u_val + labels_A
            # Pulse B (30-35)
            if 30 <= t < 35:
                u_val = u_val + labels_B
            
            # Step
            out = entity.forward_tensor(curr, {'default': u_val, 'context': ctx_vec}, args.dt)
            curr = out['next_state_flat']
            
            # Readout at every step? Or just end?
            # Let's supervise end.
            if t == 59:
                y_pred = torch.sigmoid(readout(curr[:, :args.dim_q])) # Read q only
                term_loss = nn.BCELoss()(y_pred, target)
                batch_loss += term_loss
                
                # Energy Cost
                from he_core.state import ContactState
                s_obj = ContactState(args.dim_q, args.batch_size, DEVICE, curr)
                H = entity.internal_gen(s_obj).mean()
                batch_loss += 1e-4 * H
        
        opt.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(entity.parameters(), 1.0)
        opt.step()
        
        if ep % 20 == 0:
            # Accuracy
            pred_cls = (y_pred > 0.5).float()
            acc = (pred_cls == target).float().mean()
            print(f"Ep {ep}: Loss {batch_loss.item():.4f}, Acc {acc.item():.4f}")
            
    # Save
    torch.save(entity.state_dict(), 'EXP/logic_agent.pth')
    torch.save(readout.state_dict(), 'EXP/logic_readout.pth')

if __name__ == "__main__":
    main()

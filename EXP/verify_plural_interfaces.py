"""
Phase 24.1: Plural Interfaces (A5).
Demonstrates One Core, Two Ports.
1. Train XOR on 'Token' port.
2. Add 'Symbol' port.
3. Validate that the Internal 'Soul' already understands the logic regardless of port.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_dual_port_logic(args):
    # Setup Entity
    config = {'dim_q': args.dim_q, 'dim_u': args.dim_q, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, args.dim_q, True, 1)
    entity.internal_gen = adaptive
    
    # Add Symbol Port (Port 5)
    entity.generator.add_port('symbol', args.dim_q)
    entity.to(DEVICE)
    
    # Task: XOR
    # We will train on 'default' (Token) and later test alignment on 'symbol'
    def get_batch(port_name='default'):
        A = torch.randint(0, 2, (args.batch_size,), device=DEVICE).float() * 2 - 1
        B = torch.randint(0, 2, (args.batch_size,), device=DEVICE).float() * 2 - 1
        y = (A * B < 0).long() # y can stay on CPU or move later
        
        # Scalar pulses for default, Vector pulses for symbol?
        # Let's say 'symbol' uses a fixed random projection
        u_dict = {}
        if port_name == 'default':
            inp = torch.zeros(args.batch_size, 25, args.dim_q, device=DEVICE)
            inp[:, 5, 0] = A * 10.0
            inp[:, 15, 0] = B * 10.0
            u_dict['default'] = inp
        else:
            # Symbol port uses dim_u=dim_q
            # We pulse a specific vector for '1' and '-1'
            v1 = torch.ones(args.dim_q, device=DEVICE)
            vm1 = -torch.ones(args.dim_q, device=DEVICE)
            inp = torch.zeros(args.batch_size, 25, args.dim_q, device=DEVICE)
            # A pulse
            inp[:, 5, :] = (A.view(-1, 1) > 0).float() * v1 + (A.view(-1, 1) < 0).float() * vm1
            inp[:, 5, :] *= 10.0
            # B pulse
            inp[:, 15, :] = (B.view(-1, 1) > 0).float() * v1 + (B.view(-1, 1) < 0).float() * vm1
            inp[:, 15, :] *= 10.0
            u_dict['symbol'] = inp
            
        return u_dict, y.to(DEVICE)

    classifier = nn.Linear(args.dim_q, 2).to(DEVICE)
    opt = optim.Adam(list(entity.parameters()) + list(classifier.parameters()), lr=1e-2)
    
    print("Step 1: Training XOR on 'default' port...")
    for ep in range(100):
        u_dict, y = get_batch('default')
        curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)
        for t in range(25):
            u_step = {k: v[:, t, :] for k, v in u_dict.items()}
            out = entity.forward_tensor(curr, u_step, args.dt)
            curr = out['next_state_flat']
        
        loss = nn.functional.cross_entropy(classifier(curr[:, :args.dim_q]), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0:
            acc = (classifier(curr[:, :args.dim_q]).argmax(1) == y).float().mean()
            print(f"  Ep {ep}: Acc={acc:.2f}")

    print("\nStep 2: Freezing Entity, Evaluating Core Consistency...")
    # Now we test if we can train ONLY THE SYMBOL PORT to achieve XOR
    # keeping the internal generator FIXED.
    for p in entity.internal_gen.parameters(): p.requires_grad = False
    
    # Re-init symbol port weights to see if they can align
    with torch.no_grad():
        entity.generator.ports['symbol'].W_stack.fill_(0)
        
    opt_port = optim.Adam(entity.generator.ports['symbol'].parameters(), lr=1e-2)
    
    print("Step 3: Training 'symbol' port to align with fixed logic core...")
    for ep in range(100):
        u_dict, y = get_batch('symbol')
        curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)
        for t in range(25):
            u_step = {k: v[:, t, :] for k, v in u_dict.items()}
            out = entity.forward_tensor(curr, u_step, args.dt)
            curr = out['next_state_flat']
            
        loss = nn.functional.cross_entropy(classifier(curr[:, :args.dim_q]), y)
        opt_port.zero_grad(); loss.backward(); opt_port.step()
        if ep % 20 == 0:
            acc = (classifier(curr[:, :args.dim_q]).argmax(1) == y).float().mean()
            print(f"  Ep {ep}: SymbolPort Acc={acc:.2f}")

    if acc > 0.9:
        print("\n>> SUCCESS: One Core, Two Ports. Logic transfered via latent core alignment.")
    else:
        print("\n>> FAILURE: Core too rigid or port cannot align.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    train_dual_port_logic(args)

if __name__ == "__main__":
    main()

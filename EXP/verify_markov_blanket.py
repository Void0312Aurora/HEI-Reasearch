"""
Phase 24.4: Markov Blanket (A1) Verification.
Checks if Internal State (q) is 'screened off' from External World (x) by the Blanket (Sensory s, Active a).
Method: Predictive Screening-off.
1. Train XOR model.
2. Blanket = [Input Stream, Evolution Velocity].
3. Int = q.
4. Ext = The target logic class (0 or 1).
Metric: Is Logic(Ext) identifiable from Blanket? If so, does Int add more accuracy?
Actually, in HEI, Int *is* the logic carrier.
The Markov Blanket Axiom (A1) says: Internal and External are independent given Blanket.
In our Entity:
Ext = The hidden bits (A, B) that we don't see yet.
Sensory = The pulses u_ext.
Internal = q.
Active = (Not fully defined yet, let's use the prediction of u_next or velocity).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_xor(args):
    config = {'dim_q': args.dim_q, 'dim_u': args.dim_q, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, args.dim_q, True, 1)
    entity.internal_gen = adaptive
    entity.to(DEVICE)
    classifier = nn.Linear(args.dim_q, 2).to(DEVICE)
    opt = optim.Adam(list(entity.parameters()) + list(classifier.parameters()), lr=1e-2)
    for ep in range(100):
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
    return entity

def test_markov_blanket(entity, args):
    entity.eval()
    # 1. Generate Data
    A = torch.randint(0, 2, (1000,)).float() * 2 - 1
    B = torch.randint(0, 2, (1000,)).float() * 2 - 1
    y = (A * B < 0).long()
    u = torch.zeros(1000, 25, args.dim_q, device=DEVICE)
    u[:, 5, 0] = A * 10.0
    u[:, 15, 0] = B * 10.0
    
    # Int, Blanket, Ext
    # Ext = Logic Class y
    # Sensory = pulse stream u
    # Int = q
    # Active = dot_q (approximated by delta q)
    
    internals = []
    sensories = []
    actives = []
    
    curr = torch.zeros(1000, 2*args.dim_q + 1, device=DEVICE)
    for t in range(25):
        q_prev = curr[:, :args.dim_q].clone()
        with torch.enable_grad():
            out = entity.forward_tensor(curr, {'default': u[:, t, :]}, args.dt)
            curr = out['next_state_flat'].detach()
        
        internals.append(curr[:, :args.dim_q].cpu())
        sensories.append(u[:, t, :].cpu())
        actives.append((curr[:, :args.dim_q] - q_prev).cpu()) # Active = Change in state
        
    # We take the final state (t=24)
    I = internals[-1]
    S = torch.stack(sensories, dim=1).view(1000, -1) # Full sensory history
    A = actives[-1]
    E = y.cpu()
    
    # Predictor 1: [Sensory, Active] -> Ext
    blanket = torch.cat([S, A], dim=1)
    # Predictor 2: [Sensory, Active, Internal] -> Ext
    full = torch.cat([S, A, I], dim=1)
    
    def eval_pred(X, Y):
        split = 800
        train_x, test_x = X[:split], X[split:]
        train_y, test_y = Y[:split], Y[split:]
        model = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 2)).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(200):
            opt.zero_grad()
            nn.functional.cross_entropy(model(train_x.to(DEVICE)), train_y.to(DEVICE)).backward()
            opt.step()
        with torch.no_grad():
            acc = (model(test_x.to(DEVICE)).argmax(1) == test_y.to(DEVICE)).float().mean().item()
        return acc

    acc_blanket = eval_pred(blanket, E)
    acc_full = eval_pred(full, E)
    
    print(f"Blanket-only Accuracy: {acc_blanket:.4f}")
    print(f"Full-State Accuracy:   {acc_full:.4f}")
    
    # A1 requires that Internal does not provide information about External 
    # beyond what Blanket provides.
    # In our XOR, Internal *is* the memory. 
    # Since Blanket contains 'full history', it already has the pulses.
    # So Acc_blanket should be high.
    
    if acc_full <= acc_blanket + 0.05:
        print(">> SUCCESS: Markov Blanket Screening-off verified (A1).")
    else:
        print(">> FAILURE: Internal state contains extra external info not in blanket.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    entity = train_xor(args)
    test_markov_blanket(entity, args)

if __name__ == "__main__":
    main()

"""
Phase 22 Rigorous Verification Protocol (GPU + Fixed).
Addresses critiques from temp-01.md:
1. Fixed Validation Set (not per-epoch random)
2. Linear Readout (test if dynamics provide non-linearity)
3. Memory Probe (test if state at t=delay contains A)
4. Ablation: Frozen Alpha (test if adaptive damping matters)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

# ========== DEVICE ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== DATA ==========
def generate_xor_data(batch_size, dim_u, t_pulse1=5, t_pulse2=15, seq_len=25, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    A = torch.randint(0, 2, (batch_size,)).float() * 2 - 1
    B = torch.randint(0, 2, (batch_size,)).float() * 2 - 1
    labels = (A * B < 0).long()
    
    inputs = torch.zeros(batch_size, seq_len, dim_u)
    inputs[:, t_pulse1, 0] = A * 10.0
    inputs[:, t_pulse2, 0] = B * 10.0
    
    return inputs.to(DEVICE), labels.to(DEVICE), A.to(DEVICE)

# ========== MODEL FACTORY ==========
def create_model(dim_q, dim_u, freeze_alpha=False):
    config = {
        'dim_q': dim_q,
        'dim_u': dim_u,
        'num_charts': 1,
        'learnable_coupling': True,
        'use_port_interface': False
    }
    
    entity = UnifiedGeometricEntity(config)
    
    net_V = nn.Sequential(nn.Linear(dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    for p in net_V.parameters():
        nn.init.normal_(p, mean=0.0, std=1e-5)
    
    adaptive = AdaptiveDissipativeGenerator(dim_q, net_V=net_V)
    
    if freeze_alpha:
        for p in adaptive.net_Alpha.parameters():
            p.requires_grad = False
    
    entity.generator = PortCoupledGenerator(adaptive, dim_u, True, 1)
    entity.internal_gen = adaptive
    
    return entity.to(DEVICE), adaptive

# ========== TRAIN & EVAL ==========
def train_and_evaluate(entity, classifier, train_data, val_data, args):
    classifier = classifier.to(DEVICE)
    params = list(entity.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    
    train_u, train_y, _ = train_data
    val_u, val_y, _ = val_data
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        entity.train()
        optimizer.zero_grad()
        
        curr_flat = torch.zeros(train_u.shape[0], 2*args.dim_q + 1, device=DEVICE)
        for t in range(train_u.shape[1]):
            out = entity.forward_tensor(curr_flat, train_u[:, t, :], args.dt)
            curr_flat = out['next_state_flat']
        
        logits = classifier(curr_flat[:, :args.dim_q])
        loss = nn.functional.cross_entropy(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        train_acc = (logits.argmax(dim=1) == train_y).float().mean().item()
        
        # Validation (use detach, not no_grad)
        entity.eval()
        curr_flat = torch.zeros(val_u.shape[0], 2*args.dim_q + 1, device=DEVICE)
        for t in range(val_u.shape[1]):
            out = entity.forward_tensor(curr_flat, val_u[:, t, :], args.dt)
            curr_flat = out['next_state_flat'].detach()
        
        val_logits = classifier(curr_flat[:, :args.dim_q])
        val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Train={train_acc:.2f}, Val={val_acc:.2f}")
    
    return best_val_acc

def probe_memory(entity, data, t_probe, dim_q, dt):
    """Test if state at t_probe can linearly predict A."""
    u, _, A = data
    
    entity.eval()
    # Run forward with detach (no no_grad, as entity needs autograd.grad internally)
    curr_flat = torch.zeros(u.shape[0], 2*dim_q + 1, device=DEVICE)
    for t in range(t_probe + 1):
        out = entity.forward_tensor(curr_flat, u[:, t, :], dt)
        curr_flat = out['next_state_flat'].detach()
    
    q_at_probe = curr_flat[:, :dim_q].detach()
    
    # Train linear probe
    probe = nn.Linear(dim_q, 1).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=0.01)
    A_target = (A > 0).float().unsqueeze(1)
    
    for _ in range(200):
        optimizer.zero_grad()
        pred = torch.sigmoid(probe(q_at_probe))
        loss = nn.functional.binary_cross_entropy(pred, A_target)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        pred = (probe(q_at_probe) > 0).float().squeeze()
        acc = (pred == (A > 0).float()).float().mean().item()
    
    return acc

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Phase 22 Verification Protocol (Device: {DEVICE})")
    print("=" * 60)
    
    dim_u = args.dim_q
    
    train_data = generate_xor_data(args.batch_size, dim_u, seed=42)
    val_data = generate_xor_data(256, dim_u, seed=123)
    
    results = {}
    
    # TEST 1: MLP Baseline
    print("\n[TEST 1] XOR with MLP Classifier")
    entity1, _ = create_model(args.dim_q, dim_u)
    cls1 = nn.Sequential(nn.Linear(args.dim_q, 64), nn.Tanh(), nn.Linear(64, 2))
    results['MLP'] = train_and_evaluate(entity1, cls1, train_data, val_data, args)
    print(f"  >> Best Val: {results['MLP']:.2f}")
    
    # TEST 2: Linear Classifier
    print("\n[TEST 2] XOR with Linear Classifier (Dynamics Non-linearity Test)")
    entity2, _ = create_model(args.dim_q, dim_u)
    cls2 = nn.Linear(args.dim_q, 2)
    results['Linear'] = train_and_evaluate(entity2, cls2, train_data, val_data, args)
    print(f"  >> Best Val: {results['Linear']:.2f}")
    
    # TEST 3: Memory Probe
    print("\n[TEST 3] Memory Probe at t=14")
    results['Probe'] = probe_memory(entity1, val_data, t_probe=14, dim_q=args.dim_q, dt=args.dt)
    print(f"  >> Probe Acc: {results['Probe']:.2f}")
    
    # TEST 4: Frozen Alpha Ablation
    print("\n[TEST 4] Ablation: Frozen Alpha")
    entity3, _ = create_model(args.dim_q, dim_u, freeze_alpha=True)
    cls3 = nn.Sequential(nn.Linear(args.dim_q, 64), nn.Tanh(), nn.Linear(64, 2))
    results['FrozenAlpha'] = train_and_evaluate(entity3, cls3, train_data, val_data, args)
    print(f"  >> Best Val: {results['FrozenAlpha']:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")
    
    print("\n[INTERPRETATION]")
    if results['Linear'] > 0.7:
        print("  ✓ Linear works -> Dynamics provide non-linearity!")
    else:
        print("  ✗ Linear fails -> Non-linearity from MLP only.")
    
    if results['Probe'] > 0.7:
        print("  ✓ Probe works -> State retains A information!")
    else:
        print("  ✗ Probe fails -> No memory of A.")
    
    if results['FrozenAlpha'] < results['MLP'] - 0.1:
        print("  ✓ Frozen hurts -> Adaptive damping critical!")
    else:
        print("  ~ Frozen similar -> Adaptive damping not critical.")

if __name__ == "__main__":
    main()

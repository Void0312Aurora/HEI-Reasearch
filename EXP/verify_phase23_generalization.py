"""
Phase 23.1: Generalization Tests for XOR.
Tests:
1. Time Gap Generalization (train gap=10, test gap=6,8,12,15)
2. Amplitude Generalization (train pulse=10, test pulse=5,7,12)
3. Probe Independence (separate probe-train and probe-test)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== DATA ==========
def generate_xor_data(batch_size, dim_u, t_pulse1=5, t_pulse2=15, seq_len=25, pulse_mag=10.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    A = torch.randint(0, 2, (batch_size,)).float() * 2 - 1
    B = torch.randint(0, 2, (batch_size,)).float() * 2 - 1
    labels = (A * B < 0).long()
    
    inputs = torch.zeros(batch_size, seq_len, dim_u)
    inputs[:, t_pulse1, 0] = A * pulse_mag
    inputs[:, t_pulse2, 0] = B * pulse_mag
    
    return inputs.to(DEVICE), labels.to(DEVICE), A.to(DEVICE)

# ========== MODEL ==========
def create_model(dim_q, dim_u):
    config = {'dim_q': dim_q, 'dim_u': dim_u, 'num_charts': 1, 'learnable_coupling': True, 'use_port_interface': False}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    for p in net_V.parameters(): nn.init.normal_(p, mean=0.0, std=1e-5)
    adaptive = AdaptiveDissipativeGenerator(dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, dim_u, True, 1)
    entity.internal_gen = adaptive
    return entity.to(DEVICE)

# ========== TRAIN ==========
def train_model(entity, classifier, train_data, args):
    classifier = classifier.to(DEVICE)
    params = list(entity.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    train_u, train_y, _ = train_data
    
    for epoch in range(args.epochs):
        entity.train()
        optimizer.zero_grad()
        curr = torch.zeros(train_u.shape[0], 2*args.dim_q + 1, device=DEVICE)
        for t in range(train_u.shape[1]):
            out = entity.forward_tensor(curr, train_u[:, t, :], args.dt)
            curr = out['next_state_flat']
        logits = classifier(curr[:, :args.dim_q])
        loss = nn.functional.cross_entropy(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    return entity, classifier

def evaluate(entity, classifier, test_data, args):
    test_u, test_y, _ = test_data
    entity.eval()
    curr = torch.zeros(test_u.shape[0], 2*args.dim_q + 1, device=DEVICE)
    for t in range(test_u.shape[1]):
        out = entity.forward_tensor(curr, test_u[:, t, :], args.dt)
        curr = out['next_state_flat'].detach()
    logits = classifier(curr[:, :args.dim_q])
    return (logits.argmax(dim=1) == test_y).float().mean().item()

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    
    dim_u = args.dim_q
    
    print("=" * 60)
    print(f"Phase 23.1: Generalization Tests (Device: {DEVICE})")
    print("=" * 60)
    
    # Train on standard config
    print("\n[TRAINING] Standard XOR (gap=10, pulse=10)")
    train_data = generate_xor_data(args.batch_size, dim_u, t_pulse1=5, t_pulse2=15, pulse_mag=10.0, seed=42)
    entity = create_model(args.dim_q, dim_u)
    classifier = nn.Linear(args.dim_q, 2).to(DEVICE)
    entity, classifier = train_model(entity, classifier, train_data, args)
    
    # Test on standard
    val_std = generate_xor_data(256, dim_u, t_pulse1=5, t_pulse2=15, pulse_mag=10.0, seed=123)
    acc_std = evaluate(entity, classifier, val_std, args)
    print(f"  Standard Val: {acc_std:.2f}")
    
    results = {'Standard': acc_std}
    
    # ========== TEST 1: Time Gap Generalization ==========
    print("\n[TEST 1] Time Gap Generalization")
    gaps = [6, 8, 12, 15]
    for gap in gaps:
        t2 = 5 + gap
        seq_len = t2 + 10
        test_data = generate_xor_data(256, dim_u, t_pulse1=5, t_pulse2=t2, seq_len=seq_len, pulse_mag=10.0, seed=200+gap)
        acc = evaluate(entity, classifier, test_data, args)
        results[f'Gap_{gap}'] = acc
        print(f"  Gap={gap}: {acc:.2f}")
    
    # ========== TEST 2: Amplitude Generalization ==========
    print("\n[TEST 2] Amplitude Generalization")
    pulses = [5.0, 7.0, 12.0]
    for pulse in pulses:
        test_data = generate_xor_data(256, dim_u, t_pulse1=5, t_pulse2=15, pulse_mag=pulse, seed=300+int(pulse))
        acc = evaluate(entity, classifier, test_data, args)
        results[f'Pulse_{int(pulse)}'] = acc
        print(f"  Pulse={int(pulse)}: {acc:.2f}")
    
    # ========== TEST 3: Probe Independence ==========
    print("\n[TEST 3] Probe Independence (Split Val)")
    # Get state at t=14
    probe_data = generate_xor_data(512, dim_u, t_pulse1=5, t_pulse2=15, pulse_mag=10.0, seed=400)
    entity.eval()
    u, _, A = probe_data
    curr = torch.zeros(512, 2*args.dim_q + 1, device=DEVICE)
    for t in range(15):  # Stop before B
        out = entity.forward_tensor(curr, u[:, t, :], args.dt)
        curr = out['next_state_flat'].detach()
    q_probe = curr[:, :args.dim_q].detach()
    
    # Split
    q_train, q_test = q_probe[:256], q_probe[256:]
    A_train, A_test = A[:256], A[256:]
    
    # Train probe
    probe = nn.Linear(args.dim_q, 1).to(DEVICE)
    opt = optim.Adam(probe.parameters(), lr=0.01)
    for _ in range(200):
        opt.zero_grad()
        pred = torch.sigmoid(probe(q_train))
        loss = nn.functional.binary_cross_entropy(pred, (A_train > 0).float().unsqueeze(1))
        loss.backward()
        opt.step()
    
    # Test probe
    with torch.no_grad():
        pred = (probe(q_test) > 0).float().squeeze()
        probe_acc = (pred == (A_test > 0).float()).float().mean().item()
    results['ProbeIndep'] = probe_acc
    print(f"  Probe (Independent Test): {probe_acc:.2f}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        status = "✓" if v > 0.8 else "~" if v > 0.6 else "✗"
        print(f"  {status} {k}: {v:.2f}")
    
    # Overall
    gen_accs = [results.get(f'Gap_{g}', 0) for g in gaps] + [results.get(f'Pulse_{int(p)}', 0) for p in pulses]
    avg_gen = sum(gen_accs) / len(gen_accs)
    print(f"\n  Avg Generalization: {avg_gen:.2f}")
    if avg_gen > 0.8:
        print("  >> PASS: Strong Generalization")
    elif avg_gen > 0.6:
        print("  >> PARTIAL: Moderate Generalization")
    else:
        print("  >> FAIL: Poor Generalization")

if __name__ == "__main__":
    main()

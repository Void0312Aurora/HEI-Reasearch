"""
Phase 28.1 Verification: Geometric Curvature (The Patch).
Addresses `temp-04.md` critique to strictly isolate Geometry from Computation.

Protocol:
1. Initialization: Apply Pulse (u=1) under Neutral Context (0,0) -> q_base.
2. Commutator Test:
   - Path A: q_base -> Ctx(XOR) -> Ctx(AND) -> q_AB.
   - Path B: q_base -> Ctx(AND) -> Ctx(XOR) -> q_BA.
   - Metric: ||q_AB - q_BA||_norm.
   - Semantic: |y(q_AB) - y(q_BA)|.
3. Loop Test:
   - Path: q_base -> Ctx(XOR) -> q_1 -> Ctx(AND) -> Ctx(XOR) -> q_2.
   - Metric: ||q_1 - q_2||_norm.

If System is Non-Abelian, q_AB != q_BA.
If System is Elastic/Consistent, q_1 ~ q_2.
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_readout(entity, readout_head, q_flat):
    # q_flat is (1, 2*dim +1)
    # usually q is first dim_q
    dim_q = entity.internal_gen.dim_q
    q = q_flat[:, :dim_q]
    y_logit = readout_head(q)
    y_prob = torch.sigmoid(y_logit).item()
    return y_prob

def run_stage(entity, curr, context_vec, steps=60, u_pulse=None, pulse_window=None):
    dt = 0.1
    # qs = []
    
    for t in range(steps):
        if t % 10 == 0:
             print(f"    Step {t}/{steps}")
        u_val = torch.zeros(1, 1, device=DEVICE)
        
        # Apply Pulse if defined
        if u_pulse is not None and pulse_window is not None:
             if pulse_window[0] <= t < pulse_window[1]:
                 u_val += u_pulse
                 
        ctx_tensor = context_vec.to(DEVICE)
        
        out = entity.forward_tensor(curr, {'default': u_val, 'context': ctx_tensor}, dt)
        curr = out['next_state_flat'].detach() # Crucial: Detach to prevent infinite graph growth
        # qs.append(curr)
        
    return curr, []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='EXP/logic_agent.pth')
    parser.add_argument('--readout_path', type=str, default='EXP/logic_readout.pth')
    parser.add_argument('--dim_q', type=int, default=128)
    args = parser.parse_args()
    
    # Setup
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 256), nn.Tanh(), nn.Linear(256, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1)
    entity.internal_gen = adaptive
    entity.generator.add_port('context', dim_u=2)
    entity.to(DEVICE)
    
    readout = nn.Linear(args.dim_q, 1)
    readout.to(DEVICE)
    
    try:
        entity.load_state_dict(torch.load(args.model_path))
        readout.load_state_dict(torch.load(args.readout_path))
        print("Models loaded.")
    except:
        print("Model load failed.")
        return
        
    entity.eval()
    
    # Vectors
    ctx_neutral = torch.tensor([[0.0, 0.0]])
    ctx_xor = torch.tensor([[1.0, 0.0]])
    ctx_and = torch.tensor([[0.0, 1.0]])
    
    T = 20 # Reduced steps to prevent crash, enough for switching
    
    print("\n--- Phase 28.1: Geometric Curvature Audit (Constant Input version) ---")
    
    # 1. Initialization
    # Start from Zero
    curr = torch.zeros(1, 2*args.dim_q + 1, device=DEVICE)
    # No Neutral Pulse Stage. We assume Input is Constant u=1.0.
    q_base = curr
    
    # 2. Commutator Test (With Constant Input u=1.0)
    # Path A: XOR -> AND
    print("\nTest 1: Commutator ([XOR, AND] | u=1)")
    # Run Stage A1 (XOR) with u=1
    q_A1, _ = run_stage(entity, q_base, ctx_xor, steps=T, u_pulse=1.0, pulse_window=(0, T))
    # Run Stage A2 (AND) with u=1
    q_AB, _ = run_stage(entity, q_A1, ctx_and, steps=T, u_pulse=1.0, pulse_window=(0, T))
    y_AB = get_readout(entity, readout, q_AB)
    
    # Path B: AND -> XOR
    q_B1, _ = run_stage(entity, q_base, ctx_and, steps=T, u_pulse=1.0, pulse_window=(0, T))
    q_BA, _ = run_stage(entity, q_B1, ctx_xor, steps=T, u_pulse=1.0, pulse_window=(0, T))
    y_BA = get_readout(entity, readout, q_BA)
    
    # Metrics
    dist = torch.norm(q_AB - q_BA).item()
    norm_sum = torch.norm(q_AB).item() + torch.norm(q_BA).item() + 1e-8
    dist_norm = dist / norm_sum
    
    sem_diff = abs(y_AB - y_BA)
    
    print(f"  > Path XOR->AND: y={y_AB:.4f}")
    print(f"  > Path AND->XOR: y={y_BA:.4f}")
    print(f"  > State Divergence (Raw): {dist:.4f}")
    print(f"  > State Divergence (Norm): {dist_norm:.4f}")
    print(f"  > Semantic Diff: {sem_diff:.4f}")
    
    # Dual Criteria
    if sem_diff > 0.1:
        print("  >> RESULT: Semantically Non-Abelian (Logic Output differs).")
    elif dist_norm > 0.01:
        print("  >> RESULT: Weakly Non-Abelian (State differs, Logic same).")
    else:
        print("  >> RESULT: Abelian (Order and Logic Irrelevant).")
        
    # Baseline Check: Self-Drift (vs Numerical Noise)
    # Compare "Running XOR for T" vs "Running XOR for 2T" normalized by scale?
    # Actually, critique asked for "Repeat Same Path".
    # Since deterministic, repeat is 0.
    # Let's measure "Drift from Attractor" by comparing Step T vs Step 2T in same context.
    # If system is settled, Drift should be small.
    # If Commutator Diff > Drift, then Switching matters more than Time.
    q_drift, _ = run_stage(entity, q_A1, ctx_xor, steps=T, u_pulse=1.0, pulse_window=(0, T))
    # q_A1 is T steps. q_drift is 2T steps.
    drift_val = torch.norm(q_A1 - q_drift).item()
    norm_drift = drift_val / (torch.norm(q_A1).item() + torch.norm(q_drift).item() + 1e-8)
    print(f"  > Baseline Drift (XOR T->2T): {norm_drift:.4f}")
    
    if dist_norm > norm_drift:
         print("  >> VALIDITY: Commutator Effect > Time Drift.")
    else:
         print("  >> WARNING: Commutator Effect <= Time Drift (Could be just settling time).")

    # 3. Loop Test
    print("\nTest 2: Holonomy Loop (XOR -> AND -> XOR | u=1)")
    q_1 = q_A1 # XOR result
    y_1 = get_readout(entity, readout, q_1)
    
    q_mid, _ = run_stage(entity, q_1, ctx_and, steps=T, u_pulse=1.0, pulse_window=(0, T))
    q_2, _ = run_stage(entity, q_mid, ctx_xor, steps=T, u_pulse=1.0, pulse_window=(0, T))
    y_2 = get_readout(entity, readout, q_2)
    
    loop_dist = torch.norm(q_1 - q_2).item()
    loop_norm = loop_dist / (torch.norm(q_1).item() + torch.norm(q_2).item() + 1e-8)
    sem_loop = abs(y_1 - y_2)
    
    print(f"  > Start State (XOR): y={y_1:.4f}")
    print(f"  > End State (XOR):   y={y_2:.4f}")
    print(f"  > Loop Error (Norm): {loop_norm:.4f}")
    print(f"  > Semantic Error:    {sem_loop:.4f}")
    
    if loop_norm < 0.1:
         print("  >> RESULT: Consistent/Elastic.")
    else:
         print("  >> RESULT: Hysteresis/Plastic/Broken.")

if __name__ == "__main__":
    main()

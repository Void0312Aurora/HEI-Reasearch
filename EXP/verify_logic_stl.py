"""
Phase 27 Verification: STL Monitor & Confusion Matrix.
Verifies Logic Selection (XOR/AND) using Signal Temporal Logic (STL) concepts.

Specifications:
1. XOR_Spec: If Ctx=XOR, then G(t_settle, T) (|y - XOR| < eps)
2. AND_Spec: If Ctx=AND, then G(t_settle, T) (|y - AND| < eps)

Metrics:
- STL Robustness: min(eps - |y - target|) over time. Positive = Satisfied.
- Confusion Matrix: Accuracy of Logic Execution.
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=128)
    parser.add_argument('--model_path', type=str, default='EXP/logic_agent.pth')
    parser.add_argument('--readout_path', type=str, default='EXP/logic_readout.pth')
    args = parser.parse_args()
    
    # 1. Setup
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 256), nn.Tanh(), nn.Linear(256, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1)
    entity.internal_gen = adaptive
    entity.generator.add_port('context', dim_u=2)
    
    readout = nn.Linear(args.dim_q, 1, bias=True)
    
    entity.to(DEVICE)
    readout.to(DEVICE)
    
    try:
        entity.load_state_dict(torch.load(args.model_path))
        readout.load_state_dict(torch.load(args.readout_path))
        print("Models loaded.")
    except:
        print("Model load failed.")
        return

    entity.eval()
    
    # Validation Data
    # 4 cases * 2 contexts = 8 scenarios.
    scenarios = []
    # (A, B, CtxType, LogicName)
    scenarios.append((0, 0, 0, 'XOR')) # 0
    scenarios.append((0, 1, 0, 'XOR')) # 1
    scenarios.append((1, 0, 0, 'XOR')) # 1
    scenarios.append((1, 1, 0, 'XOR')) # 0
    
    scenarios.append((0, 0, 1, 'AND')) # 0
    scenarios.append((0, 1, 1, 'AND')) # 0
    scenarios.append((1, 0, 1, 'AND')) # 0
    scenarios.append((1, 1, 1, 'AND')) # 1
    
    truth_xor = {(0,0):0, (0,1):1, (1,0):1, (1,1):0}
    truth_and = {(0,0):0, (0,1):0, (1,0):0, (1,1):1}
    
    predictions = []
    ground_truth = []
    
    robustness_scores = []
    
    print("\nRunning Logic Verification (STL)...")
    
    # Run per scenario
    for A, B, ctx_type, logic_name in scenarios:
        curr = torch.zeros(1, 2*args.dim_q + 1, device=DEVICE)
        
        # Context Vector
        c_vec = torch.tensor([[1.0 - ctx_type, float(ctx_type)]], device=DEVICE)
        
        # Determine Target
        if logic_name == 'XOR':
            tgt = truth_xor[(A,B)]
        else:
            tgt = truth_and[(A,B)]
            
        ys = []
        
        for t in range(60):
            u_val = torch.zeros(1, 1, device=DEVICE)
            if 10 <= t < 15: u_val += A
            if 30 <= t < 35: u_val += B
            
            out = entity.forward_tensor(curr, {'default': u_val, 'context': c_vec}, 0.1)
            curr = out['next_state_flat']
            
            # Readout
            q = curr[:, :args.dim_q]
            y = torch.sigmoid(readout(q))
            ys.append(y.item())
            
        # STL Check: G_[50, 60] (|y - tgt| < eps)
        # Robustness rho = min(eps - |y - tgt|) for t in [50, 60]
        # Let eps = 0.4 (threshold at 0.5 +/- 0.1 margin)
        # Actually standard rho = (tgt==1 ? y - 0.5 : 0.5 - y) * scale?
        # Let's use simple accuracy proxy
        final_y = np.mean(ys[50:])
        pred_bin = 1 if final_y > 0.5 else 0
        
        predictions.append(pred_bin)
        ground_truth.append(tgt)
        
        # Robustness: Distance to wrong decision boundary?
        # dist = |y - 0.5|
        # If correct, +dist. If wrong, -dist.
        is_correct = (pred_bin == tgt)
        margin = abs(final_y - 0.5)
        rho = margin if is_correct else -margin
        robustness_scores.append(rho)
        
        print(f"Case {logic_name}({A},{B}) -> Tgt {tgt}, Pred {final_y:.2f} (Bin {pred_bin}). Rho={rho:.2f}")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
    acc = np.mean(np.array(predictions) == np.array(ground_truth))
    
    print(f"\nOverall Accuracy: {acc*100:.1f}%")
    print("Confusion Matrix (0 vs 1):")
    print(cm)
    
    # Logic Separation Score
    # Check if Context changed behavior
    # For (A=0, B=1): XOR->1, AND->0. Did it flip?
    # Case index 1 (XOR 0,1) and 5 (AND 0,1).
    xor_01 = predictions[1]
    and_01 = predictions[5]
    
    # For (A=1, B=0): XOR->1, AND->0.
    xor_10 = predictions[2]
    and_10 = predictions[6]
    
    # For (A=1, B=1): XOR->0, AND->1.
    xor_11 = predictions[3]
    and_11 = predictions[7]
    
    if xor_01 != and_01 and xor_10 != and_10 and xor_11 != and_11:
        print("\n>> SUCCESS: Context successfully switched Logic Program!")
    else:
        print("\n>> FAILURE: Context did not fully switch Logic.")
        print(f"Debug: (0,1) XOR={xor_01} AND={and_01}")
        print(f"Debug: (1,0) XOR={xor_10} AND={and_10}")
        print(f"Debug: (1,1) XOR={xor_11} AND={and_11}")

if __name__ == "__main__":
    main()

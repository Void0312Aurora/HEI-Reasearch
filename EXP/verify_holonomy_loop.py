"""
Phase 28 Verification: Holonomy & Composition.
Tests if the Logic Agent exhibits Non-Abelian geometry (History Dependence).

Experiments:
1. Commutativity: XOR->AND vs AND->XOR.
2. Loop: XOR->AND->XOR.

Metrics:
- State Divergence (Euclidean Distance).
- Readout Divergence.
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Try to import sklearn for PCA, else fallback
try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except ImportError:
    HAS_PCA = False

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_trajectory(entity, context_sequence, steps_per_stage, init_q=None):
    """
    Run a sequence of contexts.
    context_sequence: list of (name, vector) or just vector.
    """
    if init_q is None:
        curr = torch.zeros(1, 2*entity.internal_gen.dim_q + 1, device=DEVICE)
    else:
        curr = init_q.clone()
        
    qs = []
    
    # Input is constant (0, 1) for neutral testing
    # We want to test logic SWITCHING, so keep inputs ambiguous or specific?
    # Let's use (0, 0) -> XOR=0, AND=0. Both same target?
    # No, we want distinct attractors.
    # (1, 0): XOR=1, AND=0. Distinct!
    u_val = torch.tensor([[1.0]], device=DEVICE) # Input A=1?
    # Wait, input port is dim 1?
    # Train logic used dim 1 but time-multiplexed input.
    # We should replicate the "Input Injection" or just provide constant input?
    # In training, input was A (t=10-15) then B (t=30-35).
    # If we provide CONSTANT input (A=1, B=0), effectively u=1?
    # But u is 1D.
    # Let's just provide NO input (u=0) and rely on internal state?
    # No, Logic depends on Input.
    # Let's simulate the standard input pulse (A=1, B=0) REPEATEDLY?
    # Or just hold A=1, B=0 CONSTANTLY?
    # If trained with pulses, constant input might be OOD.
    # But let's try constant u=1 (representing A=1). And assume B=0.
    
    # Actually, in training:
    # if 10<=t<15: u += A
    # if 30<=t<35: u += B
    # So u is 0 most of the time.
    # Logic is implemented via INTEGRATION of pulses.
    # If we want to switch Logic, we must provide pulses again?
    # Or does the state sustain the "Answer"?
    # The Answer is sustained (attractor).
    # So we can just pulse once, then switch context?
    # Yes.
    
    # Pulse Phase (fixed for all paths)
    # T=0-20: Pulse Input (A=1, B=0).
    # T=20+: Switch Contexts.
    
    dt = 0.1
    
    # Init Phase (Pulse)
    # Context is Neutral? Or Start with First Context?
    # Let's start with First Context.
    
    # We will execute the sequence of contexts AFTER the input pulse?
    # Or DURING?
    # Logic Selection usually implies Context is present during processing.
    # So Context A -> Pulse -> Result A.
    # Context B -> Pulse -> Result B.
    # Commutativity:
    # Path 1: Ctx A -> Pulse -> Wait -> Ctx B -> Wait.
    # Path 2: Ctx B -> Pulse -> Wait -> Ctx A -> Wait.
    # But Pulse happens once.
    # If we apply Ctx B AFTER Pulse is gone, does it change the answer?
    # "Reprogramming the Soul".
    # If I already decided "1" (XOR), and I switch to AND, do I change to "0"?
    # That is the question!
    # So: Pulse (A=1,B=0). Ctx=XOR. Result=1.
    # Then Switch Ctx=AND. Does Result become 0?
    
    for stage_idx, ctx_vec in enumerate(context_sequence):
        # Run for steps_per_stage
        for t in range(steps_per_stage):
            # Input Logic (Pulse A=1 at start of Exp, B=0)
            # Only pulse in the Very First Stage?
            global_t = stage_idx * steps_per_stage + t
            
            u_t = torch.zeros(1, 1, device=DEVICE)
            # Pulse A=1 at t=5..10
            if 5 <= global_t < 10:
                u_t += 1.0
            # Pulse B=0 at t=20..25 (Do nothing)
            
            ctx_tensor = ctx_vec.to(DEVICE)
            out = entity.forward_tensor(curr, {'default': u_t, 'context': ctx_tensor}, dt)
            curr = out['next_state_flat']
            qs.append(curr.detach().cpu().numpy().flatten())
            
    return np.array(qs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='EXP/logic_agent.pth')
    parser.add_argument('--dim_q', type=int, default=128)
    args = parser.parse_args()
    
    # Load Entity
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 256), nn.Tanh(), nn.Linear(256, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1)
    entity.internal_gen = adaptive
    entity.generator.add_port('context', dim_u=2)
    entity.to(DEVICE)
    try:
        entity.load_state_dict(torch.load(args.model_path))
        print("Model loaded.")
    except:
        print("Model load failed.")
        return
        
    entity.eval()
    
    # Define Contexts
    # XOR: [1, 0], AND: [0, 1]
    ctx_xor = torch.tensor([[1.0, 0.0]])
    ctx_and = torch.tensor([[0.0, 1.0]])
    
    T = 60 # Steps per stage (enough to settle)
    
    print("\nExperiment 1: Commutativity (XOR->AND vs AND->XOR)")
    # Path 1: XOR -> AND
    # Note: Input Pulse happens in Stage 1. Stage 2 is just Context Switch.
    path1_qs = run_trajectory(entity, [ctx_xor, ctx_and], T)
    
    # Path 2: AND -> XOR
    path2_qs = run_trajectory(entity, [ctx_and, ctx_xor], T)
    
    # Compare Final States (End of Stage 2)
    q1_final = path1_qs[-1]
    q2_final = path2_qs[-1]
    
    dist_comm = np.linalg.norm(q1_final - q2_final)
    print(f">> State Divergence (Commutativity): {dist_comm:.4f}")
    
    # Interpretation
    # Case (1,0): XOR=1, AND=0.
    # Path 1: Calc XOR(1,0)->1. Switch AND -> Should become 0?
    # Path 2: Calc AND(1,0)->0. Switch XOR -> Should become 1?
    # If they successfully switch, then q1 (->0) and q2 (->1) should be DIFFERENT.
    # So Non-Commutativity (High Distance) implies Responsiveness!
    # If Distance = 0, it means Context Switch DID NOTHING (stuck in history).
    if dist_comm > 1.0:
        print(">> Result: Non-Abelian (History Matters/Responsive). Excellent.")
    else:
        print(">> Result: Abelian (Commutative/Stuck). Warning.")
        
    print("\nExperiment 2: Loop (XOR -> AND -> XOR)")
    # Path: XOR (Pulse) -> AND -> XOR
    path_loop_qs = run_trajectory(entity, [ctx_xor, ctx_and, ctx_xor], T)
    
    q_start = path_loop_qs[T-1] # End of Stage 1 (XOR)
    q_end = path_loop_qs[-1]    # End of Stage 3 (XOR)
    
    dist_loop = np.linalg.norm(q_start - q_end)
    print(f">> Holonomy Error (Loop): {dist_loop:.4f}")
    
    # Interpretation
    # We want the state to RETURN to the XOR result.
    # So Dist should be SMALL?
    # If Dist is small, we have "Consistency".
    if dist_loop < 2.0:
        print(">> Result: Consistent Loop (Returns to Attractor).")
    else:
        print(">> Result: Hysteresis/Drift (Did not return).")

    # PCA Plot
    if HAS_PCA:
        all_qs = np.concatenate([path1_qs, path2_qs, path_loop_qs], axis=0)
        pca = PCA(n_components=2)
        pca.fit(all_qs)
        
        p1 = pca.transform(path1_qs)
        p2 = pca.transform(path2_qs)
        
        plt.figure(figsize=(8,6))
        plt.plot(p1[:,0], p1[:,1], 'b-', label='XOR -> AND')
        plt.plot(p2[:,0], p2[:,1], 'r-', label='AND -> XOR')
        
        # Markers for start/switch/end
        plt.plot(p1[0,0], p1[0,1], 'bo', markersize=10, label='Start')
        plt.plot(p1[T,0], p1[T,1], 'bx', markersize=10, label='Switch')
        plt.plot(p1[-1,0], p1[-1,1], 'bs', markersize=10, label='End')
        
        plt.plot(p2[T,0], p2[T,1], 'rx', markersize=10)
        plt.plot(p2[-1,0], p2[-1,1], 'rs', markersize=10)
        
        plt.title(f"Phase 28: Logic Commutativity (Div={dist_comm:.2f})")
        plt.legend()
        plt.savefig('EXP/holonomy_commutativity.png')
        print("Saved EXP/holonomy_commutativity.png")

if __name__ == "__main__":
    main()

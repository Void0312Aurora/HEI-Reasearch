"""
Phase 27B: Random Switch Setpoint Control (Verification).
Objective: Disprove "Timer Shortcut" hypothesis.
Method:
1. Load Phase 26 Agent (Trained with Random Switching).
2. Test A: Context switches at random times. Measure success.
3. Test B: Context is randomized noise. Measure failure.
4. Compare Mean Squared Error.
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
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--model_path', type=str, default='EXP/context_agent_random.pth')
    args = parser.parse_args()
    
    # 1. Setup Entity
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1)
    entity.internal_gen = adaptive
    entity.generator.add_port('context', dim_u=1)
    entity.to(DEVICE)
    
    # Load Weights
    try:
        entity.load_state_dict(torch.load(args.model_path))
        print(f"Loaded {args.model_path}")
    except:
        print("Failed to load model. Verification aborted.")
        return

    entity.eval()
    
    # Test A: Coherent Random Switching
    print("\nRunning Test A: Coherent Random Switching...")
    mse_a = run_test(entity, args, coherent=True)
    print(f">> MSE (Coherent): {mse_a:.4f}")
    
    # Test B: Scrambled Context
    print("\nRunning Test B: Scrambled Context (Anti-Proof)...")
    mse_b = run_test(entity, args, coherent=False)
    print(f">> MSE (Scrambled): {mse_b:.4f}")
    
    # Conclusion
    if mse_a < 0.5 and mse_b > 1.0:
        print("\n>> SUCCESS: Agent follows Context, not Timer.")
    else:
        print("\n>> FAILURE: Agent mechanism unclear.")
        
def run_test(entity, args, coherent=True):
    # Run 100 steps
    x_env = torch.zeros(1, 1, device=DEVICE)
    curr = torch.zeros(1, 2*args.dim_q + 1, device=DEVICE)
    v_drift = 0.5
    dt = 0.1
    
    total_err = 0.0
    
    # Generate Schedule
    # Random switch times
    # Just generic Poisson-like switching
    current_ctx = 1.0
    switch_prob = 0.05
    
    xs = []
    tgs = []
    actions = []
    
    for t in range(100):
        # Update Target/Context
        if coherent:
            if torch.rand(1).item() < switch_prob:
                current_ctx *= -1.0 # Flip
            target = current_ctx
            ctx_val = current_ctx
        else:
            # Scrambled: Target is constant +1, but Context is random noise?
            # Or Context is White Noise?
            # Let's say we WANT to maintain target, but Context is Random.
            # If Context drives potential, then Random Context -> Random Attractor -> High Error.
            # We compare against a Target derived from "What the context WAS in Test A".
            # Actually, simplest Anti-Proof:
            # Target = +1. Context = Random [-1, 1].
            # Agent should flail.
            target = 1.0 
            ctx_val = (torch.rand(1).item() * 2.0) - 1.0
            
        ctx_tensor = torch.tensor([[ctx_val]], device=DEVICE)
        
        # Dynamics
        u_obs = x_env.clone()
        out = entity.forward_tensor(curr, {'default': u_obs, 'context': ctx_tensor}, dt)
        curr = out['next_state_flat']
        a_t = entity.get_action(curr, 'default')
        x_env = x_env + v_drift + a_t
        
        if t % 10 == 0:
            print(f"t={t}, x={x_env.item():.2f}, tgt={target:.2f}, a={a_t.item():.2f}, ctx={ctx_val:.2f}")

        err = (x_env.item() - target)**2
        total_err += err
        
        xs.append(x_env.item())
        tgs.append(target)
        actions.append(a_t.item())
        
    # Plot last
    if coherent:
        plt.figure()
        plt.plot(xs, label='x')
        plt.plot(tgs, '--', label='Target/Ctx')
        plt.title('Test A: Coherent Switching')
        plt.savefig('EXP/verify_A_coherent.png')
    else:
        plt.figure()
        plt.plot(xs, label='x')
        plt.plot(tgs, '--', label='Target')
        # plt.plot(ctx, ...) - Context is noise
        plt.title('Test B: Scrambled Context')
        plt.savefig('EXP/verify_B_scrambled.png')
        
    return total_err / 100.0

if __name__ == "__main__":
    main()

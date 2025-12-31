"""
Phase 25.2: Active Inference (Homeostasis).
Demonstrates the system using 'Active Action' to maintain sensory stability.
Task: 1D Drifting Oscillator.
Environment:
    x_{t+1} = x_t + v_drift + a_t
    u_t = x_t
Goal: Maintain u_t near 0.
Agent:
    a_t = Agent(q_t)
    J = sum(u_t^2) + EneryCost
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
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    # 1. Setup Entity
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1) # dim_u = 1
    entity.internal_gen = adaptive
    entity.to(DEVICE)
    
    # Optimizer
    opt = optim.Adam(entity.parameters(), lr=1e-3)
    
    # Drift Velocity
    v_drift = 0.5
    
    print(f"Task: Counteract Drift v={v_drift}. Target u=0.")
    print(f"Objective: Minimize Surprise (u^2) + 0.1 * Internal Energy (H)")
    
    history_u_mean = []
    
    for ep in range(200):
        # Initial Env State
        x_env = torch.zeros(args.batch_size, 1, device=DEVICE)
        
        # Initial Internal State
        curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)
        
        total_surprise = torch.tensor(0.0, device=DEVICE)
        
        # Unroll loop
        for t in range(20):
            # 1. Agent Perceives u_t = x_env
            u_t = x_env.clone()
            
            # 2. Update Internal State q_{t+1} given u_t
            # We assume input happens 'during' the step or 'before'.
            # Let's say perception is 'instant' for this step.
            
            # We need to construct u_dict for forward_tensor
            # u is (Batch, Dim)
            # 2. Update Internal State q_{t+1} given u_t
            # We assume input happens 'during' the step or 'before'.
            # u_t is (Batch, 1). forward_tensor expects u_dict values to be (Batch, Dim).
            
            out = entity.forward_tensor(curr, {'default': u_t}, args.dt)
            curr = out['next_state_flat']
            
            # 3. Agent Acts a_t = Policy(q_{t+1})
            # We use our new get_action header
            a_t = entity.get_action(curr, 'default') # (Batch, 1)
            
            # 4. Env Evolves
            # x_{t+1} = x_t + v + a_t
            x_env = x_env + v_drift + a_t
            
            # Accumulate Surprise (Free Energy proxy)
            # F = u^2 (Accuracy) + lambda * H (Complexity/Prior)
            sur = (x_env**2).sum()
            
            # Retrieve H from 'out'? forward_tensor returns 'next_state_flat'.
            # We need H. H is computed inside run_step.
            # But UnifiedGeometricEntity.forward_tensor doesn't return H in the dict unless we modified it.
            # Let's check entity_v4.py forward_tensor return.
            # It returns {'next_state_flat': ..., 'H_val': ...} (Wait, check file).
            # The file I viewed earlier (entity_v4.py) lines 108-110:
            # return {'next_state_flat': next_flat}
            # It does NOT return H.
            # So I need to compute H here or modify entity.
            # Computing H is cheap: entity.internal_gen(s) + interaction.
            # Or just use q magnitude as proxy? No, explicit H is better.
            
            # Calculate H
            from he_core.state import ContactState
            s_next_obj = ContactState(args.dim_q, args.batch_size, DEVICE, curr)
            H_val = entity.internal_gen(s_next_obj).mean()
            
            # Check for NaN locally
            if torch.isnan(sur) or torch.isnan(H_val):
                 print(f"NaN at t={t}: Sur={sur.item()}, H={H_val.item()}")
                 break
            
            total_surprise = total_surprise + sur + 1e-4 * H_val
            
        # Backprop through time
        loss = total_surprise / (20 * args.batch_size)
    
        if torch.isnan(loss):
             print(f"Loss NaN. Sur_last={sur.item()}, H_last={H_val.item()}")
             opt.zero_grad()
        elif not loss.requires_grad:
             print("Loss has no grad (probably early break).")
             opt.zero_grad()
        else:
             opt.zero_grad()
             loss.backward()
             # Clip grad
             nn.utils.clip_grad_norm_(entity.parameters(), 0.5)
             opt.step()
        
        history_u_mean.append(loss.item())
        if ep % 20 == 0:
            print(f"  Ep {ep}: Mean Surprise (u^2) = {loss.item():.4f}")
            
    # Verification
    # Check if a_t converges to -v_drift
    print("\nVerifying Action Magnitude...")
    entity.eval()
    
    # Save Model
    torch.save(entity.state_dict(), 'EXP/active_agent.pth')
    
    curr = torch.zeros(128, 2*args.dim_q + 1, device=DEVICE)
    x_env = torch.zeros(128, 1, device=DEVICE)
    actions = []
    us = []
    
    # Dynamics require grad even in eval
    with torch.enable_grad():
        for t in range(50):
            u_t = x_env.clone()
            out = entity.forward_tensor(curr, {'default': u_t}, args.dt)
            curr = out['next_state_flat']
            a_t = entity.get_action(curr, 'default')
            x_env = x_env + v_drift + a_t
            
            actions.append(a_t.detach().mean().item())
            us.append(x_env.detach().mean().item())
            
    mean_action = sum(actions[-10:]) / 10.0
    print(f"Final Action: {mean_action:.4f} (Target: {-v_drift})")
    print(f"Final u: {us[-1]:.4f}")
    
    if abs(mean_action + v_drift) < 0.1:
        print(">> SUCCESS: Agent learned Active Homeostasis.")
    else:
        print(">> FAILURE: Agent failed to cancel drift.")
        
    plt.plot(us, label='Sensory u')
    plt.plot(actions, label='Action a')
    plt.legend()
    plt.title("Active Inference: Homeostasis")
    plt.savefig('EXP/active_inference.png')

if __name__ == "__main__":
    main()

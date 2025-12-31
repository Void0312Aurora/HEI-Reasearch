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
    
    history_u_mean = []
    
    for ep in range(200):
        # Initial Env State
        x_env = torch.zeros(args.batch_size if hasattr(args, 'batch_size') else 128, 1, device=DEVICE)
        
        # Initial Internal State
        curr = torch.zeros(128, 2*args.dim_q + 1, device=DEVICE)
        
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
            # F = 0.5 * u^2 (Precision=1)
            sur = (x_env**2).sum()
            total_surprise += sur
            
        # Backprop through time
        loss = total_surprise / (20 * 128)
        if torch.isnan(loss):
             print("Loss is NaN. Skipping step.")
             opt.zero_grad()
        else:
             opt.zero_grad()
             loss.backward()
             # Clip grad?
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

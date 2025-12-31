"""
Phase 26: Contextual Homeostasis (Hierarchical Potential).
Task: Agent must maintain x = +1 or x = -1 based on Context Signal c (+1/-1).
Demonstrates: L3 Potential Shaping (Context -> Attractor).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    
    # 1. Setup Entity
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1) # 'default' port
    entity.internal_gen = adaptive
    
    # ADD CONTEXT PORT
    # Context is 1D signal (+1 or -1)
    # add_port(name, dim_u)
    entity.generator.add_port('context', dim_u=1)
    # Init context weights small
    with torch.no_grad():
        entity.generator.ports['context'].W_stack.data.normal_(0, 0.01)
    
    entity.to(DEVICE)
    
    # Optimizer
    opt = optim.Adam(entity.parameters(), lr=1e-3)
    
    # Drift (Make it hard)
    v_drift = 0.5
    
    print(f"Task: Contextual Switching (Target +/- 1). Drift={v_drift}.")
    
    for ep in range(args.epochs):
        # Initial Env State
        x_env = torch.zeros(args.batch_size, 1, device=DEVICE)
        
        # Context Schedule: 
        # Half batch gets +1, Half batch gets -1?
        # Or switch over time? 
        # Let's do Switch over Time to test dynamics response.
        # T=0..20: Target +1. T=20..40: Target -1.
        
        # Initial Internal State
        curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)
        
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        # Unroll loop (40 steps, switch at 20)
        for t in range(40):
            # Define Context / Target
            if t < 20:
                target = 1.0
                ctx_val = 1.0
            else:
                target = -1.0
                ctx_val = -1.0
                
            ctx_tensor = torch.full((args.batch_size, 1), ctx_val, device=DEVICE)
            
            # 1. Agent Perceives
            u_obs = x_env.clone()
            
            # 2. Update Internal State (with Context)
            # u_dict key matches 'add_port' name
            out = entity.forward_tensor(curr, {'default': u_obs, 'context': ctx_tensor}, args.dt)
            curr = out['next_state_flat']
            
            # 3. Agent Acts (from default port)
            a_t = entity.get_action(curr, 'default') 
            
            # 4. Env Evolves
            x_env = x_env + v_drift + a_t
            
            # Loss: Task Error + Energy Cost
            err = ((x_env - target)**2).sum()
            
            # Energy Cost
            from he_core.state import ContactState
            s_obj = ContactState(args.dim_q, args.batch_size, DEVICE, curr)
            H = entity.internal_gen(s_obj).mean()
            
            # Check NaN
            if torch.isnan(x_env).any() or torch.isnan(H):
                print(f"NaN at t={t}")
                break
            
            # We want to minimize Error. 
            # Energy Cost ensures minimal effort/complexity.
            total_loss = total_loss + err + 1e-4 * H
            
        # Backprop
        loss = total_loss / (40 * args.batch_size)
        
        if torch.isnan(loss):
            print("Loss NaN")
            opt.zero_grad()
        elif not loss.requires_grad:
             print("Loss no grad")
             opt.zero_grad()
        else:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(entity.parameters(), 1.0)
            opt.step()
            
        if ep % 20 == 0:
            print(f"Ep {ep}: Loss {loss.item():.4f} (Err+E)")
            
    # Verification: Save Plot
    print("Verifying Switching...")
    entity.eval()
    with torch.no_grad(): # Use no_grad for plotting inference
        x_env = torch.zeros(1, 1, device=DEVICE)
        curr = torch.zeros(1, 2*args.dim_q + 1, device=DEVICE)
        
        xs = []
        targets = []
        
        for t in range(60):
            # Switch every 20 steps
            if (t // 20) % 2 == 0:
                target = 1.0
                ctx_val = 1.0
            else:
                target = -1.0
                ctx_val = -1.0
                
            ctx_tensor = torch.tensor([[ctx_val]], device=DEVICE)
            
            u_obs = x_env.clone()
            out = entity.forward_tensor(curr, {'default': u_obs, 'context': ctx_tensor}, args.dt)
            curr = out['next_state_flat']
            a_t = entity.get_action(curr, 'default')
            x_env = x_env + v_drift + a_t
            
            xs.append(x_env.item())
            targets.append(target)
            
    plt.plot(xs, label='Agent X')
    plt.plot(targets, linestyle='--', label='Target (Context)')
    plt.title("Phase 26: Contextual Switching")
    plt.savefig('EXP/context_switching.png')
    print("Saved EXP/context_switching.png")

if __name__ == "__main__":
    main()

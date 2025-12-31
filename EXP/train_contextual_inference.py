"""
Phase 26: Contextual Homeostasis (Hierarchical Potential).
Task: Agent must maintain x = +1 or x = -1 based on Context Signal c (+1/-1).
Demonstrates: L3 Potential Shaping (Context -> Attractor).
Includes Phase 27B Random Switching.
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
    parser.add_argument('--random_switch', action='store_true', help="Randomize switching time")
    parser.add_argument('--save_path', type=str, default='EXP/context_agent.pth')
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()
    
    # 1. Setup Entity
    config = {'dim_q': args.dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(args.dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    adaptive = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, 1, True, 1) # 'default' port
    entity.internal_gen = adaptive
    
    # ADD CONTEXT PORT
    entity.generator.add_port('context', dim_u=1)
    
    # Init context weights small
    with torch.no_grad():
        entity.generator.ports['context'].W_stack.data.normal_(0, 0.01)
    
    entity.to(DEVICE)
    
    # Optimizer
    opt = optim.Adam(entity.parameters(), lr=1e-3)
    
    v_drift = 0.5
    
    # Init State outside loop (TBPTT)
    x_env = torch.zeros(args.batch_size, 1, device=DEVICE)
    curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)

    for ep in range(args.epochs):
        # Detach specific variables to truncate graph but keep history
        curr = curr.detach() 
        x_env = x_env.detach()
        
        # Reset x_env occasionally to prevent infinite drift during training?
        # Or let it learn to bound it?
        # If x_env drifts too far, loss becomes huge.
        # But we want it to learn Global Stability.
        # Let's reset every 10 epochs to sample new initial conditions.
        if ep % 10 == 0:
            x_env = torch.zeros(args.batch_size, 1, device=DEVICE)
            curr = torch.zeros(args.batch_size, 2*args.dim_q + 1, device=DEVICE)
            
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        # Determine switch time
        if args.random_switch:
             switch_t = torch.randint(5, 35, (1,)).item()
        else:
             switch_t = 20
        
        # Unroll loop
        for t in range(40):
            # Define Context / Target
            if t < switch_t:
                target_val = 1.0
                ctx_val = 1.0
            else:
                target_val = -1.0
                ctx_val = -1.0
            
            ctx_tensor = torch.full((args.batch_size, 1), ctx_val, device=DEVICE)
            target = target_val
            
            # ... (Dynamics) ...
            u_obs = x_env.clone()
            out = entity.forward_tensor(curr, {'default': u_obs, 'context': ctx_tensor}, args.dt)
            curr = out['next_state_flat']
            a_t = entity.get_action(curr, 'default') 
            x_env = x_env + v_drift + a_t
            
            # Loss
            err = ((x_env - target)**2).sum()
            
            # Check NaN
            if torch.isnan(x_env).any():
                break
                
            # Energy
            from he_core.state import ContactState
            s_obj = ContactState(args.dim_q, args.batch_size, DEVICE, curr)
            H = entity.internal_gen(s_obj).mean()
            
            total_loss = total_loss + err + 1e-4 * H
            
        # Backprop (same) ...
        loss = total_loss / (40 * args.batch_size)
        
        if torch.isnan(loss):
             opt.zero_grad()
        elif not loss.requires_grad:
             opt.zero_grad()
        else:
             opt.zero_grad()
             loss.backward()
             nn.utils.clip_grad_norm_(entity.parameters(), 1.0)
             opt.step()
             
        if ep % 20 == 0:
            print(f"Ep {ep}: Loss {loss.item():.4f}")

    # Save Model
    if args.save_path:
        torch.save(entity.state_dict(), args.save_path)
        print(f"Saved model to {args.save_path}")

    # Verification: Save Plot (Only if not no_plot)
    if not args.no_plot:
        print("Verifying Switching...")
        # ... (Plotting code basically same) ...
        entity.eval()
        
        # Do NOT use no_grad() because Hamiltonian requires autograd for dynamics
        x_env = torch.zeros(1, 1, device=DEVICE)
        curr = torch.zeros(1, 2*args.dim_q + 1, device=DEVICE)
        xs = []
        targets = []
        for t in range(100):
            # 20 step switch
            if (t // 20) % 2 == 0:
                tg = 1.0
                cv = 1.0
            else:
                tg = -1.0
                cv = -1.0
            ctx_t = torch.tensor([[cv]], device=DEVICE)
            u = x_env.clone()
            out = entity.forward_tensor(curr, {'default': u, 'context': ctx_t}, args.dt)
            curr = out['next_state_flat'].detach() # Detach to clear graph
            a = entity.get_action(curr, 'default')
            x_env = x_env + v_drift + a
            xs.append(x_env.item())
            targets.append(tg)
        
        plt.figure()
        plt.plot(xs, label='Agent')
        plt.plot(targets, '--', label='Target')
        plt.title("Phase 26: Contextual Switching (TBPTT Stable)")
        plt.savefig('EXP/context_switching.png') 


if __name__ == "__main__":
    main()

"""
Phase 30.1: Autonomous Adaptation (Active Inference).
The Agent autonomously infers the Context `c` to minimize Task Loss.
Version: Per-Trial Adaptation with Full Input Protocol.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cpu') # Force CPU for stability

class AutonomousAgent(nn.Module):
    def __init__(self, model_path, readout_path, dim_q=128):
        super().__init__()
        config = {'dim_q': dim_q, 'dim_u': 1, 'num_charts': 1, 'learnable_coupling': True}
        self.entity = UnifiedGeometricEntity(config)
        
        net_V = nn.Sequential(nn.Linear(dim_q, 256), nn.Tanh(), nn.Linear(256, 1))
        adaptive = AdaptiveDissipativeGenerator(dim_q, net_V=net_V)
        self.entity.generator = PortCoupledGenerator(adaptive, 1, True, 1)
        self.entity.internal_gen = adaptive
        self.entity.generator.add_port('context', dim_u=2)
        
        try:
            self.entity.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded Entity from {model_path}")
        except Exception as e:
            print(f"Failed to load Entity: {e}")
            
        self.entity.to(DEVICE)
        self.entity.eval()
        for p in self.entity.parameters():
            p.requires_grad = False
            
        self.readout = nn.Linear(dim_q, 1)
        try:
            self.readout.load_state_dict(torch.load(readout_path, map_location=DEVICE))
            print(f"Loaded Readout from {readout_path}")
        except Exception as e:
            print(f"Failed to load Readout: {e}")
        self.readout.to(DEVICE)
        self.readout.eval()
        for p in self.readout.parameters():
            p.requires_grad = False
            
        # Latent Context (Will)
        self.z = nn.Parameter(torch.zeros(1, 2, device=DEVICE), requires_grad=False)
        
    def get_context(self):
        # Use argmax to select one-hot context (Binary Selection)
        idx = torch.argmax(self.z, dim=1)
        one_hot = torch.zeros_like(self.z)
        one_hot[0, idx] = 1.0
        return one_hot
    
    def get_context_soft(self):
        # Soft version for gradient estimation
        return torch.softmax(self.z * 3.0, dim=1)  # Temperature 1/3
    
    def forward_step(self, curr, u_in, dt=0.1):
        c_probs = self.get_context()
        out = self.entity.forward_tensor(curr, {'default': u_in, 'context': c_probs}, dt)
        next_curr = out['next_state_flat']
        
        dim_q = self.entity.internal_gen.dim_q
        q = next_curr[:, :dim_q]
        y_logit = self.readout(q)
        y_pred = torch.sigmoid(y_logit)
        
        return next_curr, y_pred

def run_trial(agent, curr_init, input_A, input_B, dt=0.1):
    """
    Run a full trial with Input Protocol (Matching Phase 27 verify_logic_stl.py):
    - Steps 10-14: Input A
    - Steps 30-34: Input B
    - Steps 50-59: Read output
    Total 60 steps.
    """
    curr = curr_init.clone()
    
    for step in range(60):
        # Input Protocol
        u_val = 0.0
        if 10 <= step < 15:
            u_val = input_A
        if 30 <= step < 35:
            u_val = input_B
            
        u_tensor = torch.tensor([[u_val]], device=DEVICE)
        curr, y_pred = agent.forward_step(curr, u_tensor, dt)
        curr = curr.detach()
    
    return curr, y_pred

def run_experiment(agent, args):
    T_trials = 30  # More trials for flip to occur
    switch_interval = 10  # Switch task every 10 trials
    dt = 0.1
    
    curr = torch.zeros(1, 2*args.dim_q + 1, device=DEVICE)
    current_task = 0  # 0: XOR, 1: AND
    
    # Fixed Inputs for this experiment
    input_A = 1.0
    input_B = 1.0
    # Ground Truth: XOR(1,1)=0, AND(1,1)=1
    
    history = {
        'task': [],
        'prob_xor': [],
        'loss': [],
        'y_pred': [],
        'target': []
    }
    
    print("\n--- Per-Trial Autonomous Adaptation ---")
    print(f"[Config] Inputs: A={input_A}, B={input_B}")
    print(f"[Config] XOR(1,1)=0, AND(1,1)=1\n")
    
    for trial in range(T_trials):
        # Task Switch
        if trial > 0 and trial % switch_interval == 0:
            current_task = 1 - current_task
            print(f"[Trial {trial}] Task Switched to: {'XOR' if current_task==0 else 'AND'}")
        
        # Target
        if current_task == 0:  # XOR
            target = 0.0
        else:  # AND
            target = 1.0
        target_tensor = torch.tensor([[target]], device=DEVICE)
        
        # Reset state each trial (fresh start)
        curr = torch.zeros(1, 2*args.dim_q + 1, device=DEVICE)
        
        # Finite Difference Adaptation
        eps = 1.0
        noise = torch.tensor([[eps, 0.0], [0.0, eps]], device=DEVICE)
        grads = torch.zeros_like(agent.z)
        z_orig = agent.z.clone()
        
        # Base Trial
        _, y_base = run_trial(agent, curr, input_A, input_B, dt)
        l_base = ((y_base - target_tensor)**2).item()
        
        # Perturb z[0] (XOR)
        agent.z.data = z_orig + noise[0].unsqueeze(0)
        _, y_p1 = run_trial(agent, curr, input_A, input_B, dt)
        l_p1 = ((y_p1 - target_tensor)**2).item()
        
        # Perturb z[1] (AND)
        agent.z.data = z_orig + noise[1].unsqueeze(0)
        _, y_p2 = run_trial(agent, curr, input_A, input_B, dt)
        l_p2 = ((y_p2 - target_tensor)**2).item()
        
        # Restore and Compute Gradients
        agent.z.data = z_orig
        grads[0,0] = (l_p1 - l_base) / eps
        grads[0,1] = (l_p2 - l_base) / eps
        
        # Regularization (prevent explosion)
        grads += 0.01 * agent.z.data
        
        # Update
        lr = 10.0  # Higher LR for faster adaptation
        agent.z.data -= lr * grads
        
        # Clip z to prevent explosion
        agent.z.data = torch.clamp(agent.z.data, -2.0, 2.0)
        
        # Execute Real Trial with Updated z
        curr, y_final = run_trial(agent, curr, input_A, input_B, dt)
        curr = curr.detach()
        
        final_loss = ((y_final - target_tensor)**2).item()
        
        # Logging
        with torch.no_grad():
            c_probs = agent.get_context()
            history['task'].append(current_task)
            history['prob_xor'].append(c_probs[0,0].item())
            history['loss'].append(final_loss)
            history['y_pred'].append(y_final.item())
            history['target'].append(target)
        
        print(f"Trial {trial:2d} | Task={'XOR' if current_task==0 else 'AND'} | "
              f"P(XOR)={c_probs[0,0].item():.2f} | "
              f"y={y_final.item():.2f} | tgt={target:.0f} | loss={final_loss:.4f} | "
              f"z=[{agent.z[0,0].item():.2f}, {agent.z[0,1].item():.2f}] | "
              f"grads=[{grads[0,0]:.3f}, {grads[0,1]:.3f}]")
        
        gc.collect()
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(history['prob_xor'], 'b-o', label='P(XOR)')
    plt.plot(1 - np.array(history['task']), 'k--', label='Ground Truth (XOR=1)', alpha=0.5)
    plt.ylabel('P(XOR)')
    plt.title('Context Inference (Active Inference)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(history['y_pred'], 'g-o', label='y_pred')
    plt.plot(history['target'], 'r--', label='target')
    plt.ylabel('Output')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(history['loss'], 'r-o')
    plt.ylabel('Loss')
    plt.xlabel('Trial')
    
    plt.tight_layout()
    plt.savefig('EXP/autonomous_adaptation.png')
    print("\nSaved EXP/autonomous_adaptation.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=128)
    args = parser.parse_args()
    
    agent = AutonomousAgent('EXP/logic_agent.pth', 'EXP/logic_readout.pth', args.dim_q)
    run_experiment(agent, args)

if __name__ == "__main__":
    main()

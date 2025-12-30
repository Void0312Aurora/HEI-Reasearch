"""
Phase 17.1: Loop A - Dynamics Training.

Objective:
Train an unstable network (Loop Gain > 1) to satisfy Small-Gain Condition and Dissipativity
using purely Self-Supervised signals.

Loss Function:
L = lambda_1 * L1_Dissipative + lambda_sg * Square(Max(0, LoopGain - 0.9))

Signals:
- L1_Dissipative: From he_core.losses.
- LoopGain: Differentiable proxy ||G_y_A|| * ||G_y_B|| * wiring_gain.
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json

from he_core.wiring import Edge, WiringDiagram, TwoEntityNetwork
from he_core.losses import SelfSupervisedLosses
from he_core.supervisor import TrainingSupervisor
from EXP.diag.network_gain import estimate_network_gain
import copy

def get_proxy_loop_gain(network):
    """
    Compute differentiable upper bound of loop gain.
    Gain ~ ||G_y_A|| * ||G_u_B|| * edge_gain ...
    
    Assumption: PortInterface used.
    Max Gain of PortInterface <= ||G_u|| * ||G_y|| * (max_contract_gain)
    """
    # Extract entities
    A = network.entities['A']
    B = network.entities['B']
    
    # helper
    def get_entity_gain_bound(ent):
        if not ent.use_port_interface:
            return torch.tensor(1.0) # Unknown
        
        # Interface = G_u @ ... @ G_y
        # Spectral norm is best, but Frobenius is upper bound and cheaper/easier diff
        g_u_norm = torch.norm(ent.interface.G_u.weight)
        g_y_norm = torch.norm(ent.interface.G_y.weight)
        
        # Contract gain
        contract_gain = 1.0
        if ent.interface.use_contract:
            contract_gain = ent.interface.contract.max_gain
            # If tensor, keep graph? currently float
            if isinstance(contract_gain, torch.Tensor):
                pass 
            else:
                contract_gain = float(contract_gain)
                
        return g_u_norm * g_y_norm * contract_gain

    gain_A = get_entity_gain_bound(A)
    gain_B = get_entity_gain_bound(B)
    
    # Wiring Gain (assumed symmetric 2-entity loop)
    # A->B and B->A
    w_ab = 0.0
    w_ba = 0.0
    for edge in network.wiring.edges:
        if edge.source_id == 'A' and edge.target_id == 'B':
            w_ab = edge.gain
        if edge.source_id == 'B' and edge.target_id == 'A':
            w_ba = edge.gain
            
    loop_gain = gain_A * w_ab * gain_B * w_ba
    return loop_gain

def train_dynamics_loop():
    print("=== Phase 17.1: Training Dynamics (Loop A) ===")
    
    # 1. Unstable Config
    config = {
        'dim_q': 2,
        'learnable_coupling': False,
        'num_charts': 1,
        'damping': 0.1,  # Low damping (Unstable tendency)
        'use_port_interface': True,
        'port_contract_method': 'tanh',
        'port_max_gain': 2.0 # High gain
    }
    
    edges = [
        Edge(source_id='A', target_id='B', gain=1.0),
        Edge(source_id='B', target_id='A', gain=1.0)
    ]
    wiring = WiringDiagram(edges)
    network = TwoEntityNetwork(config, config, wiring)
    
    # Optimizer
    # We optimize parameters of Entity A and B
    params = list(network.parameters())
    optimizer = optim.Adam(params, lr=0.05) # Increased LR
    
    # Training Loop
    epochs = 20 # Enough for verification
    steps_per_epoch = 20
    dt = 0.1 # Time step for rollouts

    history = {'loss': [], 'loop_gain_proxy': [], 'real_loop_gain': []}
    
    # ... Simplified Training ...
    
    # Supervisor
    supervisor = TrainingSupervisor(use_small_gain=True, use_trend=False) 
    
    # Checkpoint
    best_loss = float('inf')
    best_state_dict = None
    
    rollback_count = 0
    stop_count = 0
    
    print("Training Dynamics (Gain + Dissipation)...")
    
    # 2. Differentiable Rollout Helper
    def differentiable_step(ent, u_ext, dt):
        # Manually call forward_tensor
        # Need to handle state wrapper
        # ent.state is ContactState. We extract flat, pass to forward_tensor
        out = ent.forward_tensor(ent.state.flat, u_ext, dt)
        ent.state.flat = out['next_state_flat'] # Keep it attached to graph? 
        # assigning to .flat (setter) might detach if not careful.
        # ContactState.flat setter: self._data = value. 
        # If value has grad, _data has grad.
        return out

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Proxy Gain (Structural)
        proxy_gain = get_proxy_loop_gain(network)
        diff = torch.relu(proxy_gain - 0.9)
        loss_gain = diff + 10.0 * diff.pow(2)
        
        # 2. Dissipative (Physical - Kinetic)
        loss_dissipative = 0.0
        for name, ent in network.entities.items():
            # Reset to random state
            ent.reset() # detach?
            # We need valid gradients from parameters to state evolution
            # random init is constant w.r.t parameters.
            
            # Short rollout
            energies = []
            steps = 5
            for _ in range(steps):
                u_zero = torch.zeros(1, ent.dim_u, device=ent.state.device)
                out = differentiable_step(ent, u_zero, dt) # Updates ent.state
                
                # H_val (Total Energy)
                # But Phase 16 established Mechanical Energy (K) as the monotonic signal
                # for the dissipative configuration (Null Potential equivalent).
                # Even with Potential, preventing K explosion is a good proxy for "calmness".
                # Let's use K = 0.5 * p^2
                K = 0.5 * (out['next_state_flat'][0, ent.dim_q:2*ent.dim_q]**2).sum()
                energies.append(K.unsqueeze(0))
                
            energies = torch.cat(energies)
            # Minimize energy increase
            loss_diss = SelfSupervisedLosses.l1_dissipative_loss(energies, margin=0.0)
            loss_dissipative += loss_diss
            
        # Total Loss
        loss = loss_gain + 0.1 * loss_dissipative
        
        loss.backward()
        optimizer.step()
        
        # Tracking
        curr_loss = loss.item()
        history['loss'].append(curr_loss)
        history['loop_gain_proxy'].append(proxy_gain.item())
        
        # Checkpoint (Save Best)
        if curr_loss < best_loss:
            best_loss = curr_loss
            # Save deep copy of state dict
            best_state_dict = copy.deepcopy(network.state_dict())
            
        # Gating Every 10 Epochs
        if epoch % 10 == 0:
            # Metrics for Supervisor
            metrics = {
                'loop_gain': proxy_gain.item(),
                # 'robustness_slope': ... (Requires Monitor)
            }
            gate_res = supervisor.check_gates(metrics)
            action = supervisor.decide_action(gate_res)
            
            print(f"Epoch {epoch}: Loss={curr_loss:.4f}, Gain={metrics['loop_gain']:.4f}, Action={action}")
            
            if action == 'ROLLBACK':
                print("  !!! ROLLBACK TRIGGERED !!!")
                rollback_count += 1
                if best_state_dict is not None:
                    network.load_state_dict(best_state_dict)
                    print("  Restored best model.")
                else:
                    print("  No best model to restore. Continuing...")
            elif action == 'STOP':
                print("  !!! STOP TRIGGERED !!!")
                stop_count += 1
                break
            
    # Verification
    print("\nTraining Done. Verifying Real Gain...")
    network.reset() # Important
    real_gain_data = estimate_network_gain(network, pulse_scale=0.1, observation_steps=20)
    real_gain = real_gain_data['loop_gain_energy']
    print(f"Real Energy Loop Gain: {real_gain:.4f}")
    
    result = {
        'initial_proxy': history['loop_gain_proxy'][0],
        'final_proxy': history['loop_gain_proxy'][-1],
        'real_gain': real_gain,
        'success': real_gain < 1.0,
        'rollback_count': rollback_count,
        'stop_count': stop_count
    }
    
    with open('phase17_supervised_run.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Plot
    plt.plot(history['loop_gain_proxy'], label='Proxy Gain')
    plt.axhline(1.0, color='r', linestyle='--')
    plt.legend()
    plt.title('Loop A Training: Small-Gain Optimization')
    plt.savefig('loopA_training.png')

if __name__ == "__main__":
    train_dynamics_loop()

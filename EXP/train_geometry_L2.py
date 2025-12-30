"""
Phase 17.2: Loop B - Geometry Training.

Objective:
Train the geometric structure (Atlas Transitions) to be consistent (Holonomy ~ 0).
This corresponds to learning a valid global manifold topology from local charts.

Loss Function:
L = l2_holonomy_loss(x, Phi_loop(x))

Setup:
- Entity with 3 Charts (0, 1, 2).
- Loop 0->1->2->0.
- Initialize transitions randomly.
- Train to identity.
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import os

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.losses import SelfSupervisedLosses
from he_core.state import ContactState

def train_geometry_loop():
    print("=== Phase 17.2: Training Geometry (Loop B) ===")
    
    # 1. Config
    config = {
        'dim_q': 2,
        'num_charts': 3,
        'learnable_coupling': False,
        'damping': 1.0, # Irrelevant for static geometry test but needed
    }
    
    entity = UnifiedGeometricEntity(config)
    
    # 2. Setup Cycle 0->1->2->0
    cycle = [0, 1, 2, 0]
    
    # Create transitions
    # Only creating the forward links for the cycle to properly train them
    # Real atlas might have all pairs.
    pairs = [(0, 1), (1, 2), (2, 0)]
    for (i, j) in pairs:
        entity.atlas.add_transition(i, j)
        
    print(f"Cycle: {cycle}")
    
    # Optimizer
    # entity.atlas.transitions is ModuleDict
    params = list(entity.atlas.transitions.parameters())
    optimizer = optim.Adam(params, lr=0.01)
    
    epochs = 200
    batch_size = 32
    
    history = {'loss': [], 'holonomy_error': []}
    
    print("Training Holonomy...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Batch of random states in Chart 0
        q = torch.randn(batch_size, entity.dim_q)
        p = torch.randn(batch_size, entity.dim_q)
        s = torch.zeros(batch_size, 1)
        
        # We need Batched ContactState?
        # ContactState supports batch_size implicitly if tensors are batched.
        # Check ContactState.flat property.
        
        # Manual Flat Construction
        # flat: [q, p, s] dim = 2Q+1 = 5
        flat = torch.cat([q, p, s], dim=1)
        
        # We need to wrap this in ContactState to pass to transition map
        # entity.atlas.transitions["0_1"] expects ContactState
        
        curr_state = ContactState(entity.dim_q, batch_size, device=flat.device, flat_tensor=flat)
        original_flat = flat.clone()
        
        # Traverse Cycle
        valid = True
        for (src, dst) in pairs:
            key = f"{src}_{dst}"
            map_ij = entity.atlas.transitions[key]
            
            new_flat = map_ij(curr_state)
            curr_state = ContactState(entity.dim_q, batch_size, device=flat.device, flat_tensor=new_flat)
            
        final_flat = curr_state.flat
        
        # Loss
        loss = SelfSupervisedLosses.l2_holonomy_loss(original_flat, final_flat)
        
        loss.backward()
        optimizer.step()
        
        # Log
        err = loss.item() # MSE
        history['loss'].append(err)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss (MSE)={err:.6f}")
            
    # Verification
    final_error = history['loss'][-1]
    success = final_error < 1e-4
    print(f"\nFinal Holonomy Error: {final_error:.6f}")
    print(f"Success: {success}")
    
    result = {
        'initial_error': history['loss'][0],
        'final_error': final_error,
        'success': success
    }
    
    with open('phase17_loopB_result.json', 'w') as f:
        json.dump(result, f, indent=2)
        
    # Plot
    plt.plot(history['loss'])
    plt.yscale('log')
    plt.title('Loop B Training: Holonomy Minimization')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('loopB_training.png')

if __name__ == "__main__":
    train_geometry_loop()

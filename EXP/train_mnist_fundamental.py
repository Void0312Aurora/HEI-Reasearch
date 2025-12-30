
"""
Phase 21.1: Fundamental MNIST Training Script.
Clean-slate rewrite based on 'debug_refactor.py' success.
Removes CognitiveAgent stateful wrapper to prevent Graph Pollution.
Implements:
1. Manual RawMNIST Loading (No DataLoader).
2. Direct UnifiedGeometricEntity usage.
3. Explicit Training Loop with Clean State Init per Batch.
4. Steps=10 Long Horizon.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import argparse
import time

# HE Core Imports
from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.adapters import ImagePortAdapter
from he_core.losses import SelfSupervisedLosses

# Datasets
from he_core.datasets import RawMNIST

def train_fundamental():
    # 0. Setup
    # torch.set_num_threads(1) # Removed to see if this causes deadlock
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10, help='Dynamics Horizon')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--drive_scale', type=float, default=1.0)
    args = parser.parse_args()
    
    print(f"=== Phase 21: Fundamental MNIST Adaptive Training ===")
    print(f"Config: Steps={args.steps}, Q={args.dim_q}, Batch={args.batch_size}")
    
    device = torch.device('cpu') # Force CPU for stability first
    
    # 1. Data Loading (Manual)
    print("Loading Data...")
    def mnist_norm(img):
        return (img - 0.1307) / 0.3081
        
    dataset = RawMNIST('./data/MNIST', train=True, transform_func=mnist_norm)
    
    # Load all to RAM
    all_data = []
    all_targets = []
    for i in range(len(dataset)):
        d, t = dataset[i]
        all_data.append(d)
        all_targets.append(t)
    
    data_t = torch.stack(all_data).to(device)
    target_t = torch.tensor(all_targets).to(device)
    num_samples = len(data_t)
    print(f"Data Loaded: {data_t.shape}")
    
    # 2. Model Setup
    # Adapter
    adapter = ImagePortAdapter(in_channels=1, dim_out=32) # u dim = 32
    
    # Entity Config
    config = {
        'dim_q': args.dim_q,
        'dim_u': 32,
        'num_charts': 1,
        'learnable_coupling': True,
        'use_port_interface': False # We act on state directly
    }
    
    entity = UnifiedGeometricEntity(config)
    
    # Inject Adaptive Generator
    # Note: Entity creates PortCoupled(Dissipative) by default.
    # We replace Internal with Adaptive.
    net_V = nn.Sequential(
        nn.Linear(args.dim_q, 128), nn.Tanh(),
        nn.Linear(128, 1)
    )
    # Init net_V to near zero (Flat Landscape)
    for p in net_V.parameters():
        nn.init.normal_(p, mean=0.0, std=1e-5)
        
    adaptive_gen = AdaptiveDissipativeGenerator(args.dim_q, net_V=net_V)
    
    # Init Adapter Small (Weak Drive)
    for p in adapter.parameters():
        nn.init.normal_(p, mean=0.0, std=1e-4)
    
    # Re-wrap in PortCoupled to ensure dimensions match
    entity.generator = PortCoupledGenerator(
        adaptive_gen, 
        dim_u=32, 
        learnable_coupling=True, 
        num_charts=1
    )
    entity.internal_gen = adaptive_gen
    
    # Classifier
    classifier = nn.Linear(args.dim_q, 10)
    
    # Optimizer
    # Collect all params
    params = list(adapter.parameters()) + list(entity.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-4) # Lower LR for stability
    
    # 3. Training Loop
    history = {'acc': [], 'loss': [], 'metrics': []}
    
    for epoch in range(args.epochs):
        start_time = time.time()
        # Shuffle
        perm = torch.randperm(num_samples)
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0
        
        epoch_metrics = {'u': 0.0, 'q': 0.0, 'p': 0.0}
        
        for start_idx in range(0, num_samples, args.batch_size):
            idx = perm[start_idx : start_idx + args.batch_size]
            batch_x = data_t[idx]
            batch_y = target_t[idx]
            curr_batch = len(idx)
            
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            
            # A. Get Drive
            u_ext = adapter.get_drive(batch_x) * args.drive_scale
            
            # B. Initialize State (Clean)
            # q=0, p=0, s=0
            # Shape: (B, 2*dim_q + 1)
            state_flat = torch.zeros(curr_batch, 2*args.dim_q + 1, device=device)
            # If we wanted random init: torch.randn(...) * 0.1
            
            # C. Dynamics Rollout
            energies = []
            
            # Loop over steps
            # Crucial: We pass state_flat through the loop, updating it.
            # No persistent "self.state" object.
            
            # Loop over steps
            # Crucial: We pass state_flat through the loop, updating it.
            # No persistent "self.state" object.
            
            dt = 0.01 # Smaller dt for Euler stability
            current_flat = state_flat
            
            for t in range(args.steps):
                # Entity Forward
                # Checkpointing is handled inside forward_tensor if configured, 
                # but we rely on Clean Graph here.
                out = entity.forward_tensor(current_flat, u_ext, dt)
                current_flat = out['next_state_flat']
                
                # Energy Monitor (Optional, for Loss)
                # Need p from flat.
                # Flat: [q (dim_q) | p (dim_q) | s (1)]
                p_t = current_flat[:, args.dim_q : 2*args.dim_q]
                K_t = 0.5 * (p_t**2).sum(dim=1, keepdim=True)
                energies.append(K_t)
                
            # Final State
            final_flat = current_flat
            final_q = final_flat[:, :args.dim_q]
            final_p = final_flat[:, args.dim_q : 2*args.dim_q]
             
            # D. Loss
            # 1. Classification
            logits = classifier(final_q)
            loss_cls = nn.functional.cross_entropy(logits, batch_y)
            
            # 2. Dissipative
            # Concatenate energies: (B, T)
            energy_stack = torch.cat(energies, dim=1)
            loss_diss = SelfSupervisedLosses.l1_dissipative_loss(energy_stack)
            
            # Total
            loss = loss_cls + loss_diss
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0) # Prevent Explosion
            optimizer.step()
            
            # --- Metrics ---
            acc = (logits.argmax(dim=1) == batch_y).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            batches += 1
            
            with torch.no_grad():
                epoch_metrics['u'] += u_ext.norm(dim=1).mean().item()
                epoch_metrics['q'] += final_q.norm(dim=1).mean().item()
                epoch_metrics['p'] += final_p.norm(dim=1).mean().item()
            
            if batches % 50 == 0:
                 print(f"Ep {epoch} [{batches*args.batch_size}/{num_samples}]: Loss={loss.item():.4f} Acc={acc.item():.2f}")
                 
            if batches >= 200: # Early Stop for Verified Run
                 print("DEBUG: 200 Batches Reached. Training Stable. Exiting.")
                 torch.save(entity.state_dict(), 'phase21_entity.pth') # SAVE HERE
                 return
                
        # Epoch End
        avg_loss = epoch_loss / batches
        avg_acc = epoch_acc / batches
        for k in epoch_metrics: epoch_metrics[k] /= batches
        
        print(f"=== Epoch {epoch} Done ===")
        print(f"Acc: {avg_acc*100:.2f}% | Loss: {avg_loss:.4f}")
        print(f"Metrics: {epoch_metrics}")
        
        history['acc'].append(avg_acc)
        history['loss'].append(avg_loss)
        history['metrics'].append(epoch_metrics)
        
    # Save
    with open('phase21_fundamental_result.json', 'w') as f:
        json.dump(history, f, indent=2)
        
    torch.save(entity.state_dict(), 'phase21_entity.pth')

if __name__ == "__main__":
    train_fundamental()

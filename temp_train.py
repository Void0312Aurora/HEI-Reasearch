"""
Phase 18.3: Dynamics-Based Classification Training (MNIST).

Objective:
Train a Deep Dissipative Generator to classify MNIST digits.
Idea: The image creates a "Drive" (u) that tilts the energy landscape.
The system evolves from q=0 to an Attractor (q_T).
A linear probe maps q_T to Class Logits.

Stability Constraint:
We enforce L1 Dissipativity on the latent trajectory to ensure the "Inference" is a relaxation process.
"""

print("DEBUG: Module Loading")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import json

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.generator import DeepDissipativeGenerator
from he_core.adapters import ImagePortAdapter
from he_core.losses import SelfSupervisedLosses
from he_core.supervisor import TrainingSupervisor

class CognitiveAgent(nn.Module):
    def __init__(self, dim_q: int = 64):
        super().__init__()
        self.dim_q = dim_q
        self.dim_u = 32 # Drive dimension
        
        # 1. Perception (Adapter)
        self.adapter = ImagePortAdapter(in_channels=1, dim_out=self.dim_u)
        
        # 2. Dynamics (Deep Generator)
        # Deep Potential V(q)
        net_V = nn.Sequential(
            nn.Linear(dim_q, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1) # Scalar Potential
        )
        
        # Custom Generator
        self.gen = DeepDissipativeGenerator(dim_q, alpha=0.5, net_V=net_V)
        
        # Config for Entity
        config = {
            'dim_q': dim_q,
            'dim_u': self.dim_u,
            'learnable_coupling': True, # Allow drive to couple to q
            'num_charts': 1,
            'damping': 0.5,
            'use_port_interface': True
        }
        
        self.entity = UnifiedGeometricEntity(config)
        # Inject custom generator
        self.entity.internal_gen = self.gen
        
        # 3. Readout (Classifier)
        # Map final q -> 10 classes
        self.classifier = nn.Linear(dim_q, 10)
        
    def forward(self, img):
        # 1. Encode Image -> Drive
        drive_scale = getattr(self, 'drive_scale', 1.0)
        u_ext = self.adapter.get_drive(img) * drive_scale # (B, dim_u)
        
        # 2. Dynamics Rollout
        # Initialize q=0, p=0
        batch_size = img.shape[0]
        
        # Re-initialize state to match batch size
        from he_core.state import ContactState
        if self.entity.state.batch_size != batch_size:
            self.entity.state = ContactState(self.dim_q, batch_size, device=img.device)
            
        # self.entity.reset() # Removed: hardcodes batch=1
        
        # Force Zero Init for consistent inference
        self.entity.state.q = torch.zeros(batch_size, self.dim_q, device=img.device)
        self.entity.state.p = torch.zeros(batch_size, self.dim_q, device=img.device)
        self.entity.state.s = torch.zeros(batch_size, 1, device=img.device)
        
        # Evolve
        dt = 0.1
        steps = 10 # Short inference time
        energies = []
        
        for _ in range(steps):
            out = self.entity.forward_tensor(self.entity.state.flat, u_ext, dt)
            self.entity.state.flat = out['next_state_flat']
            
            # Track Kinetic Energy for Stability
            # K = 0.5 * p^2
            K = 0.5 * (self.entity.state.p**2).sum(dim=1, keepdim=True)
            energies.append(K)
            
        final_q = self.entity.state.q
        
        # 3. Classify
        logits = self.classifier(final_q)
        
        return logits, torch.cat(energies, dim=1) # (B, Steps)

import argparse

def train_mnist():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'baseline_static', 'ablation_no_diss'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lambda_diss', type=float, default=0.1)
    parser.add_argument('--drive_scale', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    print(f"=== Phase 18.3/4: MNIST Dynamics Training ===")
    print(f"Mode: {args.mode}, Lambda: {args.lambda_diss}, Drive: {args.drive_scale}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    os.makedirs('./data', exist_ok=True)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model
    model = CognitiveAgent(dim_q=64)
    # Inject Drive Scale
    model.drive_scale = args.drive_scale
    model.mode = args.mode # Pass mode to model if needed (for static baseline)
    
    # For Baseline Static, we might need a direct head if structure is different
    # Or we modify CognitiveAgent to handle it.
    if args.mode == 'baseline_static':
        # Modify model to bypass dynamics: Adapter -> Head
        # Adapter out: 32. Classifier in: 64.
        # Need a bridge.
        model.static_bridge = nn.Linear(32, 64)
        model.to(torch.device("cpu")) # CPU for safety or GPU if avail
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Supervisor
    supervisor = TrainingSupervisor(use_small_gain=False, use_trend=True) 
    
    history = {'acc': [], 'loss': [], 'rollback': 0, 'metrics': []}
    
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        
        epoch_metrics = {'u_norm': 0.0, 'q_norm': 0.0, 'p_norm': 0.0, 'k_trend': 0.0}
        steps = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward
            if args.mode == 'baseline_static':
                # Direct Path
                u = model.adapter.get_drive(data)
                q_proxy = torch.tanh(model.static_bridge(u))
                logits = model.classifier(q_proxy)
                loss_diss = torch.tensor(0.0)
            else:
                # Dynamic Path
                logits, energy_seq = model(data)
                
                # Metrics
                with torch.no_grad():
                    epoch_metrics['u_norm'] += model.adapter.get_drive(data).norm(dim=1).mean().item()
                    epoch_metrics['q_norm'] += model.entity.state.q.norm(dim=1).mean().item()
                    epoch_metrics['p_norm'] += model.entity.state.p.norm(dim=1).mean().item()
                    steps += 1
                
                # Dissipative Loss
                if args.mode == 'ablation_no_diss':
                    loss_diss = torch.tensor(0.0)
                else:
                    energies_t = energy_seq.T
                    loss_diss = SelfSupervisedLosses.l1_dissipative_loss(energies_t)
            
            # Classification Loss
            loss_cls = nn.functional.cross_entropy(logits, target)
            
            # Total
            lambda_d = args.lambda_diss if args.mode == 'standard' else 0.0
            loss = loss_cls + lambda_d * loss_diss
            
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.shape[0]
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}]: Loss={loss.item():.4f} (Cls={loss_cls.item():.4f}, Diss={loss_diss.item():.4f})")
                
        acc = 100. * correct / total
        print(f"Epoch {epoch}: Accuracy={acc:.2f}%")
        history['acc'].append(acc)
        
        # Avg Metrics
        if steps > 0:
            for k in epoch_metrics: epoch_metrics[k] /= steps
        history['metrics'].append(epoch_metrics)
        print(f"  Metrics: u={epoch_metrics['u_norm']:.2f}, q={epoch_metrics['q_norm']:.2f}, p={epoch_metrics['p_norm']:.2f}")
        
    print("Training Complete.")
    
    run_name = f"phase18_result_{args.mode}.json"
    with open(run_name, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    print("DEBUG: Script Entry")
    train_mnist()
    print("DEBUG: Script Exit")

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
        u_ext = self.adapter.get_drive(img) # (B, dim_u)
        
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

def train_mnist():
    print("=== Phase 18.3: MNIST Dynamics Training ===")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use FakeData for offline verification
    train_dataset = datasets.FakeData(size=100, image_size=(1, 28, 28), num_classes=10, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model
    model = CognitiveAgent(dim_q=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Supervisor
    supervisor = TrainingSupervisor(use_small_gain=False, use_trend=True) # Check Trend
    
    epochs = 2 # Demo run
    history = {'acc': [], 'loss': [], 'rollback': 0}
    
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            logits, energy_seq = model(data)
            
            # Loss 1: Classification
            loss_cls = nn.functional.cross_entropy(logits, target)
            
            # Loss 2: Dissipative (Stability)
            # Enforce K decay
            # energy_seq: (B, Steps) -> (Steps, B)
            energies_t = energy_seq.T
            loss_diss = SelfSupervisedLosses.l1_dissipative_loss(energies_t)
            
            loss = loss_cls + 0.1 * loss_diss
            
            loss.backward()
            optimizer.step()
            
            # Stats
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.shape[0]
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}]: Loss={loss.item():.4f} (Cls={loss_cls.item():.4f}, Diss={loss_diss.item():.4f})")
                
        acc = 100. * correct / total
        print(f"Epoch {epoch}: Accuracy={acc:.2f}%")
        history['acc'].append(acc)
        
    print("Training Complete.")
    
    # Save Result
    with open('phase18_mnist_result.json', 'w') as f:
        json.dump(history, f)
        
if __name__ == "__main__":
    train_mnist()

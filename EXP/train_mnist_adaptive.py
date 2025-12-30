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
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import os
import json

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.generator import DeepDissipativeGenerator
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
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
        
        # 2. Dynamics (Adaptive Generator)
        # Deep Potential V(q)
        net_V = nn.Sequential(
            nn.Linear(dim_q, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1) # Scalar Potential
        )
        
        # Use Adaptive Generator
        self.gen = AdaptiveDissipativeGenerator(dim_q, net_V=net_V)
        
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
        steps = 2 # Short inference time (Debug: reduced from 10)
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
    parser.add_argument('--mode', type=str, default='adaptive', choices=['adaptive'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--drive_scale', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    print(f"=== Phase 20: MNIST Adaptive Damping Training ===")
    print(f"Mode: {args.mode}, Drive: {args.drive_scale}")
    
    # Custom Transform function (Tensor -> Tensor)
    def mnist_norm(img):
        # inputs are [0,1] float tensors from RawMNIST
        return (img - 0.1307) / 0.3081

    from he_core.datasets import RawMNIST
    
    os.makedirs('./data', exist_ok=True)
    # Force single process loading 
    train_dataset = RawMNIST('./data/MNIST', train=True, transform_func=mnist_norm)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = CognitiveAgent(dim_q=64)
    # Inject Drive Scale
    model.drive_scale = args.drive_scale
    model.mode = args.mode 
    
    if args.mode == 'baseline_static':
        model.static_bridge = nn.Linear(32, 64)
        model.to(torch.device("cpu"))
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Supervisor (For monitoring)
    # We will implement custom gate logic in the loop
    
    history = {'acc': [], 'loss': [], 'gating_events': 0, 'metrics': []}
    
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        
        epoch_metrics = {'u_norm': 0.0, 'q_norm': 0.0, 'p_norm': 0.0, 'k_trend': 0.0}
        steps = 0
        gating_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward
            if args.mode == 'baseline_static':
                u = model.adapter.get_drive(data)
                q_proxy = torch.tanh(model.static_bridge(u))
                logits = model.classifier(q_proxy)
                loss_diss = torch.tensor(0.0)
                energies_t = None
            else:
                # Dynamic Path
                logits, energy_seq = model(data)
                energies_t = energy_seq.T
                
                # Metrics
                with torch.no_grad():
                    epoch_metrics['u_norm'] += model.adapter.get_drive(data).norm(dim=1).mean().item()
                    epoch_metrics['q_norm'] += model.entity.state.q.norm(dim=1).mean().item()
                    epoch_metrics['p_norm'] += model.entity.state.p.norm(dim=1).mean().item()
                    steps += 1
                
                # Dissipative Loss (Computed but conditionally applied)
                loss_diss = SelfSupervisedLosses.l1_dissipative_loss(energies_t)
            
            # Classification Loss
            loss_cls = nn.functional.cross_entropy(logits, target)
            
            # GATED LOGIC
            final_loss = loss_cls
            
            if args.mode == 'gate_standard':
                # Check Gate: Is Energy Exploding?
                # Simple heuristic: If loss_diss (monotonicity violation) is high, clamp it down.
                # Ideally we check Trend, but per-batch trend is noisy.
                # Let's use the raw dissipative loss as the "Violation Metric".
                # If loss_diss > 0 (meaning energy increased), we penalize.
                # If loss_diss < 0 (energy decreased), we let it be (essentially ignored or 0'd by ReLU in loss def).
                
                # Note: l1_dissipative_loss returns: ReLU( (K_next - K_curr) + margin ).mean()
                # So if it's > 0, strict monotonicity is violated.
                
                if loss_diss > 1e-3: # Tolerance threshold (Relaxed from 1e-4)
                    final_loss = loss_cls + args.lambda_penalty * loss_diss
                    gating_count += 1
            
            elif args.mode == 'ablation_free':
                 pass # Never add dissipative loss
            
            final_loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.shape[0]
            
            if batch_idx % 100 == 0:
                d_val = loss_diss.item()
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}]: Loss={final_loss.item():.4f} (Cls={loss_cls.item():.4f}, Diss={d_val:.4f})")
                
        acc = 100. * correct / total
        print(f"Epoch {epoch}: Accuracy={acc:.2f}%, Gating Events={gating_count}")
        history['acc'].append(acc)
        history['gating_events'] += gating_count
        
        # Avg Metrics
        if steps > 0:
            for k in epoch_metrics: epoch_metrics[k] /= steps
        history['metrics'].append(epoch_metrics)
        print(f"  Metrics: u={epoch_metrics['u_norm']:.2f}, q={epoch_metrics['q_norm']:.2f}, p={epoch_metrics['p_norm']:.2f}")
        
    print("Training Complete.")
    
    run_name = f"phase19_result_{args.mode}.json"
    with open(run_name, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    print("DEBUG: Script Entry")
    try:
        train_mnist()
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()
    print("DEBUG: Script Exit")

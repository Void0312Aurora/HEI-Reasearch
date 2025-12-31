import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os

# Path hack for imports
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'HEI'))

from HEI.he_core.entity_v5 import UnifiedGeometricEntityV5 as UnifiedGeometricEntity
from HEI.he_core.vision import SimpleVisionEncoder
from HEI.he_core.data_mnist import get_mnist_loaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_mnist_entity():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=32, help='Internal state dimension')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=3, help='Dynamics steps per visual frame')
    args = parser.parse_args()
    
    print(f"=== Training MNIST Entity (A5 Multi-modal Test) ===")
    print(f"Device: {DEVICE}")
    print(f"Config: {vars(args)}")
    
    # 1. Data
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    
    # 2. Models
    # Vision Encoder: Image -> u (drive)
    vision_enc = SimpleVisionEncoder(dim_out=args.dim_q).to(DEVICE)
    
    # Entity: Dynamics
    config = {
        'dim_q': args.dim_q,
        'dim_u': args.dim_q, # Vision drive matches state dim for direct coupling
        'dim_z': 8,
        'num_charts': 1,
        'learnable_coupling': True, # Enable flexible input mapping
        'use_adaptive_generator': True
    }
    entity = UnifiedGeometricEntity(config).to(DEVICE)
    
    # Readout: State -> Class
    readout = nn.Linear(args.dim_q, 10).to(DEVICE)
    
    # Optimizer
    params = list(vision_enc.parameters()) + list(entity.parameters()) + list(readout.parameters())
    opt = optim.Adam(params, lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. Loop
    for epoch in range(args.epochs):
        entity.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            b_size = imgs.shape[0]
            
            # Reset Entity state for I.I.D. samples (clears history/weights)
            entity.reset(batch_size=b_size)
            
            # A. Perception (A5)
            u_vision = vision_enc(imgs) # (B, dim_q)
            
            # B. Dynamics (L1)
            # Init state (q, p, s) -> 2*dim_q + 1
            s_flat = torch.zeros(b_size, args.dim_q * 2 + 1, device=DEVICE)
            
            # Run dynamics for T steps
            # Drive is constant for the static image "frame"
            for _ in range(args.steps):
                out = entity.forward_tensor(s_flat, {'default': u_vision}, args.dt)
                s_flat = out['next_state_flat']
                
            # C. Readout
            # Extract q from flat state (first dim_q)
            q_final = s_flat[:, :args.dim_q]
            logits = readout(q_final)
            
            # D. Update
            loss_cls = loss_fn(logits, labels)
            
            # A3: Free Energy reg? 
            # F = out['free_energy']
            # loss = loss_cls + 0.01 * F.mean() # Minimize Surprise
            loss = loss_cls # Start simple
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += b_size
            
            if batch_idx % 100 == 0:
                print(f"Ep {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {correct/total:.4f} |u|={u_vision.norm().item():.2f} |q|={q_final.norm().item():.2f}")
                
        print(f">> Epoch {epoch} Done. Avg Loss: {total_loss/len(train_loader):.4f} Acc: {correct/total:.4f}")
        
    print("=== Training Complete ===")

if __name__ == "__main__":
    train_mnist_entity()

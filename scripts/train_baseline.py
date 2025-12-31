import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'HEI'))

from HEI.he_core.vision import SimpleVisionEncoder
from HEI.he_core.data_mnist import get_mnist_loaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_baseline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_q', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--use_fashion', action='store_true', help='Use Fashion-MNIST')
    args = parser.parse_args()
    
    dataset_name = "Fashion-MNIST" if args.use_fashion else "MNIST"
    print(f"=== Training Pure Vision Baseline ({dataset_name}) ===")
    print(f"Device: {DEVICE}")
    print(f"Config: {vars(args)}")
    
    # 1. Data
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size, use_fashion=args.use_fashion)
    
    # 2. Model: Vision + Linear Head (No Entity Dynamics)
    vision_enc = SimpleVisionEncoder(dim_out=args.dim_q).to(DEVICE)
    readout = nn.Sequential(
        nn.Linear(args.dim_q, 10) # 32 -> 10
    ).to(DEVICE)
    
    # Optimizer
    params = list(vision_enc.parameters()) + list(readout.parameters())
    opt = optim.Adam(params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. Loop
    for epoch in range(args.epochs):
        vision_enc.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            b_size = imgs.shape[0]
            
            # Forward: Vision -> Readout (Direct path)
            features = vision_enc(imgs) # (B, dim_q)
            logits = readout(features)
            
            loss = loss_fn(logits, labels)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += b_size
            
            if batch_idx % 100 == 0:
                print(f"Ep {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {correct/total:.4f}")
                
        print(f">> Epoch {epoch} Done. Avg Loss: {total_loss/len(train_loader):.4f} Acc: {correct/total:.4f}")
        
    print("=== Baseline Training Complete ===")

if __name__ == '__main__':
    train_baseline()

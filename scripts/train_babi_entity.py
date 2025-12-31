import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os

# Path hack
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'HEI'))

from HEI.he_core.entity_v5 import UnifiedGeometricEntityV5 as UnifiedGeometricEntity
from HEI.he_core.text import SimpleTextEncoder
from HEI.he_core.data_babi import get_babi_loaders

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_babi():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=1, help='bAbI Task ID (1-20)')
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=5, help='Dynamics steps per semantic frame')
    parser.add_argument('--baseline', action='store_true', help='Run pure encoder baseline')
    args = parser.parse_args()
    
    print(f"=== Training bAbI Entity (Phase 5: Reasoning) ===")
    print(f"Task: {args.task_id} | Mode: {'Baseline' if args.baseline else 'Entity (Steps=' + str(args.steps) + ')'}")
    print(f"Device: {DEVICE}")
    
    # 1. Data
    train_loader, test_loader, tokenizer = get_babi_loaders(task_id=args.task_id, batch_size=args.batch_size)
    vocab_size = len(tokenizer.word2idx)
    print(f"Vocab Size: {vocab_size}")
    
    # 2. Models
    # Text Encoder: Token Seq -> u (drive)
    # We use dim_q as the semantic space dimension
    text_enc = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=64, hidden_dim=args.dim_q).to(DEVICE)
    
    if not args.baseline:
        # Entity: Dynamics
        config = {
            'dim_q': args.dim_q,
            'dim_u': args.dim_q, 
            'dim_z': 16,
            'num_charts': 1,
            'learnable_coupling': False, # Simplification: Direct Coupling <u, q>
            'use_adaptive_generator': True
        }
        entity = UnifiedGeometricEntity(config).to(DEVICE)
    else:
        entity = None
        
    # Readout: State -> Vocab (Answer)
    # Answers are single words from vocab
    readout = nn.Linear(args.dim_q, vocab_size).to(DEVICE)
    
    # Optimizer
    params = list(text_enc.parameters()) + list(readout.parameters())
    if entity:
        params += list(entity.parameters())
        
    opt = optim.Adam(params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. Loop
    for epoch in range(args.epochs):
        text_enc.train()
        if entity: entity.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            b_size = x.shape[0]
            
            # Reset Entity state for I.I.D. samples
            if entity: entity.reset(batch_size=b_size)
            
            # A. Perception (Text Interface)
            u_text = text_enc(x) # (B, dim_q)
            
            # B. Dynamics (L1) or Bypass
            if entity:
                # Init state (q, p, s)
                s_flat = torch.zeros(b_size, args.dim_q * 2 + 1, device=DEVICE)
                
                # Run dynamics for T steps
                # Reasoning "Thought Process"
                for _ in range(args.steps):
                    out = entity.forward_tensor(s_flat, {'default': u_text}, args.dt)
                    s_flat = out['next_state_flat']
                    
                q_final = s_flat[:, :args.dim_q]
            else:
                # Baseline: Direct path
                q_final = u_text
                
            # C. Readout
            logits = readout(q_final)
            
            # D. Update
            loss = loss_fn(logits, y)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += b_size
            
            if batch_idx % 50 == 0:
                print(f"Ep {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {correct/total:.4f} |u|={u_text.norm().item():.2f} |q|={q_final.norm().item():.2f}")
                
        # Test Validation
        # (Simple calc on test set)
        val_acc = evaluate(test_loader, text_enc, entity, readout, args, DEVICE)
        print(f">> Ep {epoch} Done. Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")
        
    print("=== Training Complete ===")

def evaluate(loader, text_enc, entity, readout, args, device):
    text_enc.eval()
    if entity: entity.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            b_size = x.shape[0]
            
            if entity: entity.reset(batch_size=b_size)
            
            u_text = text_enc(x)
            
            if entity:
                s_flat = torch.zeros(b_size, args.dim_q * 2 + 1, device=device)
                for _ in range(args.steps):
                    out = entity.forward_tensor(s_flat, {'default': u_text}, args.dt)
                    s_flat = out['next_state_flat']
                q_final = s_flat[:, :args.dim_q]
            else:
                q_final = u_text
                
            logits = readout(q_final)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += b_size
            
    return correct / total

if __name__ == '__main__':
    train_babi()

"""
Phase 6/7: Closed-Loop bAbI Training.

Changes (Refactored based on temp-04.md):
1.  Integrated Baseline mode (--baseline).
2.  Proper Supervisor integration (Gain Check).
3.  Configurable dt and seed for Phase 7 Matrix.
4.  Removed unused SelfSupervisedLosses claims.
"""

import sys
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure Path
sys.path.append(os.getcwd())

from he_core.entity_v5 import UnifiedGeometricEntityV5 as UnifiedGeometricEntity
from he_core.text import SimpleTextEncoder
from he_core.data_babi import get_babi_loaders
from he_core.supervisor import TrainingSupervisor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_closed_loop():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--stiffness', type=float, default=1.0, help="Harmonic Confinement (Theory-7)")
    parser.add_argument('--q0_scale', type=float, default=0.05, help="Weak Injection for verification")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.005)
    
    # New Args for Phase 7
    parser.add_argument('--baseline', action='store_true', help="Run pure encoder baseline")
    parser.add_argument('--seed', type=int, default=42, help="Random Seed")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step")
    
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"=== Phase 7: bAbI Task {args.task_id} ===")
    print(f"Mode: {'Baseline' if args.baseline else 'Entity (S='+str(args.steps)+', K='+str(args.stiffness)+')'}")
    print(f"Config: dt={args.dt}, Seed={args.seed}, Injection={args.q0_scale}")
    
    # 1. Data
    train_loader, val_loader, tokenizer, num_answers = get_babi_loaders(
        task_id=args.task_id, batch_size=args.batch_size, max_samples=args.max_samples
    )
    vocab_size = len(tokenizer.word2idx)
    
    # 2. Text Interface (Shared)
    text_enc = SimpleTextEncoder(vocab_size, 64, args.dim_q).to(DEVICE)
    readout = nn.Linear(args.dim_q, num_answers).to(DEVICE)
    
    # 3. Entity (Only if not baseline)
    entity = None
    if not args.baseline:
        config = {
            'dim_q': args.dim_q,
            'dim_u': args.dim_q,
            'dim_z': 16,
            'num_charts': 1,
            'learnable_coupling': True,
            'use_adaptive_generator': True,
            'stiffness': args.stiffness, # KEY FIX
            'damping': 0.1
        }
        entity = UnifiedGeometricEntity(config).to(DEVICE)
        
    # 4. Supervisor
    # Use small gain check
    supervisor = TrainingSupervisor(use_small_gain=True, use_trend=False)
    
    # Optimizer
    params = list(text_enc.parameters()) + list(readout.parameters())
    if entity:
        params += list(entity.parameters())
        
    opt = optim.Adam(params, lr=args.lr)
    loss_class = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        text_enc.train()
        if entity: entity.train()
        
        epoch_loss = 0
        total_acc = 0
        total_samples = 0
        total_gain = 0.0
        rollback_count = 0
        
        for batch_idx, (x, y, lengths) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            b_size = x.shape[0]
            
            opt.zero_grad()
            
            # Forward
            u_text = text_enc(x, lengths=lengths)
            
            if args.baseline:
                # Baseline: Direct Readout
                logits = readout(u_text)
                loss = loss_class(logits, y)
                gain_val = 0.0 # No dynamics
                
            else:
                # Entity Mode
                entity.reset(batch_size=b_size)
                
                # Weak Injection
                q0 = u_text.clone() * args.q0_scale
                entity.state.q = q0
                entity.state.p = torch.zeros_like(q0)
                entity.state.s = torch.zeros(b_size, 1, device=DEVICE)
                
                curr_flat = entity.state.flat
                u_dict = {'default': u_text}
                
                # Dynamics Loop
                for t in range(args.steps):
                    out = entity.forward_tensor(curr_flat, u_dict, dt=args.dt)
                    curr_flat = out['next_state_flat']
                
                q_final = curr_flat[:, :args.dim_q]
                logits = readout(q_final)
                loss = loss_class(logits, y)
                
                # Calculate Gain for Supervisor
                # Gain = ||q_final|| / ||u_text||
                with torch.no_grad():
                    norm_u = u_text.norm(dim=1).mean()
                    norm_q = q_final.norm(dim=1).mean()
                    gain_val = (norm_q / (norm_u + 1e-6)).item()

            loss.backward()
            
            # Supervisor Gate
            metrics = {'loop_gain': gain_val} # Proxy for Small Gain
            gate = supervisor.check_gates(metrics)
            action = supervisor.decide_action(gate)
            
            if action == 'ROLLBACK':
                # Skip Update
                opt.zero_grad() 
                rollback_count += 1
            else:
                opt.step()
                
            total_gain += gain_val
            
            epoch_loss += loss.item() * b_size
            acc = (logits.argmax(1) == y).float().sum().item()
            total_acc += acc
            total_samples += b_size
            
        avg_loss = epoch_loss / total_samples
        avg_acc = total_acc / total_samples
        avg_gain = total_gain / (len(train_loader) * b_size)  # Approximate per sample
        # Correction: total_gain is summed per batch? No, gain_val is scalar per batch.
        # gain_val was calculated once per batch.
        # So we should divide by number of batches.
        avg_gain = total_gain / len(train_loader)
        rollback_rate = rollback_count / len(train_loader)
        
        # Validation
        val_acc = evaluate(entity, text_enc, readout, val_loader, args)
        
        if epoch % 5 == 0:
            print(f"Ep {epoch}: Loss {avg_loss:.4f} | Train Acc {avg_acc:.4f} | Val Acc {val_acc:.4f} | Gain {avg_gain:.2f} | Rollback {rollback_rate:.2%}")

    print(f"=== Complete (Mode: {'Baseline' if args.baseline else 'Entity'}) ===")

def evaluate(entity, text_enc, readout, loader, args):
    if entity: entity.eval()
    text_enc.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y, lengths in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            b_size = x.shape[0]
            
            u_text = text_enc(x, lengths=lengths)
            
            if args.baseline:
                logits = readout(u_text)
            else:
                q0 = u_text.clone() * args.q0_scale
                entity.reset(batch_size=b_size)
                entity.state.q = q0
                entity.state.p = torch.zeros_like(q0)
                entity.state.s = torch.zeros(b_size, 1, device=DEVICE)
                
                curr_flat = entity.state.flat
                u_dict = {'default': u_text}
                
                for t in range(args.steps):
                    out = entity.forward_tensor(curr_flat, u_dict, dt=args.dt)
                    curr_flat = out['next_state_flat']
                    
                q_final = curr_flat[:, :args.dim_q]
                logits = readout(q_final)
                
            correct += (logits.argmax(1) == y).sum().item()
            total += b_size
            
    return correct / total

if __name__ == '__main__':
    train_closed_loop()

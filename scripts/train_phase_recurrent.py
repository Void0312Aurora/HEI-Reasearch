
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from he_core.entity import Entity
from he_core.language_interface import LanguagePort, SimpleTokenizer
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.kernel.kernels import PlasticKernel

class SyntheticDataset(torch.utils.data.Dataset):
    """
    Structured Synthetic Dataset for Sanity Check.
    Tasks:
    1. Copy: Input "A B C" -> Target "A B C"
    2. Repeat: Input "A" -> Target "A A A"
    3. Pattern: Input "A B" -> Target "A B A B"
    """
    def __init__(self, vocab_size: int, seq_len: int = 16, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Task: Simple Pattern "A B C A B C..."
            # Choose 3 random tokens
            pattern = torch.randint(10, self.vocab_size, (3,)) # Reserve 0-9 for special
            
            # Repeat pattern to fill seq_len
            seq = pattern.repeat(self.seq_len // 3 + 1)[:self.seq_len]
            data.append(seq)
        return torch.stack(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class RecurrentTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # 1. Init Config
        self.dim_q = args.dim_q
        self.vocab_size = args.vocab_size
        
        # 2. Init Generator (Net V)
        # This is the "Core Commitment" - trainable potential
        self.net_V = nn.Sequential(
            nn.Linear(self.dim_q, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1) # Scalar Potential
        ).to(device)
        
        # 3. Init Adaptive Generator (Dynamics)
        # This encapsulates the H = K + V + Phi logic
        self.generator = AdaptiveDissipativeGenerator(
            dim_q=self.dim_q,
            net_V=self.net_V,
            dim_z=0, # No extra latent z for now
            stiffness=args.stiffness,
            contact_stiffness=1.0, # Lambda
            alpha_max=args.alpha_max,
            hyperbolic_c=args.hyperbolic_c
        ).to(device)
        
        # 4. Init Language Port
        self.port = LanguagePort(
            vocab_size=self.vocab_size,
            dim_q=self.dim_q,
            dim_embed=args.dim_embed,
            num_encoder_layers=2,
            num_decoder_layers=2
        ).to(device)
        
        # 5. Init Optimizers
        # Separate optimizers for Port and Core (Safety)
        self.opt_port = optim.Adam(self.port.parameters(), lr=args.lr_port)
        self.opt_core = optim.Adam(
            list(self.net_V.parameters()) + list(self.generator.net_Alpha.parameters()), 
            lr=args.lr_core
        )
        
    def _init_state(self, batch_size):
        """Initialize state (q, p, s)"""
        # Random init near zero
        q = torch.randn(batch_size, self.dim_q, device=self.device) * 0.1
        p = torch.randn(batch_size, self.dim_q, device=self.device) * 0.1
        s = torch.zeros(batch_size, 1, device=self.device)
        return q, p, s
        
    def train_step(self, batch_tokens):
        """
        Recurrent Training Step (TBPTT)
        input: batch_tokens [B, L]
        """
        B, L = batch_tokens.shape
        chunk_size = 64 # TBPTT chunk size
        
        # Reset Gradients
        self.opt_port.zero_grad()
        if self.opt_core: self.opt_core.zero_grad()
        
        # Init State (Detached at start of batch)
        q, p, s = self._init_state(B)
        
        # Metrics
        total_nll = 0
        total_reg = 0
        total_lyap = 0
        
        # Lists for logging
        q_traj = []
        
        # === Momentary Encoding (Causality Fix) ===
        # u_t = Proj(Embed(x_t)) - NO Context, NO Transformer Leakage
        # We can implement this by manually using port's embedding + projection
        # or adding a helper in Port.
        # Here we manually do it:
        # 1. Embed
        embeds = self.port.encoder.embedding(batch_tokens) # [B, L, D_emb]
        # 2. Pos encode? Maybe not for momentary sensation, but helpful for time-awareness?
        # Theory says "u_t depends on x_t". Pos encoding makes it u_t(x_t, t). acceptable.
        embeds = self.port.encoder.pos_encoder(embeds)
        # 3. Project (Skip Transformer)
        u_seq = self.port.encoder.projection(embeds) # [B, L, D_u]
        
        # === TBPTT Loop ===
        # Split sequence into chunks
        # We predict x_{t+1}. So we iterate t from 0 to L-2.
        # Targets are x_{1...L-1}. (Last token x_L is target of x_{L-1})
        # Actually standard: Input x_{0...L-2}, Target x_{1...L-1}
        
        L_eff = L - 1
        
        loss_accum = 0.0
        
        for t in range(L_eff):
            # 1. TBPTT Detach Check
            if t > 0 and t % chunk_size == 0:
                # Detach State to cut gradient graph
                q = q.detach()
                p = p.detach()
                s = s.detach()
            
            # Input force (Momentary Sensation)
            u_t = u_seq[:, t, :] # [B, dim_u]
            F_ext = u_t
            
            # --- Hamiltonian Dynamics Step ---
            # H = K(p,q) + V(q) + Phi(s)
            # Contact Update:
            # dot_q = dH/dp
            # dot_p = -dH/dq - p * dH/ds
            # dot_s = p * dH/dp - H (Standard Contact Form)
            
            # We need H and its grads.
            # Generator forward gives H.
            
            # Enable grad for physical step
            q.requires_grad_(True)
            p.requires_grad_(True)
            s.requires_grad_(True)
            
            # Compute H
            # We construct a wrapper state
            # Note: Generator typically computes H based on q, p, s
            # But our generator forward() computes H.
            
            # Manual H components for transparency & grad
            # K(p,q)
            lambda_q = (1.0 + self.args.hyperbolic_c * (q**2).sum(-1, keepdim=True))
            metric_inv = lambda_q ** (-2)
            K = 0.5 * metric_inv * (p**2).sum(dim=1, keepdim=True)
            
            # V(q)
            V_val = self.net_V(q)
            # Confining V
            V_conf = 0.5 * self.args.stiffness * (q**2).sum(dim=1, keepdim=True)
            V_tot = V_val + V_conf
            
            # Phi(s)
            alpha_raw = self.generator.net_Alpha(q)
            alpha = self.args.alpha_max * torch.sigmoid(alpha_raw) + 1e-3
            # Contact Potential
            Phi = 0.5 * 1.0 * (s**2) + alpha * s
            
            H = K + V_tot + Phi # [B, 1]
            H_sum = H.sum()
            
            # Gradients
            # create_graph=True needed for BPTT through dynamics
            grads = torch.autograd.grad(H_sum, [q, p, s], create_graph=True)
            dH_dq, dH_dp, dH_ds = grads
            
            # Equations of Motion (Contact)
            # dot_q = dH/dp
            # dot_p = -dH/dq - p * dH/ds
            # dot_s = p * dH/dp - H
            
            dot_q = dH_dp
            dot_p = -dH_dq - p * dH_ds + F_ext # Add external forcing
            dot_s = (p * dH_dp).sum(dim=1, keepdim=True) - H
            
            # Euler Integration
            dt = self.args.dt
            q_next = q + dot_q * dt
            p_next = p + dot_p * dt
            s_next = s + dot_s * dt
            
            # Update
            q, p, s = q_next, p_next, s_next
            
            q_traj.append(q)
            
            # === Per-Step Loss Calculation (Decoder) ===
            # Predict x_{t+1} using z_{t+1}=(q,p,s)
            # Use q as primary readout + p? Let's stick to q for now as per Port interface
            # Target is batch_tokens[:, t+1]
            
            # Decoder
            # q: [B, D] -> Dec -> [B, 1, V]
            # No prev_tokens -> Pure State Readout
            logits = self.port.decoder(q, prev_tokens=None).squeeze(1) # [B, V]
            target_t = batch_tokens[:, t+1] # [B]
            
            l_nll = nn.CrossEntropyLoss()(logits, target_t)
            
            # Regularization
            l_reg = 1e-3 * (q**2).mean() + 1e-4 * (s**2).mean()
            
            loss_step = l_nll + l_reg
            loss_accum += loss_step
            
            total_nll += l_nll.item()
            total_reg += l_reg.item()
            
            # TBPTT Backward at chunk end or sequence end
            if (t + 1) % chunk_size == 0 or t == L_eff - 1:
                # Average per token in this chunk
                loss_mean = loss_accum / (t % chunk_size + 1) 
                
                loss_mean.backward()
                
                # Clip
                nn.utils.clip_grad_norm_(self.port.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.net_V.parameters(), 1.0)
                
                # Step
                self.opt_port.step()
                if self.args.train_core:
                    self.opt_core.step()
                    
                # Zero Grad
                self.opt_port.zero_grad()
                if self.opt_core: self.opt_core.zero_grad()
                
                loss_accum = 0.0
                
                # Detach state for next chunk logic is handled at start of loop
                # But careful: q, p, s are updated IN PLACE in python variables.
                # We need to detach them HERE for the next loop iteration's start?
                # No, the logic `if t > 0 ... detach` at top handles it.
                # BUT `backward()` clears the graph. We MUST detach `q,p,s` immediately after backward
                # to prevent next chunk from trying to backprop into cleared graph.
                q = q.detach()
                p = p.detach()
                s = s.detach()
                
        return {
            "nll": total_nll / L_eff,
            "reg": total_reg / L_eff,
            "q_mean": torch.stack(q_traj).mean().item()
        }

def main():
    parser = argparse.ArgumentParser()
    
    # Model Config
    parser.add_argument("--vocab_size", type=int, default=100) # Small for synthetic
    parser.add_argument("--dim_q", type=int, default=32)
    parser.add_argument("--dim_embed", type=int, default=64)
    
    # Physics Config
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--stiffness", type=float, default=1.0)
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--hyperbolic_c", type=float, default=0.1)
    
    # Training Config
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_port", type=float, default=1e-3)
    parser.add_argument("--lr_core", type=float, default=1e-4)
    parser.add_argument("--train_core", action="store_true", help="Enable core training (Phase 1)")
    
    parser.add_argument("--save_path", type=str, default="checkpoints/recurrent_sanity.pt")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | Train Core: {args.train_core}")
    
    # Data
    dataset = SyntheticDataset(args.vocab_size, seq_len=20, num_samples=2000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Trainer
    trainer = RecurrentTrainer(args, device)
    
    # Loop
    for step in range(args.steps):
        try:
            batch = next(iter(dataloader)).to(device)
        except StopIteration:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            batch = next(iter(dataloader)).to(device)
            
        metrics = trainer.train_step(batch)
        
        if step % 20 == 0:
            print(f"Step {step:04d} | NLL: {metrics['nll']:.4f} | Reg: {metrics['reg']:.4e} | Q: {metrics['q_mean']:.2f}")
            
    # Save
    torch.save(trainer.port.state_dict(), args.save_path)
    print(f"Saved to {args.save_path}")

if __name__ == "__main__":
    main()

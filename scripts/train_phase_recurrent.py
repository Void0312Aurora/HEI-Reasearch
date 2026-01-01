
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
        Recurrent Training Step (BPTT)
        input: batch_tokens [B, L]
        """
        B, L = batch_tokens.shape
        
        # Reset Gradients
        self.opt_port.zero_grad()
        if self.opt_core: self.opt_core.zero_grad()
        
        # Init State
        q, p, s = self._init_state(B)
        
        # Metrics
        total_nll = 0
        total_lyap = 0
        
        # Lists for logging
        q_traj = []
        
        # === Time Loop (Recurrent) ===
        # We process t=0...L-2 to predict t=1...L-1
        # Input: x_t -> Sensation u_t -> Dynamics -> z_{t+1} -> Predict x_{t+1}
        
        # Create attention mask (assuming all valid for synthetic)
        mask = torch.ones_like(batch_tokens)
        
        # Encode all tokens first (or step-by-step? Step-by-step is more "Contact")
        # But Transformer encoder needs context.
        # "Phase 1: Teacher-forced LM" -> We can use causal masked encoder or full encoder?
        # A5 Axiom: Language is Observation.
        # Standard LM: x_{0:t} -> predict x_{t+1}
        # HEI Recurrent:
        #   u_t = Encoder(x_t) (Immediate Sensation) OR Encoder(x_{0:t}) (Contextual Sensation)
        #   Let's use "Encoder(x_{0:t})" via causal masking if we want full context,
        #   OR just Encoder(x_t) if we want "Momentary Readout".
        #   Temp-01 suggests: "u_t = encoder(x_t)" (Momentary) or blanket state.
        #   Let's stick to TokenEncoder doing a full pass with mask, then picking u_t.
        
        # 1. Batch Encode (Parallel for efficiency, but conceptually sequential)
        # Use causal mask to ensure u_t only depends on x_{0:t}??
        # Usually Encoder is Bidirectional (BERT style) or Causal (GPT style).
        # Our TokenEncoder is TransformerEncoder (Bidirectional by default).
        # To be strict causal, we need causal mask.
        
        causal_mask = torch.triu(torch.ones(L, L, device=self.device), diagonal=1).bool()
        # Transformer expects True for masked positions (future)
        # Note: TokenEncoder takes [B, L] mask where 0 is padding. 
        # But nn.TransformerEncoderLayer src_mask is different.
        # Our TokenEncoder wrapper doesn't expose src_mask for causality easily.
        # FIX: For V1, let's assume "u_t" is derived from x_t independent of context 
        # (purely momentary input driving the recurrent core), OR accept bidirectional context.
        # Use Encoder.forward directly to get sequence [B, L, dim_u]
        # LanguagePort.forward returns pooled u by default.
        u_seq = self.port.encoder.forward(batch_tokens, attention_mask=mask) # [B, L, dim_u]
        
        state_trajectory = []
        
        # 2. Dynamics Integration Loop
        dt = self.args.dt
        
        for t in range(L - 1):
            # Input force (Contextualized Sensation)
            u_t = u_seq[:, t, :] # [B, dim_u]
            
            # Map u_t to Force dimension? 
            # If dim_u == dim_q, add directly to p? Or use as guidance?
            # HEI Core usually treats u as "External input".
            # For AdaptiveGenerator, we need to inject it.
            # Minimal coupling: Force +/- k * (u_t - q) ? (Attractor)
            # Or just Additive Force: F_ext = u_t
            
            F_ext = u_t
            
            # Dynamics Step (RK4 or Euler)
            # Using symplectic Euler for simplicity in V1 script, or call generator logic
            # Generator.forward gives H. We need gradients of H.
            
            # --- Hamiltonian Dynamics Step ---
            # H(q, p, s)
            # dot_q = dH/dp
            # dot_p = -dH/dq + F_ext
            # dot_s = ...
            
            # Enable grad for physical step
            # We manualy implement symplectic step here to ensure graph connectivity
            
            # 1. q_half = q + 0.5 * dt * dH/dp
            # H depends on p via K(p) = 0.5 * lam^-2 * p^2
            # dH/dp = lam^-2 * p
            lam_q = (1.0 + self.args.hyperbolic_c * (q**2).sum(-1, keepdim=True))
            # approximate lam_q for metric
            metric_inv = 1.0 # Simplify for V1 start if c=0, else use lambda
            
            # Simple Euler-like step for BPTT stability first
            # dot_q = p
            # dot_p = -grad_V - damping - ... + F_ext
            
            # Let's rely on autograd for H gradients
            # But that is slow inside loop.
            # Explicit forces:
            
            # Potential Force: -grad_V(q)
            # We can use torch.autograd.grad but create_graph=True
            q.requires_grad_(True)
            V_val = self.net_V(q).sum()
            grad_V = torch.autograd.grad(V_val, q, create_graph=True)[0]
            
            # Dissipative Force: -alpha(q) * p (simplified contact)
            alpha = self.generator.net_Alpha(q).sigmoid() # [B, 1]
            F_diss = -alpha * p
            
            # Update p
            # p_new = p + dt * (F_ext + F_diss - grad_V)
            # Note: F_ext (u_t) drives the system
            delta_p = (F_ext + F_diss - grad_V) * dt
            p_next = p + delta_p
            
            # Update q
            # q_new = q + dt * p_next (Symplectic-ish)
            q_next = q + p_next * dt
            
            # Update s (Action functional)
            # dot_s = p * dot_q - H + ... (Skip for NLL, keep for Lyap)
            s_next = s # Placeholder
            
            # Update State
            q, p, s = q_next, p_next, s_next
            
            # Collect for Decoding
            state_trajectory.append(q)
            q_traj.append(q)
            
        # Stack trajectory: [B, L-1, dim_q]
        states_stack = torch.stack(state_trajectory, dim=1)
        
        # 3. Decode & Loss
        # We predict x_{t+1} from state at t+1. 
        # Flatten time into batch to use StateDecoder per-step (z_t -> x_{t+1})
        # This prevents cross-attention to future states and forces dependence on z.
        flat_states = states_stack.reshape(-1, self.dim_q) # [B*(L-1), dim_q]
        
        # Decoder
        # Passing prev_tokens=None forces Decoder to use BOS query + State Memory
        # This ensures the prediction relies on 'z', not just copying 'x_t'.
        logits = self.port.decoder(flat_states, prev_tokens=None) # [B*(L-1), 1, V]
        logits = logits.squeeze(1) # [B*(L-1), V]
        
        # Targets
        targets = batch_tokens[:, 1:] # [B, L-1]
        flat_targets = targets.reshape(-1) # [B*(L-1)]
        
        # NLL Loss
        loss_nll = nn.CrossEntropyLoss()(logits, flat_targets)
        
        # Lyapunov / Energy Loss (Regularization)
        # H should decrease (or satisfy contact cond). 
        # Here we just penalize divergence: ||q||^2
        loss_reg = 1e-3 * (states_stack**2).mean()
        
        loss_total = loss_nll + loss_reg
        
        # Backward
        loss_total.backward()
        
        # Clip Grad
        nn.utils.clip_grad_norm_(self.port.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.net_V.parameters(), 1.0)
        
        # Step
        self.opt_port.step()
        if self.args.train_core:
            self.opt_core.step()
            
        return {
            "loss": loss_total.item(),
            "nll": loss_nll.item(),
            "reg": loss_reg.item(),
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

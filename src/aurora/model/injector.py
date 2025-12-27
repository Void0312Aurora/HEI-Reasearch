"""
Aurora Injector Network (CCD v3.1).
===================================

Maps Char Stream -> Physical State (Psi, u).
Ref: Axiom 2.1.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AuroraInjector(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, space_dim: int, num_layers: int = 2):
        super().__init__()
        self.space_dim = space_dim
        
        # 1. Char Encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Context Processing (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Heads
        # A. Mass Head (Scalar m > 0)
        self.mass_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Ensure positive mass
        )
        
        # B. Rest Position Head (Direction only, Radius from Entropy)
        # Output unit vector in R^n
        self.dir_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, space_dim)
        )
        
        # C. Charge Head (Lie Algebra element J in so(space_dim)?)
        # Or internal logic space? Let's assume J lives in so(dim).
        # Output skew-symmetric k*k
        k = space_dim # Or different logical dim? Let's match space dim for now.
        self.charge_dim = k
        self.charge_flat = k * k
        self.charge_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.charge_flat)
        )
        
        # D. Momentum Head (p(0) perturbation)
        self.momentum_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, space_dim)
        )

    def forward(self, char_ids: torch.Tensor, entropy_r: torch.Tensor):
        """
        Args:
            char_ids: (Batch, Seq)
            entropy_r: (Batch, Seq) Target radius from entropy stats.
            
        Returns:
            mass: (Batch, Seq, 1)
            q_rest: (Batch, Seq, Dim)
            J_rest: (Batch, Seq, Dim, Dim) Skew-Symmetric
            p_init: (Batch, Seq, Dim)
        """
        x = self.embedding(char_ids) # (B, S, E)
        feat = self.transformer(x)   # (B, S, E) (Contextualized)
        
        # 1. Mass
        mass = self.mass_head(feat) + 1e-4 # Avoid zero mass
        
        # 2. Rest Position
        # Direction
        raw_dir = self.dir_head(feat)
        # Normalize to unit sphere
        u_dir = F.normalize(raw_dir, p=2, dim=-1)
        # Scale by entropy radius
        q_rest = u_dir * entropy_r.unsqueeze(-1)
        
        # 3. Charge
        raw_J = self.charge_head(feat)
        B, S, _ = raw_J.shape
        raw_J = raw_J.view(B, S, self.charge_dim, self.charge_dim)
        # Force Skew-Symmetric (Lie Algebra)
        J_rest = 0.5 * (raw_J - raw_J.transpose(-1, -2))
        
        # 4. Momentum
        # This is the "Innovation" term added to transported momentum
        p_init = self.momentum_head(feat)
        
        return mass, q_rest, J_rest, p_init

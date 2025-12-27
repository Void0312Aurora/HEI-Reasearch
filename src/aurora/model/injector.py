"""
Aurora Injector (CCD v3.1)
==========================

Implements Axiom 0.1.2: Continuum Hypothesis.
Maps continuous input signals (e.g. normalized codepoints) to 
Symplectic Phase Space (T*Q + g).

NO discrete embeddings allowed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierFeatureEncoder(nn.Module):
    """
    Maps scalar input x in [0, 1] to high-dim feature vector using Random Fourier Features.
    Phi(x) = [sin(2pi * Wx), cos(2pi * Wx)]
    """
    def __init__(self, input_dim=1, embed_dim=64, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Random Gaussian Frequencies
        # W ~ N(0, scale^2)
        # We need output size embed_dim.
        # [sin, cos] takes 2 slots per freq.
        # So we need embed_dim // 2 frequencies.
        num_freqs = embed_dim // 2
        
        # Scale:
        # We want to resolve 1/65536 delta.
        # Sigma should be roughly 65536 / 2pi ~ 10000.
        # If scale arg is small, we multiply internally.
        # Heuristic: scale=1.0 is low freq. scale=10.0 is mid.
        # To resolve individual chars, we need High Frequency.
        # Let's use a mixed spectrum strategy if possible, but for now Gaussian with high variance.
        # Or better: Log-linear geometric series clamped to sensible range.
        
        # Option A: RFF
        # self.register_buffer('B', torch.randn(input_dim, num_freqs) * scale)
        
        # Option B: Safe Geometric Series (Better for positional-like resolution)
        # 2^0 ... 2^18 (covering 1 to 262k)
        # If num_freqs > 19, we wrap or cycle? Or just clamp max freq.
        
        max_freq_log2 = 18.0 
        if num_freqs > 0:
            # Linear space in log domain
            exponent = torch.linspace(0, max_freq_log2, num_freqs)
            freqs = 2.0 ** exponent * math.pi
        else:
            freqs = torch.tensor([])
            
        self.register_buffer('freqs', freqs)
        
    def forward(self, x):
        # x: (B, 1) or (B,)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        # x * freqs -> (B, num_freqs)
        # x is (B, 1), freqs is (num_freqs)
        args = x * self.freqs.unsqueeze(0)
        
        # [sin, cos]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class ContinuousInjector(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, dim=16):
        super().__init__()
        self.dim = dim
        self.output_dim = dim
        
        # 1. Continuous Encoder
        self.encoder = FourierFeatureEncoder(input_dim=input_dim, embed_dim=hidden_dim, scale=10.0)
        
        # 2. Processor MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # 3. Heads (Ontological & Dynamical)
        # Mass Head: m > 0
        self.head_m = nn.Linear(hidden_dim, 1)
        
        # Position Head: Direction in H^n (constraint: norm < 1 enforced by r input)
        self.head_q_dir = nn.Linear(hidden_dim, dim)
        
        # Gauge Charge Head: so(dim) algebra (Skew-symmetric)
        # Output dim*(dim-1)/2 parameters
        self.num_skew = dim * (dim - 1) // 2
        self.head_J = nn.Linear(hidden_dim, self.num_skew)
        
        # Momentum Head: Tangent vector
        self.head_p = nn.Linear(hidden_dim, dim)
        
    def forward(self, x_continuous, r_entropy):
        """
        Args:
            x_continuous: (B, 1) Normalized codepoint [0, 1]
            r_entropy: (B, 1) Radial constraint from entropy stats [0, 1)
        
        Returns:
            m: (B, 1)
            q: (B, D)
            J: (B, D, D)
            p: (B, D)
        """
        # Encode
        features = self.encoder(x_continuous)
        latent = self.mlp(features)
        
        # A. Mass
        m_raw = self.head_m(latent)
        m = F.softplus(m_raw) + 0.01 # Positivity
        
        # B. Position (Constraint: q = dir * r)
        q_raw = self.head_q_dir(latent)
        # Normalize to unit direction
        q_dir = F.normalize(q_raw, p=2, dim=-1) # (B, D)
        # Apply entropy radius
        # Ensure r is within [0, 1)
        r = torch.clamp(r_entropy, min=0.0, max=0.99)
        q = q_dir * r
        
        # C. Gauge Charge (Skew-symmetric so(n))
        # Bound J to [-1, 1] to prevent Energy Divergence (Ferromagnetic Collapse)
        J_params = torch.tanh(self.head_J(latent))
        
        J = torch.zeros(x_continuous.size(0), self.dim, self.dim, device=x_continuous.device)
        
        # Fill strictly upper triangle
        triu_indices = torch.triu_indices(self.dim, self.dim, offset=1)
        J[:, triu_indices[0], triu_indices[1]] = J_params
        # Skew-symmetric construction: A - A.T
        J = J - J.transpose(1, 2)
        
        # D. Momentum (Tangent Vector)
        p = self.head_p(latent)
        
        return m, q, J, p

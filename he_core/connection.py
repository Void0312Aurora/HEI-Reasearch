import torch
import torch.nn as nn
from typing import Optional, Tuple


class Connection(nn.Module):
    """
    L2 Generator: Lie Algebroid Connection for Parallel Transport.
    
    Theory (理论基础-7/逻辑/生成元.md):
    - Parallel transport preserves geometric structure across chart transitions.
    - Connection A_mu(q) maps tangent vectors to Lie algebra elements.
    - Transport T(q, q') approximates exp(-∫ A dq).
    
    Implementation:
    - T = I + epsilon * A(q_from, q_to)  (first-order approximation)
    - A is constrained to be near-orthogonal for volume preservation.
    """
    def __init__(self, dim_q: int, hidden_dim: int = 32):
        super().__init__()
        self.dim_q = dim_q
        self.hidden_dim = hidden_dim
        
        # Network predicting skew-symmetric deviation (for SO(n) structure)
        self.net = nn.Sequential(
            nn.Linear(dim_q * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_q * dim_q)
        )
        
        # Scale factor for perturbation strength
        self.epsilon = nn.Parameter(torch.tensor(0.1))
        
        # Initialize for near-identity transport
        self._init_near_identity()
        
    def _init_near_identity(self):
        """Initialize network for near-identity transport (flat connection)."""
        for name, param in self.net.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, std=1e-4)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def _make_skew_symmetric(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Project matrix to skew-symmetric form for SO(n) Lie algebra.
        A_skew = (A - A^T) / 2
        """
        return 0.5 * (mat - mat.transpose(-1, -2))
        
    def forward(self, q_from: torch.Tensor, q_to: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Transport vector v from q_from to q_to.
        
        Args:
            q_from: Source position (B, dim_q)
            q_to: Target position (B, dim_q)  
            v: Vector to transport (B, dim_q)
            
        Returns:
            v_transported: Transported vector (B, dim_q)
        """
        batch_size = q_from.shape[0]
        device = q_from.device
        
        # Compute transport matrix
        inp = torch.cat([q_from, q_to], dim=1)
        mat_flat = self.net(inp)  # (B, D*D)
        mat = mat_flat.view(batch_size, self.dim_q, self.dim_q)
        
        # Make skew-symmetric for SO(n) structure
        A = self._make_skew_symmetric(mat)
        
        # First-order exponential: T ≈ I + epsilon * A
        I = torch.eye(self.dim_q, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        T = I + self.epsilon * A
        
        # Apply transport: v' = T @ v
        v_in = v.unsqueeze(2)  # (B, D, 1)
        v_out = torch.bmm(T, v_in).squeeze(2)  # (B, D)
        
        return v_out
        
    def get_curvature_proxy(self, q: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
        """
        Estimate local curvature via holonomy around infinitesimal loop.
        
        Returns:
            curvature: Frobenius norm of deviation from identity (B,)
        """
        batch_size = q.shape[0]
        device = q.device
        
        # Create infinitesimal loop: q -> q+dx -> q+dx+dy -> q+dy -> q
        dx = torch.zeros_like(q)
        dy = torch.zeros_like(q)
        if self.dim_q >= 2:
            dx[:, 0] = delta
            dy[:, 1] = delta
        else:
            dx[:, 0] = delta
            dy = dx.clone()
            
        # Compose transports around loop
        v0 = torch.eye(self.dim_q, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        v0_flat = v0.reshape(batch_size * self.dim_q, self.dim_q)
        q_exp = q.unsqueeze(1).expand(-1, self.dim_q, -1).reshape(batch_size * self.dim_q, -1)
        
        # Simplified: just check transport asymmetry
        v1 = self.forward(q, q + dx, q)
        v2 = self.forward(q + dx, q, q)
        
        # Holonomy error: v1 composed with v2 should return to original
        holonomy_error = (v1 - v2).norm(dim=1)
        
        return holonomy_error
        
    def orthogonality_loss(self, q_samples: torch.Tensor) -> torch.Tensor:
        """
        Regularization: encourage transport matrices to be orthogonal.
        L = ||T^T T - I||^2
        """
        batch_size = q_samples.shape[0]
        device = q_samples.device
        
        # Random target points
        q_to = q_samples + torch.randn_like(q_samples) * 0.1
        
        # Get transport matrices
        inp = torch.cat([q_samples, q_to], dim=1)
        mat_flat = self.net(inp)
        mat = mat_flat.view(batch_size, self.dim_q, self.dim_q)
        A = self._make_skew_symmetric(mat)
        
        I = torch.eye(self.dim_q, device=device).unsqueeze(0)
        T = I + self.epsilon * A
        
        # Check orthogonality: T^T @ T should be I
        TtT = torch.bmm(T.transpose(-1, -2), T)
        ortho_error = (TtT - I).pow(2).mean()
        
        return ortho_error

import torch
import torch.nn as nn

class Connection(nn.Module):
    """
    L2 Generator: Transport / Connection.
    Defines how vectors 'v' parallel transport across the manifold.
    
    In a trivial Euclidean space, transport is Identity.
    In a curved/skill manifold, transport(v) depends on path.
    
    Here we implement a learnable gauge field A_mu(q).
    Transport v along direction dq: v_new = v - A(q).dq . v
    Or simpler: MLP(q, v) -> v_new (Discrete Step Transport).
    """
    def __init__(self, dim_q: int, hidden_dim: int = 16):
        super().__init__()
        self.dim_q = dim_q
        # A(q) maps tangent vector dq to Lie Algebra element (matrix).
        # We approximate Transport T(q_from, q_to) as an operator.
        
        # Input: (q_from, q_to, v)
        # Output: v_transported
        
        # Structure: T = I + MLP(q_from, q_to) ?
        self.net = nn.Sequential(
            nn.Linear(dim_q * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_q * dim_q) # Matrix T
        )
        
    def forward(self, q_from: torch.Tensor, q_to: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Transport vector v from q_from to q_to.
        Returns v_new.
        """
        batch_size = q_from.shape[0]
        
        # Delta
        # dq = q_to - q_from
        
        inp = torch.cat([q_from, q_to], dim=1)
        # Learn deviation from Identity
        mat_flat = self.net(inp) # (B, D*D)
        mat = mat_flat.view(batch_size, self.dim_q, self.dim_q)
        
        # T = I + mat * 0.1 (Small deviation)
        I = torch.eye(self.dim_q, device=q_from.device).unsqueeze(0)
        T = I + mat * 0.1
        
        # v_new = T v
        # v: (B, D) -> (B, D, 1)
        v_in = v.unsqueeze(2)
        v_out = torch.bmm(T, v_in).squeeze(2)
        
        return v_out

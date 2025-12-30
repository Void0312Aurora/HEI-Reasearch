import torch
import torch.nn as nn
from typing import Callable, Optional
from he_core.state import ContactState
from he_core.generator import BaseGenerator, DissipativeGenerator

class PortCoupling(nn.Module):
    """
    Defines the shape B(q) in the coupling term <u, B(q)>.
    Default: B(q) = -q (Linear Force Coupling).
    Learnable: B(q) = W q (Linear Map).
    Mixture: B(q) = sum w_k * W_k q (Gated Linear Map).
    """
    def __init__(self, dim_q: int, dim_u: int, learnable: bool = False, num_charts: int = 1):
        super().__init__()
        self.dim_q = dim_q
        self.dim_u = dim_u
        self.learnable = learnable
        self.num_charts = num_charts
        
        if learnable:
            # We need K matrices if num_charts > 1
            # Dictionary of matrices? Or tensor?
            # Tensor (K, dim_u, dim_q)
            self.W_stack = nn.Parameter(torch.Tensor(num_charts, dim_u, dim_q))
            
            # Init
            with torch.no_grad():
                 # Initialize all near Identity/Negative Identity
                 for k in range(num_charts):
                     if dim_q == dim_u:
                         self.W_stack[k].copy_(-torch.eye(dim_q) + torch.randn(dim_q, dim_q)*0.01)
        
    def forward(self, q: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns B(q).
        weights: (B, K) or None.
        """
        if not self.learnable:
            return -q # Assumes dim_q == dim_u
            
        # q: (Batch, dim_q)
        # weights: (Batch, num_charts)
        
        # B(q) = sum_k w_k * W_k . q
        # Linear map for each chart: y_k = W_k q
        # Result y = sum w_k y_k
        
        # 1. Compute all candidates
        # q -> (Batch, 1, dim_q)
        q_in = q.unsqueeze(1) # (B, 1, Dq)
        # W_stack -> (K, Du, Dq)
        # We need (Batch, K, Du).
        # We can do: Einsum?
        # y_all = q W^T
        # Shape: (Batch, K, Du)
        
        # Let's broadcast W: (1, K, Du, Dq)
        W_b = self.W_stack.unsqueeze(0) # (1, K, Du, Dq)
        
        # q_b: (Batch, 1, Dq, 1) -> W @ q?
        # y_k = W_k q
        # (K, Du, Dq) @ (Batch, Dq, 1) -> No
        
        # Einsum: b=batch, k=chart, u=dim_u, q=dim_q
        # W: k u q
        # q: b q
        # out: b k u
        y_all = torch.einsum('kuq,bq->bku', self.W_stack, q)
        
        if weights is None:
            # If no weights provided but K>1, average? Or use chart 0?
            # Default to uniform or error?
            if self.num_charts > 1:
                # Assume independent sum? No, Mixture requires weights.
                # Fallback: Mean
                weights = torch.ones(q.shape[0], self.num_charts, device=q.device) / self.num_charts
            else:
                weights = torch.ones(q.shape[0], 1, device=q.device)
                
        # 2. Weighted Sum
        # weights: (B, K) -> (B, K, 1)
        w_b = weights.unsqueeze(2)
        
        B_q = (y_all * w_b).sum(dim=1) # (B, Du)
        
        return B_q

class PortCoupledGenerator(nn.Module):
    """
    Template 3: Hamiltonian with Port Coupling.
    H(x, u, t) = H_int(x) + H_port(x, u)
    H_port = <u, B(q)>
    """
    def __init__(self, internal_generator: BaseGenerator, dim_u: int, learnable_coupling: bool = False, num_charts: int = 1):
        super().__init__()
        self.internal = internal_generator
        self.dim_u = dim_u
        self.dim_q = internal_generator.dim_q
        self.coupling = PortCoupling(self.dim_q, dim_u, learnable=learnable_coupling, num_charts=num_charts)
        
    def forward(self, state: ContactState, u_ext: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns Total H.
        u_ext: (B, dim_u) External Input.
        weights: (B, K) Chart Weights.
        """
        # 1. Internal H
        H_int = self.internal(state)
        
        # 2. Port H
        if u_ext is None:
            return H_int
            
        # H_port = sum(u * B(q))
        # B(q) depends on weights now
        B_q = self.coupling(state.q, weights=weights)
        
        # Dot product
        H_port = (u_ext * B_q).sum(dim=1, keepdim=True)
        
        return H_int + H_port

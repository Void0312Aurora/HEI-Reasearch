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
    """
    def __init__(self, dim_q: int, dim_u: int, learnable: bool = False):
        super().__init__()
        self.dim_q = dim_q
        self.dim_u = dim_u
        self.learnable = learnable
        
        if learnable:
            # Linear map from q (D_q) to B (D_u)
            # Actually B(q) must have shape (dim_u).
            # Wait, <u, B(q)>. u is (dim_u). B(q) is (dim_u).
            # So map q -> dim_u.
            self.W = nn.Linear(dim_q, dim_u, bias=False)
            
            # Init to Identity/-Identity to start close to default?
            if dim_q == dim_u:
                 with torch.no_grad():
                     self.W.weight.copy_(-torch.eye(dim_q))
        
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Returns B(q)"""
        if self.learnable:
            return self.W(q)
        else:
            return -q # Assumes dim_q == dim_u

class PortCoupledGenerator(nn.Module):
    """
    Template 3: Hamiltonian with Port Coupling.
    H(x, u, t) = H_int(x) + H_port(x, u)
    H_port = <u, B(q)>
    """
    def __init__(self, internal_generator: BaseGenerator, dim_u: int, learnable_coupling: bool = False):
        super().__init__()
        self.internal = internal_generator
        self.dim_u = dim_u
        self.dim_q = internal_generator.dim_q
        self.coupling = PortCoupling(self.dim_q, dim_u, learnable=learnable_coupling)
        
    def forward(self, state: ContactState, u_ext: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns Total H.
        u_ext: (B, dim_u) External Input (from Env or Replay).
               If None (or Zero), returns H_int.
        """
        # 1. Internal H
        H_int = self.internal(state)
        
        # 2. Port H
        if u_ext is None:
            return H_int
            
        # H_port = sum(u * B(q))
        # u is (B, U). B(q) is (B, U).
        B_q = self.coupling(state.q)
        
        # Dot product
        H_port = (u_ext * B_q).sum(dim=1, keepdim=True)
        
        return H_int + H_port

import torch
import torch.nn as nn
from typing import Callable, Optional
from he_core.state import ContactState
from he_core.generator import BaseGenerator, DissipativeGenerator

class PortCoupling(nn.Module):
    """
    Defines the shape B(q) in the coupling term <u, B(q)>.
    Default: B(q) = -q (Linear Force Coupling)
    Force = -dH/dq = - d/dq <u, -q> = u.
    """
    def __init__(self, dim_q: int, dim_u: int):
        super().__init__()
        self.dim_q = dim_q
        self.dim_u = dim_u
        # Optional learnable coupling map?
        # For now, assume dim_u = dim_q and B(q) = -q
        assert dim_u == dim_q, "Simple coupling requires dim_u == dim_q"
        
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Returns B(q)"""
        return -q

class PortCoupledGenerator(nn.Module):
    """
    Template 3: Hamiltonian with Port Coupling.
    H(x, u, t) = H_int(x) + H_port(x, u)
    H_port = <u, B(q)>
    """
    def __init__(self, internal_generator: BaseGenerator, dim_u: int):
        super().__init__()
        self.internal = internal_generator
        self.dim_u = dim_u
        self.dim_q = internal_generator.dim_q
        self.coupling = PortCoupling(self.dim_q, dim_u)
        
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

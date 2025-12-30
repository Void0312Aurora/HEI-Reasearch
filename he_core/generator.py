import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from he_core.state import ContactState

class BaseGenerator(nn.Module, ABC):
    def __init__(self, dim_q: int):
        super().__init__()
        self.dim_q = dim_q
        
    @abstractmethod
    def forward(self, state: ContactState) -> torch.Tensor:
        """Returns H(q, p, s) of shape (B, 1)"""
        pass

class DissipativeGenerator(BaseGenerator):
    """
    Template for A3 (Unified Potential) and A4 (Identity).
    H(q,p,s) = K(p) + V(q) + alpha * s
    K(p) = 0.5 * p^2
    V(q) = Neural Potential or Quadratic
    alpha * s term generates friction 'alpha'.
    """
    def __init__(self, dim_q: int, alpha: float = 0.1):
        super().__init__(dim_q)
        self.alpha = alpha # Damping coeff
        
        # Potential V(q)
        self.net_V = nn.Sequential(
            nn.Linear(dim_q, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
    def forward(self, state: ContactState) -> torch.Tensor:
        q, p, s = state.q, state.p, state.s
        
        # Kinetic
        K = 0.5 * (p**2).sum(dim=1, keepdim=True)
        
        # Potential
        V = self.net_V(q)
        
        # Contact Potential (Dissipation Source)
        # Linear in s
        S_term = self.alpha * s
        
        return K + V + S_term

class DrivenGenerator(DissipativeGenerator):
    """
    H with External Drive.
    H = H_0 + H_int(q, u)
    Simple coupling: u * q (Force)
    """
    def __init__(self, dim_q: int, alpha: float = 0.1):
        super().__init__(dim_q, alpha)
        
    def forward_with_drive(self, state: ContactState, drive: torch.Tensor) -> torch.Tensor:
        # Standard H
        H_0 = super().forward(state)
        
        # Coupling H_int = -drive * q (Force potential)
        # Force = -dH/dq = drive.
        # So H_int = - (drive * q).sum()
        H_int = - (drive * state.q).sum(dim=1, keepdim=True)
        
        return H_0 + H_int
        return H_0 + H_int

class DeepDissipativeGenerator(DissipativeGenerator):
    """
    Dissipative Generator with customizable Deep Potential V(q).
    Allows passing an arbitrary nn.Module for V(q).
    """
    def __init__(self, dim_q: int, alpha: float = 0.1, net_V: nn.Module = None):
        super().__init__(dim_q, alpha)
        
        if net_V is not None:
            self.net_V = net_V
        else:
            # Default Deep MLP
            self.net_V = nn.Sequential(
                nn.Linear(dim_q, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            
    # forward logic is inherited from DissipativeGenerator
    # as it uses self.net_V(q)

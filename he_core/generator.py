import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
from he_core.state import ContactState

class BaseGenerator(nn.Module, ABC):
    def __init__(self, dim_q: int):
        super().__init__()
        self.dim_q = dim_q
        
    @abstractmethod
    def forward(self, state: ContactState, z: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    def __init__(self, dim_q: int, alpha: float = 0.1, stiffness: float = 0.0):
        super().__init__(dim_q)
        self.alpha = alpha # Damping coeff
        self.stiffness = stiffness # Harmonic Confinement (k)
        
        # Potential V(q)
        self.net_V = nn.Sequential(
            nn.Linear(dim_q, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
    def _infer_net_v_in_dim(self) -> Optional[int]:
        if isinstance(self.net_V, nn.Linear):
            return self.net_V.in_features
        if isinstance(self.net_V, nn.Sequential):
            for layer in self.net_V:
                if isinstance(layer, nn.Linear):
                    return layer.in_features
        return getattr(self.net_V, "in_features", None)

    def _prepare_potential_input(self, q: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        in_dim = self._infer_net_v_in_dim()
        q_dim = q.shape[1]

        if in_dim is None:
            return torch.cat([q, z], dim=1) if z is not None else q

        if in_dim == q_dim:
            return q

        if z is None:
            pad = torch.zeros(q.shape[0], max(in_dim - q_dim, 0), device=q.device, dtype=q.dtype)
            return torch.cat([q, pad], dim=1)[:, :in_dim]

        input_qz = torch.cat([q, z], dim=1)
        if input_qz.shape[1] == in_dim:
            return input_qz
        if input_qz.shape[1] > in_dim:
            return input_qz[:, :in_dim]
        pad = torch.zeros(q.shape[0], in_dim - input_qz.shape[1], device=q.device, dtype=q.dtype)
        return torch.cat([input_qz, pad], dim=1)

    def forward(self, state: ContactState, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, p, s = state.q, state.p, state.s
        
        # Kinetic
        K = 0.5 * (p**2).sum(dim=1, keepdim=True)
        
        # Potential
        V_inp = self._prepare_potential_input(q, z)
        V = self.net_V(V_inp)
        
        # Harmonic Confinement (First Principles Stability)
        # V_conf = 0.5 * k * q^2
        if self.stiffness > 0:
            V_conf = 0.5 * self.stiffness * (q**2).sum(dim=1, keepdim=True)
            V = V + V_conf
        
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
    def __init__(self, dim_q: int, alpha: float = 0.1, stiffness: float = 0.0, net_V: nn.Module = None):
        super().__init__(dim_q, alpha, stiffness)
        
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

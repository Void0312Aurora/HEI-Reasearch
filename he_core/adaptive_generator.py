
import torch
import torch.nn as nn
from he_core.state import ContactState
from he_core.generator import DeepDissipativeGenerator

class AdaptiveDissipativeGenerator(DeepDissipativeGenerator):
    """
    Theoretical Closure: State-Dependent Damping.
    H(q,p,s) = K(p) + V(q) + Alpha(q) * s
    Alpha(q) >= 0 is learned.
    """
    def __init__(self, dim_q: int, net_V: nn.Module = None):
        super().__init__(dim_q, alpha=0.0, net_V=net_V) # Base alpha unused
        
        # Learnable Damping Field A(q)
        self.net_Alpha = nn.Sequential(
            nn.Linear(dim_q, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus() # Ensure alpha >= 0
        )
        
    def forward(self, state: ContactState) -> torch.Tensor:
        q, p, s = state.q, state.p, state.s
        
        # Kinetic
        K = 0.5 * (p**2).sum(dim=1, keepdim=True)
        
        # Potential
        V = self.net_V(q)
        
        # Adaptive Dissipation
        # alpha is now a field A(q) -> (B, 1)
        # Add small epsilon to prevent singularity if needed, though softplus is safe
        alpha_q = self.net_Alpha(q) + 1e-3 
        
        # Contact Potential
        # S_term = alpha(q) * s
        S_term = alpha_q * s
        
        return K + V + S_term

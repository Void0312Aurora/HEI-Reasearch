
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
    def __init__(self, dim_q: int, net_V: nn.Module = None, dim_z: int = 0, stiffness: float = 0.0):
        super().__init__(dim_q, alpha=0.0, net_V=net_V, stiffness=stiffness) # Base alpha unused
        self.dim_z = dim_z
        
        # Learnable Damping Field A(q)
        # If z is used, Alpha might depend on z too?
        # For strict A3, Alpha(q) is geometric. V(q, z) represents intent.
        # Let's keep Alpha(q) pure for now, unless requested.
        self.net_Alpha = nn.Sequential(
            nn.Linear(dim_q, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus() # Ensure alpha >= 0
        )
        
        self._init_stable_alpha()
        
    def _init_stable_alpha(self):
        """
        Theoretical Stability:
        Initialize Alpha(q) to be nearly constant (nabla Alpha ~ 0) but positive.
        This prevents the -s * grad(Alpha) term from pumping energy during early training.
        """
        for name, param in self.net_Alpha.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1e-5) # Almost zero gradient
            elif 'bias' in name:
                # For hidden layers, bias=0
                # For output layer, bias gives the initial damping level
                # softplus(x) = alpha. Let alpha=0.1. x ~ -2.2
                if param.shape[0] == 1: # Output layer (assuming latent=64)
                    nn.init.constant_(param, -2.0) # Start with small positive damping
                else:
                    nn.init.zeros_(param)
        
    def forward(self, state: ContactState, z: torch.Tensor = None) -> torch.Tensor:
        q, p, s = state.q, state.p, state.s
        
        # Kinetic
        K = 0.5 * (p**2).sum(dim=1, keepdim=True)
        
        # Potential V(q, z)
        if self.dim_z > 0 and z is not None:
            # Check if net_V expects concatenated input
            # We assume net_V is designed to handle dim_q + dim_z if created externally
            # OR we concatenate here if net_V handles it.
            # Best practice: The entity or script creates net_V with correct input dim.
            # We just concat.
            inp = torch.cat([q, z], dim=1)
            V = self.net_V(inp)
        else:
            V = self.net_V(q)
            
        # Harmonic Confinement (First Principles Stability)
        if self.stiffness > 0:
            V_conf = 0.5 * self.stiffness * (q**2).sum(dim=1, keepdim=True)
            V = V + V_conf
        
        # Adaptive Dissipation
        # alpha is now a field A(q) -> (B, 1)
        # Add small epsilon to prevent singularity if needed, though softplus is safe
        alpha_q = self.net_Alpha(q) + 1e-3 
        
        # Contact Potential
        # S_term = alpha(q) * s
        S_term = alpha_q * s
        
        return K + V + S_term

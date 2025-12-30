"""
Port Adapters.
Bridges raw sensory data (Images, Audio) to Energy Port variables (u, q_init).

Phase 18.1: ImagePortAdapter.
CONCEPTS:
- Perception as "Drive": Image provides a constant external force `u_ext` that biases the dynamics.
- Perception as "State": Image sets the initial condition `q_0`, and dynamics "relax" or "inference" from there.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImagePortAdapter(nn.Module):
    """
    Adapts (B, C, H, W) images to Port Variables.
    """
    def __init__(self, in_channels: int = 1, dim_out: int = 2):
        super().__init__()
        self.dim_out = dim_out
        
        # Simple ConvNet Encoder
        # MNIST: 1x28x28 -> dim_out
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, dim_out) # Output is u or q
        )
        
        # Initialize small for stability
        with torch.no_grad():
            self.encoder[-1].weight.normal_(0, 0.01)
            self.encoder[-1].bias.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: Encode to vector."""
        return self.encoder(x)
        
    def get_drive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return External Force `u_ext`.
        Usage: entity.forward(..., u_ext=adapter.get_drive(img))
        """
        return self.encoder(x)
        
    def get_initial_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return Initial Position `q_0`.
        Usage: entity.state.q = adapter.get_initial_state(img)
        """
        return self.encoder(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVisionEncoder(nn.Module):
    """
    Maps visual input (Image) to Dynamic Drive (u).
    Architecture: LeNet-like 2-layer CNN + MLP projection.
    
    A5 Axiom: Demonstrates perception (Vision) as a drive for the Entity.
    """
    def __init__(self, in_channels: int = 1, dim_out: int = 32):
        super().__init__()
        # Input: (B, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(2)
        
        # Flatten: 32 * 7 * 7
        self.flat_dim = 32 * 7 * 7
        
        self.proj = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, dim_out)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already normalized by DataLoader
        # x shape: (B, 1, 28, 28)
            
        h = F.relu(self.conv1(x)) # 28 -> 28
        h = self.mp1(h)           # 28 -> 14
        h = F.relu(self.conv2(h)) # 14 -> 14
        h = self.mp2(h)           # 14 -> 7
        
        h = h.reshape(h.size(0), -1)
        return self.proj(h)

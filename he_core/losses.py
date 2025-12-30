import torch
import torch.nn as nn
from typing import Dict, Any

class SelfSupervisedLosses:
    """
    Self-Supervised Losses for Phase 17.
    Converts diagnostic signals into differentiable objectives.
    """
    
    @staticmethod
    def l1_dissipative_loss(energies: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """
        Penalize energy increase (Dissipation Check).
        Target: E[t+1] <= E[t]
        Loss = mean(ReLU(E[t+1] - E[t] + margin))
        
        Args:
            energies: Tensor of shape (Steps, Batch) or (Steps,)
            margin: Tolerance margin (default 0.0)
            
        Returns:
            Scalar Loss
        """
        # Ensure shape (Steps, ...)
        if energies.dim() == 1:
            energies = energies.unsqueeze(1)
            
        diffs = energies[1:] - energies[:-1]
        
        # We want difference to be negative (dissipative)
        # If diff > -margin, we penalize.
        # Wait, if diff = -0.1 and margin=0, ReLU(-0.1) = 0. OK.
        # If diff = 0.1, ReLU(0.1) = 0.1. Penalized. OK.
        
        loss = torch.relu(diffs + margin).mean()
        return loss

    @staticmethod
    def l2_holonomy_loss(initial_state_flat: torch.Tensor, final_state_flat: torch.Tensor) -> torch.Tensor:
        """
        Penalize Holonomy Loop Error (Consistency Check).
        Loss = MSE(x_initial, x_final)
        
        Args:
            initial_state_flat: (B, D)
            final_state_flat: (B, D)
            
        Returns:
            Scalar Loss
        """
        return torch.nn.functional.mse_loss(final_state_flat, initial_state_flat)

    @staticmethod
    def robustness_hinge_loss(robustness_values: torch.Tensor, safe_margin: float = 1.0) -> torch.Tensor:
        """
        Penalize low robustness (Safety Check).
        Target: robustness > safe_margin
        Loss = ReLU(safe_margin - robustness)
        
        Args:
            robustness_values: Tensor of robustness scores
            safe_margin: Minimum required robustness
            
        Returns:
            Scalar Loss (Mean over inputs)
        """
        return torch.relu(safe_margin - robustness_values).mean()

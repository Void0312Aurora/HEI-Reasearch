"""
Gate-0 Initial Gain Regression Test.

This test asserts that after the initialization fixes (ActionInterface near-Identity,
Entity reset scale=0.1), the initial Port Gain is below the critical threshold (< 2.0).

This fixes the "evidence chain gap" identified in temp-05.md: proving that the
system starts within a trainable region, not just that the gate can catch violations.
"""
import torch
import unittest
from he_core.entity_v4 import UnifiedGeometricEntity

class TestGate0InitialGain(unittest.TestCase):
    def test_initial_gain_under_threshold(self):
        """
        Verify that init Gain < 2.0 under fixed seed.
        Gain = ||action|| / ||u||.
        """
        print("\n--- Gate-0 Initial Gain Test ---")
        
        # Fixed seed for reproducibility
        torch.manual_seed(42)
        
        config = {
            'dim_q': 2,
            'learnable_coupling': True,
            'num_charts': 3,
            'damping': 0.1
        }
        entity = UnifiedGeometricEntity(config)
        entity.reset() # Uses scale=0.1
        
        # Simulate one step with a small input
        u_ext = torch.randn(1, config['dim_q']) * 0.1
        state_flat = entity.state.flat.detach()
        
        out = entity.forward_tensor(state_flat, u_ext)
        
        action = out['action']
        action_norm = action.norm()
        u_norm = u_ext.norm()
        
        gain = action_norm / (u_norm + 1e-6)
        
        print(f"  ||action|| = {action_norm.item():.4f}")
        print(f"  ||u||      = {u_norm.item():.4f}")
        print(f"  Gain       = {gain.item():.4f}")
        
        # Threshold: 2.0 (Small Gain Theorem bound from Protocol 5)
        threshold = 2.0
        
        self.assertLess(gain.item(), threshold, f"Initial Gain {gain.item():.4f} >= {threshold}! System starts outside trainable region.")
        
        print(f"  PASS: Gain {gain.item():.4f} < {threshold}")
        
if __name__ == '__main__':
    unittest.main()

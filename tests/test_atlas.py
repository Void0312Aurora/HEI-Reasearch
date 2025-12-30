import torch
import unittest
from he_core.atlas import Atlas
from he_core.generator import DissipativeGenerator

class TestAtlas(unittest.TestCase):
    def test_transition_sync(self):
        print("\n--- Atlas Consistency Test ---")
        dim_q = 2
        atlas = Atlas(num_charts=2, dim_q=dim_q)
        atlas.add_transition(0, 1)
        
        gen = DissipativeGenerator(dim_q)
        
        # Init distinct states
        atlas.states[0].q = torch.ones(1, dim_q) * 1.0 # Chart 0 at 1.0
        atlas.states[1].q = torch.ones(1, dim_q) * 2.0 # Chart 1 at 2.0
        
        # Initial Loss
        # Map is initialized random/linear.
        loss_0 = atlas.compute_consistency_loss(0, 1)
        print(f"Initial Consistency Loss: {loss_0.item()}")
        
        # Sync x_1 towards phi(x_0)
        # alpha=0.5 -> Should halve the distance effectively (if phi is identity-ish)
        # But phi is random Linear.
        # Whatever phi predicts, x_1 will move towards it.
        # So Loss = ||phi(x0) - x1|| should decrease.
        
        atlas.sync_overlap(0, 1, alpha=0.5)
        
        loss_1 = atlas.compute_consistency_loss(0, 1)
        print(f"Post-Sync Consistency Loss: {loss_1.item()}")
        
        self.assertLess(loss_1, loss_0, "Sync did not reduce consistency mismatch")

if __name__ == '__main__':
    unittest.main()

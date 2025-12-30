"""
Test DeepDissipativeGenerator.
Verifies it accepts custom potentials and computes energy correctly.
"""
import unittest
import torch
import torch.nn as nn
from he_core.generator import DeepDissipativeGenerator
from he_core.state import ContactState

class TestDeepGenerator(unittest.TestCase):
    def test_custom_potential(self):
        print("\n--- Test DeepDissipativeGenerator ---")
        dim_q = 10
        batch_size = 5
        
        # Custom V: Quadratic Well V = 0.5 * k * q^2
        class QuadraticPotential(nn.Module):
            def forward(self, q):
                return 0.5 * (q**2).sum(dim=1, keepdim=True)
                
        net_V = QuadraticPotential()
        gen = DeepDissipativeGenerator(dim_q, alpha=0.1, net_V=net_V)
        
        # State
        state = ContactState(dim_q, batch_size)
        state.q = torch.ones(batch_size, dim_q) # q=1
        state.p = torch.zeros(batch_size, dim_q) # p=0
        state.s = torch.zeros(batch_size, 1) # s=0
        
        # Expected H = K + V + S
        # K=0, S=0.
        # V = 0.5 * 10 * 1^2 = 5.0
        
        H = gen.forward(state)
        print(f"  H value: {H[0].item()}")
        self.assertAlmostEqual(H[0].item(), 5.0)
        
        # Check backward
        H.sum().backward()
        print("  Backward OK")

if __name__ == '__main__':
    unittest.main()

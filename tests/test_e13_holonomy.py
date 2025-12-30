import torch
import unittest
from he_core.connection import Connection
from EXP.diag.holonomy import compute_holonomy

class TestE13Holonomy(unittest.TestCase):
    def test_flat_holonomy(self):
        print("\n--- Task 13.3: Holonomy Test (Flat) ---")
        dim_q = 2
        
        # Mock Connection: Identity
        class IdentityConnection(torch.nn.Module):
            def forward(self, qf, qt, v): return v
            
        points = [torch.randn(2) for _ in range(5)] # Random loop
        v0 = torch.randn(2)
        
        metrics = compute_holonomy(IdentityConnection(), points, v0)
        print(f"Flat Error: {metrics['holonomy_error']}")
        
        self.assertAlmostEqual(metrics['holonomy_error'], 0.0)
        self.assertTrue(metrics['is_flat'])
        
    def test_curved_holonomy(self):
        print("\n--- Task 13.3: Holonomy Test (Curved/Learnable) ---")
        dim_q = 2
        conn = Connection(dim_q)
        
        points = [torch.randn(2) for _ in range(5)]
        v0 = torch.randn(2)
        
        # Untrained random connection likely has error
        metrics = compute_holonomy(conn, points, v0)
        print(f"Curved Error: {metrics['holonomy_error']}")
        
        # Might be small due to initialization 0.1 factor, but should be non-zero
        self.assertGreater(metrics['holonomy_error'], 1e-6)

if __name__ == '__main__':
    unittest.main()

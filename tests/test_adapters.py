"""
Test ImagePortAdapter.
Verifies shape compatibility and gradient flow.
"""
import unittest
import torch
from he_core.adapters import ImagePortAdapter

class TestAdapters(unittest.TestCase):
    def test_image_adapter(self):
        print("\n--- Test ImagePortAdapter ---")
        batch_size = 4
        dim_out = 2
        
        # Fake MNIST batch: (B, 1, 28, 28)
        img = torch.randn(batch_size, 1, 28, 28, requires_grad=True)
        
        adapter = ImagePortAdapter(in_channels=1, dim_out=dim_out)
        
        # 1. Test Drive
        u_ext = adapter.get_drive(img)
        print(f"  Drive Shape: {u_ext.shape}")
        self.assertEqual(u_ext.shape, (batch_size, dim_out))
        
        # 2. Test Gradient Flow
        loss = u_ext.sum()
        loss.backward()
        self.assertIsNotNone(img.grad)
        print("  Gradient Flow: OK")
        
        # 3. Test Init State
        q_0 = adapter.get_initial_state(img)
        self.assertEqual(q_0.shape, (batch_size, dim_out))

if __name__ == '__main__':
    unittest.main()

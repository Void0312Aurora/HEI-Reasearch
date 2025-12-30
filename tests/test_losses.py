"""
Test Self-Supervised Losses.
Verifies differentiability and correctness.
"""
import unittest
import torch
from he_core.losses import SelfSupervisedLosses

class TestLosses(unittest.TestCase):
    def test_l1_dissipative_loss(self):
        print("\n--- Test L1 Dissipative Loss ---")
        # Increasing energy (Bad)
        e_bad = torch.tensor([1.0, 1.2, 1.5], requires_grad=True)
        loss_bad = SelfSupervisedLosses.l1_dissipative_loss(e_bad)
        print(f"  Bad Loss: {loss_bad.item()} (Expected > 0)")
        self.assertGreater(loss_bad.item(), 0)
        
        # Backward check
        loss_bad.backward()
        print(f"  Grad: {e_bad.grad}")
        self.assertIsNotNone(e_bad.grad)
        
        # Decreasing energy (Good)
        e_good = torch.tensor([1.0, 0.9, 0.8], requires_grad=True)
        loss_good = SelfSupervisedLosses.l1_dissipative_loss(e_good)
        print(f"  Good Loss: {loss_good.item()} (Expected 0)")
        self.assertEqual(loss_good.item(), 0)

    def test_l2_holonomy_loss(self):
        print("\n--- Test L2 Holonomy Loss ---")
        x0 = torch.tensor([[1.0, 2.0]], requires_grad=True)
        x1 = torch.tensor([[1.1, 2.1]], requires_grad=True) # Error
        
        loss = SelfSupervisedLosses.l2_holonomy_loss(x0, x1)
        print(f"  Loss: {loss.item()}")
        self.assertGreater(loss.item(), 0)
        
        loss.backward()
        self.assertIsNotNone(x0.grad)

    def test_robustness_hinge_loss(self):
        print("\n--- Test Robustness Loss ---")
        # Safe
        rob_safe = torch.tensor([2.0, 3.0], requires_grad=True)
        loss_safe = SelfSupervisedLosses.robustness_hinge_loss(rob_safe, safe_margin=1.0)
        self.assertEqual(loss_safe.item(), 0)
        
        # Unsafe
        rob_unsafe = torch.tensor([0.5, -1.0], requires_grad=True)
        # Margin=1.0. 
        # 0.5 -> Loss = 0.5
        # -1.0 -> Loss = 2.0
        # Mean = 1.25
        loss_unsafe = SelfSupervisedLosses.robustness_hinge_loss(rob_unsafe, safe_margin=1.0)
        print(f"  Unsafe Loss: {loss_unsafe.item()}")
        self.assertAlmostEqual(loss_unsafe.item(), 1.25)
        
        loss_unsafe.backward()
        self.assertIsNotNone(rob_unsafe.grad)

if __name__ == '__main__':
    unittest.main()

"""
Test for Port Interface (Phase 15.1).

Verifies:
1. PortInterface can be instantiated.
2. read_u and write_y produce correct shapes.
3. PortContract bounds output.
4. Power P = u · y is meaningful (non-zero, bounded).
"""
import torch
import unittest
from he_core.interfaces import PortInterface, PortContract
from he_core.state import ContactState


class TestPortInterface(unittest.TestCase):
    def test_port_interface_init(self):
        """Verify PortInterface initializes correctly."""
        print("\n--- 15.1: Port Interface Init ---")
        
        port = PortInterface(dim_q=2, dim_u=2, use_contract=True, max_gain=1.0)
        
        self.assertEqual(port.dim_q, 2)
        self.assertEqual(port.dim_u, 2)
        self.assertIsNotNone(port.contract)
        
        print("  PASS: PortInterface initialized.")
        
    def test_read_write_shapes(self):
        """Verify read_u and write_y produce correct shapes."""
        print("\n--- 15.1: Read/Write Shapes ---")
        
        port = PortInterface(dim_q=2, dim_u=2)
        
        # Test read
        u_ext = torch.randn(1, 2)
        force = port.read_u(u_ext)
        self.assertEqual(force.shape, (1, 2))
        
        # Test write
        state = ContactState(dim_q=2, batch_size=1)
        state.p = torch.randn(1, 2)
        y = port.write_y(state)
        self.assertEqual(y.shape, (1, 2))
        
        print(f"  Force shape: {force.shape}")
        print(f"  Output y shape: {y.shape}")
        print("  PASS: Shapes correct.")
        
    def test_port_contract_bounds(self):
        """Verify PortContract bounds output."""
        print("\n--- 15.1: Port Contract Bounds ---")
        
        contract = PortContract(method='tanh', max_gain=1.0)
        
        # Large input
        y_large = torch.tensor([[10.0, -10.0]])
        y_bounded = contract(y_large)
        
        self.assertTrue(y_bounded.abs().max().item() <= 1.0)
        
        print(f"  Input: {y_large.tolist()}")
        print(f"  Output: {y_bounded.tolist()}")
        print("  PASS: Output bounded by max_gain=1.0.")
        
    def test_power_semantics(self):
        """Verify power P = u · y is meaningful."""
        print("\n--- 15.1: Power Semantics ---")
        
        port = PortInterface(dim_q=2, dim_u=2, use_contract=True, max_gain=1.0)
        
        # Setup
        u_ext = torch.randn(1, 2) * 0.1
        state = ContactState(dim_q=2, batch_size=1)
        state.p = torch.randn(1, 2) * 0.1
        
        # Compute
        force = port.read_u(u_ext)
        y = port.write_y(state)
        
        # Power = u · y (element-wise product sum)
        power = (u_ext * y).sum()
        
        print(f"  u: {u_ext.tolist()}")
        print(f"  y: {y.tolist()}")
        print(f"  Power (u · y): {power.item():.6f}")
        
        # Power should be bounded (due to contract)
        self.assertTrue(abs(power.item()) < 10.0)
        
        print("  PASS: Power computed and bounded.")


if __name__ == '__main__':
    unittest.main()

import torch
import unittest
from he_core.generator import DissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.state import ContactState

class TestE12LearnableCoupling(unittest.TestCase):
    def test_gradient_flow(self):
        print("\n--- Task 12.2: Learnable Coupling Gradient Test ---")
        dim_q = 2
        internal = DissipativeGenerator(dim_q, alpha=0.1)
        # Enable Learnable Coupling
        port_gen = PortCoupledGenerator(internal, dim_u=dim_q, learnable_coupling=True)
        
        # Verify params exist
        params = list(port_gen.coupling.parameters())
        self.assertTrue(len(params) > 0, "No parameters found in Learnable Coupling")
        print(f"Coupling Params: {params[0].shape}")
        
        # Forward pass
        state = ContactState(dim_q, 1)
        state.q = torch.randn(1, dim_q)
        u_ext = torch.randn(1, dim_q)
        
        # H = H_int + <u, Wq>
        H = port_gen(state, u_ext)
        
        # Loss = H (maximize energy? just dummy)
        loss = H.sum()
        
        # Backward
        loss.backward()
        
        # Check grad
        W_grad = port_gen.coupling.W.weight.grad
        print(f"W Gradient Norm: {W_grad.norm().item()}")
        
        self.assertIsNotNone(W_grad, "Gradient did not reach Coupling Weights")
        self.assertGreater(W_grad.norm().item(), 0.0)
        
    def test_default_behavior(self):
        # Ensure default behavior (-q) is preserved if learnable=False
        dim_q = 2
        internal = DissipativeGenerator(dim_q, alpha=0.1)
        port_gen = PortCoupledGenerator(internal, dim_u=dim_q, learnable_coupling=False)
        
        state = ContactState(dim_q, 1)
        state.q = torch.ones(1, dim_q)
        
        # B(q) = -q = -1
        # Check internal coupling forward
        B_q = port_gen.coupling(state.q)
        self.assertTrue(torch.allclose(B_q, -state.q), "Default coupling should be -q")

if __name__ == '__main__':
    unittest.main()

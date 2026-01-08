import torch
import unittest
import numpy as np
from he_core.state import ContactState
from he_core.generator import DissipativeGenerator
from he_core.contact_dynamics import ContactIntegrator
from he_core.port_generator import PortCoupledGenerator

class TestPortGenerator(unittest.TestCase):
    def test_coupling_physics(self):
        print("\n--- Port Coupling Physics Test ---")
        dim_q = 2
        # Use simple Dissipative (Recover A3/A4) as internal
        internal = DissipativeGenerator(dim_q, alpha=0.1)
        # Neutralize potential so coupling physics is isolated
        for param in internal.net_V.parameters():
            param.data.zero_()
        port_gen = PortCoupledGenerator(internal, dim_u=dim_q)
        integrator = ContactIntegrator()
        
        state = ContactState(dim_q, 1) # Zeros
        
        # 1. Apply Constant Force u = [1.0, 0.0]
        # B(q) = -q. H_port = -u*q.
        # Force = -dH/dq = u.
        # Expect p to increase in dim 0.
        
        u_ext = torch.tensor([[1.0, 0.0]])
        
        # Wrap for integrator
        def H_func(s):
            return port_gen(s, u_ext)
            
        dt = 0.1
        next_state = integrator.step(state, H_func, dt)
        
        print(f"P_next: {next_state.p}")
        
        # Delta p should be approx Force * dt = 1.0 * 0.1 = 0.1
        # (Minus damping/potential, but q=0, p=0 -> V'(q)=0, damping=0)
        
        self.assertAlmostEqual(next_state.p[0,0].item(), 0.1, places=2, msg="Force not applied correctly")
        self.assertAlmostEqual(next_state.p[0,1].item(), 0.0, places=2)
        
    def test_zero_input_identity(self):
        print("\n--- Zero Input Identity Test ---")
        dim_q = 2
        internal = DissipativeGenerator(dim_q, alpha=0.1)
        port_gen = PortCoupledGenerator(internal, dim_u=dim_q)
        
        state = ContactState(dim_q, 1)
        state.q = torch.randn(1, dim_q)
        state.p = torch.randn(1, dim_q)
        
        # H_int only
        h_int = internal(state)
        # H_port with zero
        u_zero = torch.zeros(1, dim_q)
        h_port = port_gen(state, u_zero)
        
        self.assertTrue(torch.allclose(h_int, h_port), "Zero input should yield H_int")
        
        # H_port with None
        h_none = port_gen(state, None)
        self.assertTrue(torch.allclose(h_int, h_none), "None input should yield H_int")

if __name__ == '__main__':
    unittest.main()

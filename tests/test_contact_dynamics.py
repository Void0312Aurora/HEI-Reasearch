import torch
import unittest
import numpy as np
from he_core.state import ContactState
from he_core.contact_dynamics import ContactIntegrator
from he_core.generator import DissipativeGenerator

class TestContactDynamics(unittest.TestCase):
    def test_damping_behavior(self):
        dim_q = 2
        alpha = 0.5
        gen = DissipativeGenerator(dim_q, alpha)
        integrator = ContactIntegrator()
        
        # Init state with momentum
        state = ContactState(dim_q, 1)
        state.p = torch.ones(1, dim_q) * 1.0 # High momentum
        state.q = torch.zeros(1, dim_q)      # Origin
        
        # Step
        # H = 0.5 p^2 + V(q) + alpha s
        # dot_p = - (dH/dq + p * dH/ds)
        # dH/dq = dV/dq. dH/ds = alpha.
        # dot_p = - (Force + p * alpha).
        # Should see decay force -p*alpha
        
        p_init = state.p.clone()
        
        # Run 1 step
        new_state = integrator.step(state, gen, dt=0.1)
        
        p_next = new_state.p
        
        # Expected Change from damping: - p * alpha * dt = -1.0 * 0.5 * 0.1 = -0.05
        # dV/dq term is small/random (Tanh near 0).
        
        # Check that p decreased (magnitude)
        p_norm_init = p_init.norm()
        p_norm_next = p_next.norm()
        
        print(f"P_init: {p_norm_init}, P_next: {p_norm_next}")
        # Relaxed check: significant decay or close to expected
        self.assertLess(p_norm_next, p_norm_init, "Momentum did not decay in Dissipative Generator")

    def test_dimensions(self):
        dim_q = 3
        bs = 5
        state = ContactState(dim_q, bs)
        gen = DissipativeGenerator(dim_q)
        integrator = ContactIntegrator()
        
        new_state = integrator.step(state, gen, dt=0.01)
        self.assertEqual(new_state.q.shape, (bs, dim_q))
        self.assertEqual(new_state.p.shape, (bs, dim_q))
        self.assertEqual(new_state.s.shape, (bs, 1))

if __name__ == '__main__':
    unittest.main()

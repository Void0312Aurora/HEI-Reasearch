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

    def test_semi_implicit_matches_linear_contact(self):
        """
        Semi-implicit integrator should match the analytic solution for the
        linear contact system:
          H = 0.5||p||^2 + alpha*s
        which implies:
          p(t+dt) = exp(-alpha dt) p(t)
          q(t+dt) = q(t) + (1-exp(-alpha dt))/alpha * p(t)
        """
        class LinearContact(torch.nn.Module):
            def __init__(self, dim_q: int, alpha: float):
                super().__init__()
                self.dim_q = dim_q
                self.alpha = float(alpha)

            def forward(self, state: ContactState) -> torch.Tensor:
                p = state.p
                s = state.s
                K = 0.5 * (p ** 2).sum(dim=1, keepdim=True)
                return K + self.alpha * s

        dim_q = 4
        bs = 8
        alpha = 50.0  # stiff damping (Euler would be unstable for dt=0.1)
        dt = 0.1

        gen = LinearContact(dim_q, alpha)
        integrator = ContactIntegrator(method="semi")

        state = ContactState(dim_q, bs)
        state.q = torch.randn(bs, dim_q) * 0.3
        state.p = torch.randn(bs, dim_q)
        state.s = torch.zeros(bs, 1)

        q0 = state.q.clone()
        p0 = state.p.clone()

        new_state = integrator.step(state, gen, dt=dt)

        decay = float(torch.exp(torch.tensor(-alpha * dt)))
        p_expected = p0 * decay
        q_expected = q0 + ((1.0 - decay) / alpha) * p0

        self.assertTrue(torch.allclose(new_state.p, p_expected, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(new_state.q, q_expected, atol=1e-4, rtol=1e-4))

if __name__ == '__main__':
    unittest.main()

import torch
import unittest
import numpy as np
from he_core.state import ContactState
from he_core.contact_dynamics import ContactIntegrator
from he_core.generator import DissipativeGenerator
from EXP.diag.fastslow import compute_jacobian_spectrum, report_spectral_gap

class TestFastSlow(unittest.TestCase):
    def test_spectral_gap(self):
        print("\n--- Fast-Slow Structure Test ---")
        dim_q = 2
        # Overdamped parameters
        # Damping alpha = 10.0
        # Stiffness (gradient of V) ~ 1.0 (if q small, Tanh linear part has slope 1*weights?)
        # Let's verify Potential slope.
        # V = Linear(16) -> Tanh -> Linear(1).
        # Init weights are usually uniform/kaiming.
        
        # We can construct a simple Quadratic Potential generator for control?
        # Or just trust DissipativeGenerator randomness but scale alpha high.
        
        alpha = 10.0
        gen = DissipativeGenerator(dim_q, alpha)
        integrator = ContactIntegrator()
        
        state = ContactState(dim_q, 1) # Zeros
        dt = 0.01 # Small dt for stability
        
        # Spectrum
        spec = compute_jacobian_spectrum(integrator, gen, state, dt)
        
        gap = report_spectral_gap(spec)
        
        # Expect at least one gap > 2.0 (Alpha is 10, K/Alpha is small)
        self.assertGreater(gap, 2.0, "Failed to detect significant Spectral Gap in Overdamped system")
        
        # Check that we have slow modes (near 0) and fast modes (near -10)
        real_parts = spec.real
        slowest = real_parts[0] # Should be close to 0 ( e.g. -0.1)
        fastest = real_parts[-1] # Should be close to -alpha (-10)
        
        print(f"Slowest: {slowest}, Fastest: {fastest}")
        
        self.assertGreater(slowest, -2.0, "Slow mode too fast")
        self.assertLess(fastest, -8.0, "Fast mode too slow (Damping missing?)")

if __name__ == '__main__':
    unittest.main()

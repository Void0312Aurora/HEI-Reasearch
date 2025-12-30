import torch
import unittest
import numpy as np
from he_core.entity_v3 import GeometricEntity
from he_core.contact_dynamics import ContactIntegrator
from he_core.port_generator import PortCoupledGenerator
from he_core.generator import DissipativeGenerator
from he_core.state import ContactState
from EXP.diag.spectral_gap import compute_spectral_properties, report_spectral_gap
from EXP.diag.port_gain import compute_port_gain

class TestE11Diagnostics(unittest.TestCase):
    def test_spectral_protocol(self):
        print("\n--- Protocol 1: Spectral Gap ---")
        dim_q = 2
        # Overdamped Internal Generator
        internal = DissipativeGenerator(dim_q, alpha=10.0)
        port_gen = PortCoupledGenerator(internal, dim_q)
        integrator = ContactIntegrator()
        
        state = ContactState(dim_q, 1)
        
        # Generator wrapper for zero input
        def H_func(s):
            return port_gen(s, None)
            
        metrics = compute_spectral_properties(integrator, H_func, state, dt=0.01)
        passed = report_spectral_gap(metrics, threshold=2.0)
        
        self.assertTrue(passed, "Spectral Gap Protocol failed on Stiff System")
        self.assertGreater(metrics['max_gap'], 5.0)

    def test_port_gain_protocol(self):
        print("\n--- Protocol 5: Port Gain ---")
        config = {'dim_q': 2, 'dim_ext': 2, 'damping': 0.5}
        entity = GeometricEntity(config)
        
        # Mock Input
        u_in = torch.randn(1, 1, 2)
        
        gain = compute_port_gain(entity, u_in)
        
        # For a dissipative system with 0.5 damping, gain shouldn't be massive.
        # It maps u -> Force -> p.
        # Force = u. p_new = p_old + (u - gamma*p)*dt.
        # If p_old=0, p_new = u*dt.
        # Active out = Linear(p).
        # So Out ~ u * dt. Gain ~ dt (0.1).
        
        print(f"Measured Gain: {gain}")
        self.assertLess(gain, 2.0, "Port Gain too high (Instability Risk)")

if __name__ == '__main__':
    unittest.main()

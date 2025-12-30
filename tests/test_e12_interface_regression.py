import unittest
import torch
import numpy as np
from he_core.entity_v3 import GeometricEntity
from he_core.interfaces import ActionInterface, AuxInterface
from EXP.run_phase12 import run_experiment
# We need to adapt run_experiment to accept pre-built Entity or Config with Interface
# Currently run_experiment builds Entity manually.
# We will verify Interfaces using GeometricEntity directly in this test.

from he_core.generator import DissipativeGenerator
from he_core.port_generator import PortCoupledGenerator
from he_core.contact_dynamics import ContactIntegrator
from EXP.diag.spectral_gap import compute_spectral_properties
from he_core.state import ContactState

class TestE12InterfaceRegression(unittest.TestCase):
    def test_interface_invariance(self):
        print("\n--- Task 12.4: Interface Regression ---")
        dim_q = 2
        dim_ext = 2
        
        # 1. Action Interface
        config_action = {
            'dim_q': dim_q,
            'dim_ext': dim_ext,
            'interface_type': 'action'
        }
        entity_action = GeometricEntity(config_action)
        # Verify it has ActionInterface
        self.assertIsInstance(entity_action.interface, ActionInterface)
        
        # 2. Aux Interface (Swap in place? or new entity)
        config_aux = {
            'dim_q': dim_q,
            'dim_ext': dim_ext,
            'interface_type': 'aux' # logic inside GeometricEntity needed for this config
        }
        # GeometricEntity currently defaults to ActionInterface.
        # We manually swap.
        entity_aux = GeometricEntity(config_action)
        entity_aux.interface = AuxInterface(dim_q, dim_ext)
        
        # 3. Check Protocol 1 (Internal Dynamics) on both
        # The internal generator is same.
        # Spectral Gap should be identical (invariant).
        
        st = ContactState(dim_q, 1)
        
        # Extract generator from entity
        # Entity wraps generator.
        gen = entity_action.generator 
        
        def H_func(s): return gen(s)
        
        metrics_1 = compute_spectral_properties(entity_action.integrator, H_func, st)
        metrics_2 = compute_spectral_properties(entity_aux.integrator, H_func, st)
        
        print(f"Gap Action: {metrics_1['max_gap']}")
        print(f"Gap Aux: {metrics_2['max_gap']}")
        
        self.assertAlmostEqual(metrics_1['max_gap'], metrics_2['max_gap'], msg="Internal Dynamics should be invariant to Interface")
        
        # 4. Check Step Output
        # ActionInterface: active = p
        # AuxInterface: active = q
        
        entity_action.state.q.fill_(0.0)
        entity_action.state.p.fill_(1.0)
        out_action = entity_action.interfaces['action'].project_out(entity_action.state)
        # Should be p-like
        
        entity_aux.state.q.fill_(1.0) # q=1
        entity_aux.state.p.fill_(0.0)
        out_aux = entity_aux.interface.project_out(entity_aux.state) # Manually swapped
        # Should be q-like (1.0)
        
        print(f"Out Action (p=1): {out_action}")
        print(f"Out Aux (q=1): {out_aux}")
        
        self.assertFalse(torch.allclose(out_action, out_aux), "Interfaces should produce different projections")

if __name__ == '__main__':
    unittest.main()

import sys
import os
import unittest
import torch
import numpy as np

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from he_core.entity import Entity
from he_core.kernel.kernels import PlasticKernel

class TestSpecCompliance(unittest.TestCase):
    def setUp(self):
        self.config = {
            'kernel_type': 'plastic',
            'dim_q': 2,
            'active_mode': True,
            'active_gain': 0.1,
            'seed': 42
        }
        self.entity = Entity(self.config)
        
    def test_log_contract(self):
        """
        Verify that Entity.step() returns all required fields for SPEC v0.2.
        Required: action, x_int, x_blanket, sensory, active, pred_error, pred_x.
        """
        # Mock Observation
        obs = {'x_ext_proxy': [1.0, 2.0]}
        
        # Step
        out = self.entity.step(obs)
        
        # 1. Check Keys
        required_keys = [
            'action', 'x_int', 'x_blanket', 'u_t', 'pred_x', 'meta',
            'sensory', 'active', 'pred_error'
        ]
        
        for k in required_keys:
            self.assertIn(k, out, f"Missing key: {k}")
            
        # 2. Check Shapes/Types
        self.assertIsInstance(out['action'], np.ndarray)
        self.assertIsInstance(out['x_int'], np.ndarray)
        self.assertIsInstance(out['pred_x'], np.ndarray)
        
        # 3. Check Specific Values (weak check)
        # sensory (u_env) is in x_blanket?
        # x_blanket = [u_env, u_self]
        # x_blanket dim should be 2 * dim_q
        self.assertEqual(out['x_blanket'].shape[1], 2 * self.config['dim_q'])
        
        # 4. Check internal consistency
        # active (action) should be part of blanket?
        # Actually action is flattened u_self.
        pass

if __name__ == '__main__':
    unittest.main()

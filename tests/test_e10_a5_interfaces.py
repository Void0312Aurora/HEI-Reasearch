import torch
import unittest
import numpy as np
from he_core.entity_v3 import GeometricEntity
from he_core.interfaces import ActionInterface, AuxInterface

class TestE10A5(unittest.TestCase):
    def test_interface_swapping(self):
        print("\n--- A5 Interface Swapping Test ---")
        config = {'dim_q': 2, 'dim_ext': 2, 'damping': 0.1}
        entity = GeometricEntity(config)
        
        # 1. Run with Default (ActionInterface)
        obs = {'x_ext_proxy': np.array([1.0, 0.5])}
        out_1 = entity.step(obs)
        print(f"ActionInterface Out: p_norm={out_1['p_norm']}, active={out_1['active']}")
        
        # 2. Swap to AuxInterface
        aux = AuxInterface(dim_q=2, dim_ext=2)
        entity.set_interface(aux)
        
        out_2 = entity.step(obs)
        print(f"AuxInterface Out: p_norm={out_2['p_norm']}, active={out_2['active']}")
        
        # Check that outputs are valid
        self.assertIsNotNone(out_2['active'])
        self.assertNotEqual(out_1['active'][0], out_2['active'][0], "Interfaces should produce different projections")
        
        # Check State Continuity (p_norm should not explode)
        self.assertLess(out_2['p_norm'], 10.0, "State exploded after interface swap")

if __name__ == '__main__':
    unittest.main()

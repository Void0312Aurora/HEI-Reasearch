"""
Gate Test for L1 Dynamics (Dissipation).
Enforces strict energy decay under null potential conditions.
Configuration: Damping=5.0, V(q)=0.
"""
import unittest
import os
import sys

import torch
import numpy as np

# Ensure HEI is on path for `he_core` imports when running `pytest` from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from he_core.entity_v4 import UnifiedGeometricEntity

class ZeroModule(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1, device=x.device)

class TestL1Gate(unittest.TestCase):
    def test_dissipation_strict(self):
        print("\n=== L1 Gate: Strict Dissipation Check ===")
        config = {
            'dim_q': 2,
            'learnable_coupling': False,
            'num_charts': 1,
            'damping': 5.0, # Strong damping
            'use_port_interface': False
        }
        entity = UnifiedGeometricEntity(config)
        
        # Patch Potential to Zero
        entity.internal_gen.net_V = ZeroModule()
        
        # Initialize High Velocity
        entity.reset()
        entity.state.p = torch.randn(1, 2) * 2.0
        
        # Track Kinetic Energy
        energies = []
        dt = 0.05
        steps = 50
        
        for _ in range(steps):
            K = 0.5 * (entity.state.p**2).sum().item()
            energies.append(K)
            
            # Step
            u_zero = torch.zeros(1, entity.dim_u)
            out = entity.forward_tensor(entity.state.flat, u_zero, dt)
            entity.state.flat = out['next_state_flat'].detach()
            
        energies = np.array(energies)
        
        # Check 1: Monotonicity (allowing numerical noise epsilon)
        epsilon = 1e-5
        diffs = energies[1:] - energies[:-1]
        violations = np.sum(diffs > epsilon)
        
        print(f"  Init K: {energies[0]:.4f}")
        print(f"  Final K: {energies[-1]:.4f}")
        print(f"  Violations: {violations}")
        
        self.assertEqual(violations, 0, f"Energy increased in {violations} steps!")
        
        # Check 2: Trend
        slope = (energies[-1] - energies[0]) / (steps * dt)
        print(f"  Slope: {slope:.6f}")
        self.assertLess(slope, -0.01, "Energy did not decay significantly with high damping!")
        
        print("  PASS: System is strictly dissipative.")

if __name__ == '__main__':
    unittest.main()

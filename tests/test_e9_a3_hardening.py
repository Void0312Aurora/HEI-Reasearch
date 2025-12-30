import sys
import os
import unittest
import numpy as np
import torch

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from he_core.entity import Entity

class RandomKernel:
    def __init__(self, dim_q):
        self.dim_q = dim_q
    
    def forward(self, x, u):
        # Random Walk
        return x + torch.randn_like(x) * 0.1

    def __call__(self, x, u):
        return self.forward(x, u)

class TestA3Hardening(unittest.TestCase):
    def setUp(self):
        # We need a learnable entity to show F reduction
        self.config = {
            'dim_q': 2,
            'kernel_type': 'plastic',
            'active_mode': True,
            'learning_rule': 'delta', # Learning enabled
            'active_gain': 0.1,
            'eta': 0.05 # Fast learning for test
        }
        
    def run_phase(self, entity, steps=100, mode='online'):
        errors = []
        u_env = np.zeros(2)
        
        # If offline, we stop external input driving?
        # But for A3 we want to see if internal dynamics minimize error?
        # A3 A2 says: Offline = Low stimulus.
        # But we verify "Same Functional".
        
        for t in range(steps):
            if mode == 'online':
                u_env += np.random.randn(2) * 0.1
                obs = {'x_ext_proxy': u_env}
            else:
                # Offline: Low stimulus (zero input)
                obs = {'x_ext_proxy': np.zeros(2)}
            
            # Step
            out = entity.step(obs)
            
            # Metric: Prediction Error Squared
            # F = |u - pred|^2?
            # In offline, u=0. So F = |0 - pred|^2 = |pred|^2.
            # Does system minimize its own activity (energy minimization)?
            # Yes, PlasticKernel with decay mins energy.
            
            # But we want to distinguish from "Random".
            # If RandomKernel acts, |pred|^2 (energy) might grow (Random Walk).
            # PlasticKernel (Dissipative) should reduce |pred|^2.
            
            # Actually, let's use the explicit 'pred_error' if available, or Energy.
            if mode == 'online':
                err = np.linalg.norm(out['pred_error'])**2
            else:
                # In offline, "pred_error" is effectively State Energy/Prediction magnitude
                # A3 says "Same Functional".
                # If F = Free Energy = Accuracy + Complexity.
                # Offline: Accuracy (u=0) -> Minimize Prediction.
                err = np.linalg.norm(out['pred_x'])**2
                
            errors.append(err)
            
        return errors

    def test_a3_consistency(self):
        print("\n--- A3 Hardening Test ---")
        
        # 1. Standard (PASS)
        # Should show decreasing Energy/Error in Offline
        entity = Entity(self.config)
        
        # Pre-heat
        self.run_phase(entity, 50, 'online')
        
        # Offline Phase
        f_std = self.run_phase(entity, 50, 'offline')
        
        # Slope check
        slope_std = np.polyfit(range(len(f_std)), f_std, 1)[0]
        print(f"Standard Offline Slope: {slope_std:.6f}")
        
        # 2. Paradigm Switch (FAIL)
        # Swap Kernel to Random Walk for Offline
        entity_bad = Entity(self.config)
        self.run_phase(entity_bad, 50, 'online')
        
        # SWAP
        entity_bad.kernel = RandomKernel(self.config['dim_q'])
        
        f_bad = self.run_phase(entity_bad, 50, 'offline')
        slope_bad = np.polyfit(range(len(f_bad)), f_bad, 1)[0]
        print(f"Swapped Offline Slope: {slope_bad:.6f}")
        
        # Assertions
        # Standard: Dissipative/Minimizing -> Slope < 0 (or close to 0 if converged)
        # Random: Random Walk -> Energy grows -> Slope > 0 usually
        
        self.assertLess(slope_std, 1e-4, "Standard Entity not minimizing F in offline")
        
        # Swapped should be significantly worse (growing or flat-high)
        # Note: If pred_x is bounded (tanh), energy might saturate, so slope is 0.
        # But Standard slope is negative. So bad > std holds.
        self.assertGreater(slope_bad, slope_std + 1e-6, "Paradigm Switch not detected (Slopes similar)")
        
if __name__ == '__main__':
    unittest.main()

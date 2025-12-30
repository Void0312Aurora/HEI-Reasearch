import sys
import os
import unittest
import numpy as np
import torch

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from he_core.entity import Entity

class TestA4Hardening(unittest.TestCase):
    def setUp(self):
        # Standard Config (Damping=0.5 for clear recovery in short test)
        self.config_std = {
            'dim_q': 2,
            'kernel_type': 'plastic', 
            'active_mode': False,
            'damping': 0.5,
            'active_gain': 0.1
        }
        
        # FAIL Config (Damping=0.0) -> Conservative / Non-recovering
        self.config_fail = self.config_std.copy()
        self.config_fail['damping'] = 0.0

    def run_perturbation_test(self, entity, steps=100, perturb_t=20, impulse=1.0):
        # Stub
        pass

    def measure_recovery(self, config, label=""):
        # 1. Baseline Trace
        entity_base = Entity(config)
        # Set Initial State to Origin (Likely Fixed Point or near one)
        entity_base.x_int = torch.zeros_like(entity_base.x_int)
        
        traj_base = []
        steps = 300 # Long run to settle
        perturb_t = 20
        impulse = 0.5
        
        for t in range(steps):
            obs = {'x_ext_proxy': np.zeros(2)}
            out = entity_base.step(obs)
            traj_base.append(out['x_int'].flatten())
            
        # 2. Perturbed Trace
        entity_pert = Entity(config)
        entity_pert.x_int = torch.zeros_like(entity_pert.x_int)
        
        traj_pert = []
        # Use same steps and perturb_t
        
        for t in range(steps):
            obs = {'x_ext_proxy': np.zeros(2)}
            
            # Apply Perturbation
            if t == perturb_t:
                # Add impulse manually before step?
                # Or step() then modify?
                # Entity.x_int is state.
                with torch.no_grad():
                    entity_pert.x_int += impulse
            
            out = entity_pert.step(obs)
            traj_pert.append(out['x_int'].flatten())

        # 3. Compute Deviation on Momentum (Kinetic Energy Proxy)
        # x_int = [q, p, s, ...]
        # We assume dim_q = 2 from config
        dim_q = config['dim_q']
        
        # p is indices [dim_q : 2*dim_q]
        # traj arrays are [Steps, DimTotal]
        traj_base = np.array(traj_base)
        traj_pert = np.array(traj_pert)
        
        p_base = traj_base[:, dim_q : 2*dim_q]
        p_pert = traj_pert[:, dim_q : 2*dim_q]
        
        # Kinetic Energy = 0.5 * p^2
        k_base = 0.5 * np.sum(p_base**2, axis=1)
        k_pert = 0.5 * np.sum(p_pert**2, axis=1)
        
        # Difference in Energy
        dists = np.abs(k_pert - k_base)
        
        # Use Mean of last 20 steps to avoid oscillation aliasing
        window = 20
        final_dist = np.mean(dists[-window:])
        max_dist = np.max(dists)
        
        recovery_ratio = final_dist / (max_dist + 1e-9)
        
        print(f"[{label}] Max Delta E_k: {max_dist:.6f}, Final Mean E_k: {final_dist:.6f}, Ratio: {recovery_ratio:.4f}")
        return recovery_ratio

    def test_a4_identity_recovery(self):
        print("\n--- A4 Hardening Test (Identity Recovery) ---")
        
        # 1. Standard (Should recover due to damping)
        ratio_std = self.measure_recovery(self.config_std, "Standard")
        
        # 2. No Damping (Control)
        # Numerical dissipation might cause some decay, but should be much slower than Standard
        ratio_fail = self.measure_recovery(self.config_fail, "No Damping")
        
        # Assertions
        # Standard should decay significantly (Ratio < 0.1)
        self.assertLess(ratio_std, 0.1, "Standard Entity failed to recover (Ratio too high)")
        
        # No Damping should retain more energy than Standard
        # We enforce a separation margin
        self.assertGreater(ratio_fail, ratio_std * 2.0, "No-Damping Entity decayed as fast as Standard (Dissipative Indistinguishable)")

if __name__ == '__main__':
    unittest.main()

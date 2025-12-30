import sys
import os
import unittest
import numpy as np

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from EXP.experiments.run_entity import run_experiment_entity

class TestE4Ablation(unittest.TestCase):
    """
    Phase 5.3: W-Ablation.
    Distinguish between "PID Gain Utility" and "Structural Utility".
    Compare:
    1. Active (Normal Learning)
    2. Active (W Fixed at Zero) - "Lobotomized"
    """
    def setUp(self):
        self.config = {
            'seed': 42,
            'dim_q': 2,
            'kernel_type': 'plastic', # Default
            'dt': 0.1,
            'omega': 1.0,
            'eta': 0.05,
            'env_type': 'limit_cycle',
            'drive_amp': 1.0,
            'env_gamma': 0.1,
            'online_steps': 500,
            'offline_steps': 0,
            'max_steps': 500,
            'output_dir': 'tests/outputs_e4_ablation',
            'force_u_self_zero': False,
            'active_gain': 0.1,
            'active_mode': True
        }
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def test_structural_attribution(self):
        print("\nRunning E4 Ablation Test...")
        
        # 1. Normal Learning
        print("Running Learned W...")
        cfg_learn = self.config.copy()
        _, _, _, _, log_learn = run_experiment_entity(cfg_learn)
        var_learn = np.var(np.vstack(log_learn['x_ext_proxy']))
        
        # 2. Fixed Zero W
        # How to enforce W=0?
        # We can implement a "Fixed Plastic" kernel or just set ETA=0 and Init W=0?
        # Init W is 0 by default. So if eta=0, W stays 0.
        print("Running Zero W (Lobotomized)...")
        cfg_zero = self.config.copy()
        cfg_zero['eta'] = 0.0
        _, _, _, _, log_zero = run_experiment_entity(cfg_zero)
        var_zero = np.var(np.vstack(log_zero['x_ext_proxy']))
        
        print(f"Learned W Variance: {var_learn:.4f}")
        print(f"Zero W Variance:    {var_zero:.4f}")
        
        # Interpretation
        # If Learned < Zero, then Structure helps Stabilization.
        # If Learned > Zero, then Structure allows Dynamics (Relaxation).
        
        ratio = var_learn / var_zero
        print(f"Ratio (Learned/Zero): {ratio:.4f}")
        
        # Check Weights in Learned
        w_norm = self._get_final_weight_norm(log_learn)
        print(f"Learned Weight Norm: {w_norm:.4f}")
        
        # Assertion: Structure does *something* statistically significant
        self.assertNotAlmostEqual(ratio, 1.0, delta=0.01, msg="Structure had no effect vs PID Baseline")

    def _get_final_weight_norm(self, log):
        dim_q = self.config['dim_q']
        x_int = np.array(log['x_int']).squeeze(1)
        w_start = 4 * dim_q + 1
        return np.linalg.norm(x_int[-1, w_start:])

if __name__ == "__main__":
    unittest.main()

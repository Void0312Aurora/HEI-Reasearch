import sys
import os
import unittest
import numpy as np

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from EXP.experiments.run_entity import run_experiment_entity

class TestE4Utility(unittest.TestCase):
    """
    Phase 5: E4 Utility Gate.
    Verifies that Active Inference (using learned W) reduces Prediction Error / State Deviation
    compared to a Passive Baseline.
    """
    def setUp(self):
        self.config = {
            'seed': 42,
            'dim_q': 2,
            'kernel_type': 'plastic',
            'dt': 0.1,
            'omega': 1.0,
            'eta': 0.05,
            'env_type': 'limit_cycle',
            'drive_amp': 1.0, 
            'env_gamma': 0.1,
            'online_steps': 500,
            'offline_steps': 0,
            'max_steps': 500,
            'output_dir': 'tests/outputs_e4',
            'force_u_self_zero': False,
            'active_gain': 1.0
        }
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def test_active_vs_passive(self):
        print("\nRunning E4 Utility Test...")
        
        # 1. Passive Run (active_mode = False)
        cfg_pass = self.config.copy()
        cfg_pass['active_mode'] = False
        _, _, _, _, log_pass = run_experiment_entity(cfg_pass)
        
        # 2. Active Run (active_mode = True)
        cfg_act = self.config.copy()
        cfg_act['active_mode'] = True
        _, _, _, _, log_act = run_experiment_entity(cfg_act)
        
        # Metrics: Prediction Error proxy
        # Since we want to show 'better fitting' or 'stabilization'.
        # Let's measure the "Input Enegy" seen by the internal state?
        # A good regulator minimizes the entropy of the sensation (Friston).
        # So effective u_t energy should be lower? OR x_ext variance?
        
        # u_t = u_env (plus u_self effect on next step).
        # But u_self acts on Env.
        # If u_self cancels disturbance, x_ext should be closer to "Expected".
        # But here expectation is learned from x_ext...
        
        # Let's measure variance of x_ext (Env State).
        # If Active Agent stabilizes the environment, x_ext variance might decrease.
        x_ext_pass = np.vstack(log_pass['x_ext_proxy'])
        x_ext_act = np.vstack(log_act['x_ext_proxy'])
        
        var_pass = np.var(x_ext_pass)
        var_act = np.var(x_ext_act)
        
        # Also measure "Structure Learning" (Weight Max)
        # Active agent should also learn weights.
        
        print(f"Passive Env Variance: {var_pass:.4f}")
        print(f"Active Env Variance:  {var_act:.4f}")
        
        # We expect Active Agent (trying to minimize error) to potentially stabilize 
        # or at least change dynamics significantly.
        # For E4, let's assert DIFFERENCE first, and Variance Reduction if Task is stabilization.
        # LimitCycle is driven. If we cancel drive, variance drops.
        
        ratio = var_act / var_pass
        print(f"Variance Ratio (Act/Pass): {ratio:.4f}")
        
        # Gate: Significant effect
        # Hardened E4 Gate: Active Control must reduce variance by at least 40% (Ratio < 0.6)
        # Empirical result was ~0.25 (75% reduction)
        self.assertLess(ratio, 0.6, f"Active Inference failed to stabilize environment significantly (Ratio={ratio:.2f})")
        
        # Check Weights
        w_act = self._get_final_weight_norm(log_act)
        print(f"Active Weight Norm: {w_act:.4f}")
        self.assertGreater(w_act, 0.1, "Weights not learned in Active Mode")

    def _get_final_weight_norm(self, log):
        dim_q = self.config['dim_q']
        x_int = np.array(log['x_int']).squeeze(1)
        w_start = 4 * dim_q + 1
        return np.linalg.norm(x_int[-1, w_start:])

if __name__ == "__main__":
    unittest.main()

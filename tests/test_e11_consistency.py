import unittest
import numpy as np
from EXP.diag.consistency import compute_consistency_drift, report_consistency

class TestE11Consistency(unittest.TestCase):
    def test_dissipative_trajectory(self):
        print("\n--- Protocol A3: Dissipative Test ---")
        # Mock Dissipative: s[t+1] = s[t] - 0.1
        T = 50
        s_val = 10.0
        traj = []
        for _ in range(T):
            traj.append({'s_val': s_val})
            s_val -= 0.1
            
        metrics = compute_consistency_drift(traj)
        passed = report_consistency(metrics)
        
        self.assertTrue(passed)
        self.assertEqual(metrics['status'], "DISSIPATIVE")
        self.assertAlmostEqual(metrics['drift_mean'], -0.1)

    def test_random_walk_trajectory(self):
        print("\n--- Protocol A3: Random Walk Test ---")
        # Mock Random Walk: s[t+1] = s[t] + N(0, 1)
        np.random.seed(42)
        s_val = 0.0
        traj = []
        for _ in range(100):
            traj.append({'s_val': s_val})
            s_val += np.random.randn()
            
        metrics = compute_consistency_drift(traj)
        # Random Walk should fail "STABLE" check (std > 0.1) and fail "DISSIPATIVE" (mean ~ 0)
        # Unless random seed makes mean negative.
        
        print(f"Random Walk Mean: {metrics['drift_mean']}, Std: {metrics['drift_std']}")
        
        # It should likely be RANDOM_WALK (mean approx 0, std high)
        # OR RUNAWAY if mean drifts positive.
        
        passed = report_consistency(metrics)
        
        # We expect FAIL for Random Walk
        self.assertFalse(passed, "Random Walk should not pass as Structured Drive")
        self.assertEqual(metrics['status'], "RANDOM_WALK")

if __name__ == '__main__':
    unittest.main()

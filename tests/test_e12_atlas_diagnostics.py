import unittest
import numpy as np
from EXP.diag.atlas_metrics import compute_atlas_metrics, report_atlas_metrics

class TestE12AtlasDiagnostics(unittest.TestCase):
    def test_coverage_and_switching(self):
        print("\n--- Task 12.3: Atlas Metrics Test ---")
        # Synthetic Trajectory
        # T=100.
        # t=0..50: Chart 0 Dominant
        # t=50..100: Chart 1 Dominant
        
        traj = []
        for t in range(100):
            if t < 50:
                w = [0.9, 0.1]
                loss = 0.01
            else:
                w = [0.1, 0.9]
                loss = 0.02
                
            traj.append({
                'chart_weights': w,
                'consistency_loss': loss
            })
            
        metrics = compute_atlas_metrics(traj)
        report_atlas_metrics(metrics)
        
        # Expect coverage approx [0.5, 0.5]
        cov = metrics['coverage_distribution']
        self.assertAlmostEqual(cov[0], 0.5, delta=0.1)
        self.assertAlmostEqual(cov[1], 0.5, delta=0.1)
        
        # Expect 1 switch
        # t=49 (0), t=50 (1) -> Switch
        self.assertEqual(metrics['switch_count'], 1)
        
        # Mean loss approx 0.015
        self.assertAlmostEqual(metrics['mean_consistency_loss'], 0.015, delta=0.005)

if __name__ == '__main__':
    unittest.main()

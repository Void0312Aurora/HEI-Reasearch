import unittest
import numpy as np
from EXP.diag.dmd import compute_dmd, report_dmd_analysis

class TestE11DMD(unittest.TestCase):
    def test_dmd_synthetic(self):
        print("\n--- Protocol 3: DMD Synthetic Test ---")
        # Synthetic Data: Slow mode (0.99) + Fast mode (0.5)
        T = 100
        t = np.arange(T)
        
        # Mode 1: Slow decay
        # x1 = exp(-0.01 t) ~ 0.99^t
        v1 = np.array([1.0, 0.0])
        x1 = np.outer(v1, 0.99**t) # (2, T)
        
        # Mode 2: Fast decay
        # x2 = 0.5^t
        v2 = np.array([0.0, 1.0])
        x2 = np.outer(v2, 0.5**t)
        
        X = x1 + x2
        
        metrics = compute_dmd(X, r=2)
        passed = report_dmd_analysis(metrics)
        
        self.assertTrue(passed, "DMD failed to detect slow mode")
        self.assertGreaterEqual(metrics['slow_mode_count'], 1)
        
        # Check eigenvalues directly
        eigs = metrics['eigenvalues']
        mags = np.abs(eigs)
        print(f"Eigenvalues: {mags}")
        
        # Should find ~0.99 and ~0.5
        has_slow = np.any((mags > 0.98) & (mags < 1.0))
        has_fast = np.any((mags > 0.4) & (mags < 0.6))
        
        self.assertTrue(has_slow, "Specific slow eigenvalue approx 0.99 not found")
        self.assertTrue(has_fast, "Specific fast eigenvalue approx 0.5 not found")

if __name__ == '__main__':
    unittest.main()

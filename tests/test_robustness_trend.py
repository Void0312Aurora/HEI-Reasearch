"""
Verify Robustness Trend Monitor (Phase 16.3).

Simulates a sequence of robustness values (degrading vs improving)
and checks if the monitor correctly identifies the trend.
"""
import unittest
import numpy as np
from EXP.diag.network_monitors import RobustnessTrendMonitor

class TestRobustnessTrend(unittest.TestCase):
    def test_improving_trend(self):
        print("\n--- 16.3: Improving Trend ---")
        monitor = RobustnessTrendMonitor(window_size=10)
        
        # Simulate improving robustness: 0.1, 0.2, ...
        for i in range(20):
            monitor.add_sample(0.1 * i + np.random.normal(0, 0.01))
            
        stats = monitor.get_trend()
        print(f"  Slope: {stats['slope']:.4f} (Expected ~0.1)")
        print(f"  Current: {stats['current']:.4f}")
        
        self.assertTrue(stats['slope'] > 0.05)
        print("  PASS: Detected improving trend.")
        
    def test_degrading_trend(self):
        print("\n--- 16.3: Degrading Trend ---")
        monitor = RobustnessTrendMonitor(window_size=10)
        
        # Simulate degrading robustness: 5.0, 4.9, ...
        for i in range(20):
            monitor.add_sample(5.0 - 0.1 * i + np.random.normal(0, 0.01))
            
        stats = monitor.get_trend()
        print(f"  Slope: {stats['slope']:.4f} (Expected ~ -0.1)")
        
        self.assertTrue(stats['slope'] < -0.05)
        print("  PASS: Detected degrading trend.")
        
    def test_stable_trend(self):
        print("\n--- 16.3: Stable Trend ---")
        monitor = RobustnessTrendMonitor(window_size=10)
        
        # Simulate stable robustness: 1.0 + noise
        for i in range(20):
            monitor.add_sample(1.0 + np.random.normal(0, 0.01))
            
        stats = monitor.get_trend()
        print(f"  Slope: {stats['slope']:.4f} (Expected ~0.0)")
        
        self.assertTrue(abs(stats['slope']) < 0.05)
        print("  PASS: Detected stable trend.")

if __name__ == '__main__':
    unittest.main()

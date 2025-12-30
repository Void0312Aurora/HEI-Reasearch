import unittest
import numpy as np
from EXP.diag.monitors import robustness_always, robustness_eventually

class TestE13STL(unittest.TestCase):
    def test_always_property(self):
        print("\n--- Task 13.4: STL Always Test ---")
        # Trace: x decreases from 10 to 0.
        trace = [{'val': 10 - i} for i in range(10)]
        
        # Spec: Always[0, 10] (val > -1)
        # Min val is 1 (at i=9). 1 > -1. Pred = val - (-1) = val + 1.
        # Robustness = min(val + 1) = 2.
        
        def pred_gz(x): return x['val'] + 1.0 # val > -1
        
        rho = robustness_always(trace, 0, 10, pred_gz)
        print(f"Robustness (Always > -1): {rho}")
        
        self.assertEqual(rho, 2.0)
        self.assertGreater(rho, 0.0) # Satisfied

    def test_eventually_property(self):
        print("\n--- Task 13.4: STL Eventually Test ---")
        # Trace: x stays 10, then dips to 2 at step 5.
        trace = [{'val': 10} for _ in range(10)]
        trace[5]['val'] = 2
        
        # Spec: Eventually[0, 10] (val < 5) -> (5 - val > 0)
        # Max of (5 - val).
        # At step 5, 5-2 = 3. Other steps 5-10 = -5.
        # Max is 3.
        
        def pred_small(x): return 5.0 - x['val']
        
        rho = robustness_eventually(trace, 0, 10, pred_small)
        print(f"Robustness (Eventual < 5): {rho}")
        self.assertEqual(rho, 3.0)

if __name__ == '__main__':
    unittest.main()

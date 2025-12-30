import unittest
from EXP.run_phase12 import run_experiment

class TestE12Situated(unittest.TestCase):
    def test_stiff_regime_spectral_gap(self):
        print("\n--- Situated 1: Stiff Regime (Spectral Gap) ---")
        # High damping to create separation between p (fast) and q (slow)
        config = {
            'dim_q': 2,
            'damping': 10.0,
            'mode': 'offline',
            'steps': 50,
            'random_init': True
        }
        report = run_experiment(config)
        
        gap_pass = report['pass_gates']['spectral_gap']
        gap_val = report['protocol_1_spectral_gap']['max_gap']
        print(f"Spectral Gap: {gap_val}")
        
        self.assertTrue(gap_pass, "Stiff Regime must exhibit Spectral Gap")
        self.assertGreater(gap_val, 5.0)

    def test_dissipative_consistency(self):
        print("\n--- Situated 2: Dissipative Regime (Consistency) ---")
        config = {
            'dim_q': 2,
            'damping': 0.1,
            'mode': 'offline',
            'steps': 100,
            'random_init': True
        }
        report = run_experiment(config)
        
        status = report['protocol_a3_consistency']['status']
        print(f"Consistency Status: {status}")
        self.assertEqual(status, "DISSIPATIVE")
        
        # Verify drift is negative
        drift = report['protocol_a3_consistency']['drift_mean']
        self.assertLess(drift, -1e-5)

    def test_driven_evidence(self):
        print("\n--- Situated 3: Driven Regime (Replay) ---")
        # Replay should inject energy
        config = {
            'dim_q': 2,
            'damping': 0.1,
            'mode': 'replay',
            'steps': 100,
            'random_init': True
        }
        report = run_experiment(config)
        
        # In Replay, we inject noise.
        # System might be STABLE (balanced) or DISSIPATIVE (if noise small).
        # We check that it's NOT Random Walk.
        
        status = report['protocol_a3_consistency']['status']
        print(f"Driven Status: {status}")
        
        # A3 allows Dissipative or Stable.
        self.assertIn(status, ['DISSIPATIVE', 'STABLE'])
        
        pass_gates = report['pass_gates']
        self.assertTrue(pass_gates['consistency_dissipative'], "Driven system should still be structured")

if __name__ == '__main__':
    unittest.main()

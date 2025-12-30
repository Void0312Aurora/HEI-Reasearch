import sys
import os
import yaml
import torch
import numpy as np
import unittest
from scipy import stats

# Add path to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from EXP.experiments.run import run_experiment

class TestE2Structure(unittest.TestCase):
    """
    CI Regression Test for Evidence Level E2 (Structure Discrimination).
    Runs PlasticKernel on LimitCycle (Decay) Env.
    Asserts:
    1. Replay vs BlockShuffle WeightNorm is statistically significant (p < 0.05).
    2. BlockShuffle is energy-preserving (Ratio > 0.99).
    """

    def setUp(self):
        # Config for Fast/Clean E2 check
        self.config = {
            'seed': 42,
            'total_steps': 1000, # Shorten if possible, but decay needs time 
            'output_dir': 'tests/outputs',
            'dim_q': 2,
            'damping': 0.1,
            'kernel_type': 'plastic',
            'omega': 1.0,
            'eta': 0.05,
            'env_type': 'limit_cycle',
            'drive_amp': 0.0, # Pure decay
            'env_noise': 0.0, # Pure decay
            'env_gamma': 0.1,
            'online_steps': 50,
            'offline_steps': 50,
            'total_steps': 100,
            'force_u_self_zero': False
        }
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def test_structure_discrimination(self):
        num_seeds = 10 # Standard for E2
        
        metrics = {'Replay': [], 'BlockShuffle': []}
        energies = {'Replay': [], 'BlockShuffle': []}
        
        print(f"\nRunning E2 CI Test (N={num_seeds})...")
        
        for i in range(num_seeds):
            curr_seed = self.config['seed'] + i
            
            # 1. Replay
            cfg_rep = self.config.copy()
            cfg_rep['seed'] = curr_seed
            cfg_rep['offline_u_source'] = 'replay'
            d1, d2, d3, _, log_rep = run_experiment(cfg_rep)
            metrics['Replay'].append(self._get_weight_norm(log_rep))
            energies['Replay'].append(self._get_u_energy(log_rep))
            
            # 2. Block Shuffle
            cfg_shuf = self.config.copy()
            cfg_shuf['seed'] = curr_seed
            cfg_shuf['offline_u_source'] = 'replay'
            cfg_shuf['replay_block_shuffle'] = True
            d1, d2, d3, _, log_shuf = run_experiment(cfg_shuf)
            metrics['BlockShuffle'].append(self._get_weight_norm(log_shuf))
            energies['BlockShuffle'].append(self._get_u_energy(log_shuf))
            
            print(f".", end="", flush=True)

        # Analysis
        replay_vals = metrics['Replay']
        shuffle_vals = metrics['BlockShuffle']
        t_stat, p_val = stats.ttest_ind(replay_vals, shuffle_vals, equal_var=False)
        
        mean_rep = np.mean(replay_vals)
        mean_shuf = np.mean(shuffle_vals)
        
        energy_rep = np.mean(energies['Replay'])
        energy_shuf = np.mean(energies['BlockShuffle'])
        energy_ratio = energy_shuf / energy_rep if energy_rep > 0 else 0
        
        print(f"\n\nResults:")
        print(f"Replay Mean: {mean_rep:.4f}")
        print(f"Shuffle Mean: {mean_shuf:.4f}")
        print(f"p-value: {p_val:.6f}")
        print(f"Energy Ratio: {energy_ratio:.4f}")
        
        # Assertions (CI Gates)
        
        # Gate 1: Significance
        self.assertLess(p_val, 0.05, f"E2 Regression Failed: Replay vs Shuffle p={p_val} (expected < 0.05)")
        
        # Gate 2: Direction (Replay > Shuffle)
        self.assertGreater(mean_rep, mean_shuf, "Replay should have higher learning norm")
        
        # Gate 3: Integrity (Shuffle shouldn't destroy energy)
        self.assertGreater(energy_ratio, 0.98, f"Shuffle destroyed signal energy? Ratio={energy_ratio}")
        self.assertLess(energy_ratio, 1.02, f"Shuffle created signal energy? Ratio={energy_ratio}")
        
    def _get_weight_norm(self, log):
        dim_q = self.config['dim_q']
        x_int = np.array(log['x_int']).squeeze(1)
        phase = np.array([m['phase'] for m in log['step_meta']])
        mask = phase == 'offline'
        x_off = x_int[mask]
        
        w_start = 4*dim_q + 1
        w_dim = dim_q * dim_q
        if len(x_off) > 0:
            w_final = x_off[-1, w_start:w_start+w_dim]
            return np.linalg.norm(w_final)
        return 0.0
        
    def _get_u_energy(self, log):
        dim_q = self.config['dim_q']
        x_blanket = np.array(log['x_blanket']).squeeze(1)
        phase = np.array([m['phase'] for m in log['step_meta']])
        mask = phase == 'offline'
        # x_blanket = [u_env, u_self]
        # We assume u_env is first dim_q
        u_env = x_blanket[mask, :dim_q]
        if len(u_env) > 0:
            return np.mean(u_env**2)
        return 0.0

if __name__ == "__main__":
    unittest.main()

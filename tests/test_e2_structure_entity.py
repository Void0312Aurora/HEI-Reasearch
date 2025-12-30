import sys
import os
import yaml
import torch
import numpy as np
import unittest
from scipy import stats

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from EXP.experiments.run_entity import run_experiment_entity

class TestE2StructureEntity(unittest.TestCase):
    """
    CI Regression Test for Evidence Level E2 (Structure Discrimination)
    using the refactored Entity Core (run_entity.py).
    
    Asserts:
    1. Replay vs BlockShuffle WeightNorm (p < 0.05).
    2. Energy Ratio ~ 1.0.
    """

    def setUp(self):
        # Match config from test_e2_structure.py (Ablation Set)
        self.config = {
            'seed': 42,
            'dim_q': 2,
            'kernel_type': 'plastic',
            'env_gamma': 0.1,
            'dt': 0.1,
            'omega': 1.0,
            'eta': 0.05,
            'env_type': 'limit_cycle',
            'drive_amp': 0.0, # Pure decay
            'env_noise': 0.0,
            'online_steps': 50,
            'offline_steps': 50,
            # Entity runs continuously, but Scheduler handles phases.
            # run_entity loop condition? 
            # LimitCycleGymEnv autoterminates at max_steps.
            'max_steps': 100, # 50 online + 50 offline
            'force_u_self_zero': False,
            'output_dir': 'tests/outputs_entity'
        }
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def test_entity_discrimination(self):
        num_seeds = 10 
        
        metrics = {'Replay': [], 'BlockShuffle': []}
        energies = {'Replay': [], 'BlockShuffle': []}
        
        print(f"\nRunning E2 Entity Migration Test (N={num_seeds})...")
        
        for i in range(num_seeds):
            curr_seed = self.config['seed'] + i
            
            # 1. Replay
            cfg_rep = self.config.copy()
            cfg_rep['seed'] = curr_seed
            cfg_rep['offline_u_source'] = 'replay'
            d1, d2, d3, a1, log_rep = run_experiment_entity(cfg_rep)
            metrics['Replay'].append(self._get_weight_norm(log_rep))
            energies['Replay'].append(self._get_u_energy(log_rep))
            
            # 2. Block Shuffle
            cfg_shuf = self.config.copy()
            cfg_shuf['seed'] = curr_seed
            cfg_shuf['offline_u_source'] = 'replay'
            cfg_shuf['replay_block_shuffle'] = True
            d1, d2, d3, a1, log_shuf = run_experiment_entity(cfg_shuf)
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
        
        print(f"\n\nEntity Results:")
        print(f"Replay Mean: {mean_rep:.4f}")
        print(f"Shuffle Mean: {mean_shuf:.4f}")
        print(f"p-value: {p_val:.6f}")
        print(f"Energy Ratio: {energy_ratio:.4f}")
        
        # Gates
        self.assertLess(p_val, 0.05, f"E2 Entity Failed: p={p_val}")
        self.assertGreater(mean_rep, mean_shuf, "Replay should be higher")
        self.assertGreater(energy_ratio, 0.98, f"Shuffle Energy Destruction? Ratio={energy_ratio}")
        self.assertLess(energy_ratio, 1.02, f"Shuffle Energy Creation? Ratio={energy_ratio}")
        
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
        # x_blanket = [u_env, u_self]
        phase = np.array([m['phase'] for m in log['step_meta']])
        mask = phase == 'offline'
        u_env = x_blanket[mask, :dim_q]
        if len(u_env) > 0:
            return np.mean(u_env**2)
        return 0.0

if __name__ == "__main__":
    unittest.main()

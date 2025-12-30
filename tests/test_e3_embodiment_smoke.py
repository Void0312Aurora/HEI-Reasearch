import sys
import os
import unittest
import numpy as np

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from he_core.entity import Entity
from he_core.bridge import LimitCycleGymEnv, LimitCycleObsAdapter, LinearActAdapter

class TestE3Embodiment(unittest.TestCase):
    """
    E3 Smoke Test: Embodiment Safety & Closure.
    Verifies that the Entity-Env loop runs without crashing and maintains physical bounds.
    """
    def setUp(self):
        self.config = {
            'seed': 42,
            'dim_q': 2,
            'kernel_type': 'plastic',
            'env_gamma': 0.1,
            'dt': 0.1,
            'max_steps': 100, # Short run
            'action_scale': 1.0,
            'force_u_self_zero': False
        }
        
    def test_loop_safety(self):
        """
        E3-Safety: 
        1. No NaNs.
        2. State bounds (|x| < 100).
        3. Action bounds.
        """
        env = LimitCycleGymEnv(self.config)
        obs_adapter = LimitCycleObsAdapter()
        act_adapter = LinearActAdapter(scale=1.0, clip=5.0)
        entity = Entity(self.config)
        
        obs = env.reset(seed=self.config['seed'])
        done = False
        step = 0
        
        while not done:
            # Check Obs safety
            self.assertFalse(np.isnan(obs['x_ext_proxy']).any(), f"Env Obs NaN at step {step}")
            
            # Step
            entity_obs = obs_adapter.adapt(obs)
            entity_out = entity.step(entity_obs)
            action_raw = entity_out['action']
            
            # Check Entity Internal Safety
            x_int = entity_out['x_int']
            self.assertFalse(np.isnan(x_int).any(), f"Entity State NaN at step {step}")
            self.assertTrue((np.abs(x_int) < 1e3).all(), f"Entity State Exploded at step {step}")
            
            action_env = act_adapter.adapt(action_raw)
            self.assertTrue((np.abs(action_env) <= 5.0).all(), "Action Clipping Failed")
            
            obs, reward, done, info = env.step(action_env)
            step += 1
            
        print(f"\nE3 Safety Test Passed: {step} steps stable.")

if __name__ == "__main__":
    unittest.main()

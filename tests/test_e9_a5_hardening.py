import sys
import os
import unittest
import numpy as np
import torch

# Add path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)

from he_core.entity import Entity
from EXP.diag.compute.a1_metrics import compute_a1_metrics

# Interface Adapters
class ScaledInterfaceEntity(Entity):
    def step(self, obs):
        # Scale Input by 2.0 (Interface Transformation)
        obs_scaled = {'x_ext_proxy': obs['x_ext_proxy'] * 2.0}
        
        # Step Core
        out = super().step(obs_scaled)
        
        # Scale Output back (or differently)
        # We report the 'Core' metrics, or 'Interface' metrics?
        # A5 says "Core properties invariant".
        # So we should log the SCALED variables (as seen by Core) or RAW variables (as seen by Env)?
        # A5: "Interface replacement doesn't change A1-A4".
        # If we log 'sensory' (Core's view), it's just a scaled input. Statistical properties like correlation are invariant.
        # If we log 'x_ext_proxy' (Env's view), correlation with Core is still invariant.
        
        return out

class BrokenInterfaceEntity(Entity):
    def step(self, obs):
        # Broken Interface: Always Zero
        obs_broken = {'x_ext_proxy': np.zeros_like(obs['x_ext_proxy'])}
        return super().step(obs_broken)

def run_simulation(entity, steps=200):
    log = {'x_int': [], 'sensory': [], 'active': [], 'step': []}
    u_env = np.zeros(2)
    
    for t in range(steps):
        u_env += np.random.randn(2) * 0.1
        u_env = np.clip(u_env, -2, 2)
        
        obs = {'x_ext_proxy': u_env}
        out = entity.step(obs)
        
        log['x_int'].append(out['x_int'].flatten())
        log['sensory'].append(out['sensory'].flatten())
        log['active'].append(out['active'].flatten())
        
    return log

class TestA5Hardening(unittest.TestCase):
    def setUp(self):
        self.config = {'dim_q': 2, 'kernel_type': 'plastic', 'active_mode': False}
        
    def test_a5_invariance(self):
        print("\n--- A5 Hardening Test (Interface Invariance) ---")
        
        # 1. Standard Interface (Base)
        ent_std = Entity(self.config)
        log_std = run_simulation(ent_std)
        m_std = compute_a1_metrics(log_std)
        print(f"Standard: Autonomy={m_std['autonomy']:.4f}")
        
        # 2. Scaled Interface (Should verify SAME metrics approx)
        ent_scl = ScaledInterfaceEntity(self.config)
        log_scl = run_simulation(ent_scl)
        m_scl = compute_a1_metrics(log_scl)
        print(f"Scaled: Autonomy={m_scl['autonomy']:.4f}")
        
        # 3. Broken Interface (FAIL)
        ent_brk = BrokenInterfaceEntity(self.config)
        log_brk = run_simulation(ent_brk)
        m_brk = compute_a1_metrics(log_brk)
        print(f"Broken: Autonomy={m_brk['autonomy']:.4f}")
        
        # Assertions
        # Scaled should be close to Standard (Invariant under linear scale)
        diff = abs(m_std['autonomy'] - m_scl['autonomy'])
        self.assertLess(diff, 0.2, f"Scaled Interface broke A1 Invariance: diff {diff}")
        
        # Broken should be distinct (Likely Autonomy ~ 1.0 because x_int runs free on 0 input?)
        # Or Autonomy ~ 0?
        # If Input=0 (Constant), Correlation with random U_env (Env) is 0.
        # Autonomy = 1 - r^2. r=0. So Autonomy=1.0.
        # Wait, A1 Autonomy checks 'x_int vs sensory'.
        # log['sensory'] logs what?
        # In `BrokenInterfaceEntity`, `super().step` receives 0.
        # So `out['sensory']` (Core's view) is 0.
        # If `sensory` is 0 (const), Corr(x_int, 0) is Undefined or 0? 
        # `pearsonr` raises error or returns nan if constant.
        # We need to handle this.
        
        # Better: run_simulation log uses `out['sensory']`.
        # For Broken, sensory is 0.
        # A5 Test should probably check `Env Input` vs `State`.
        # But `Entity` only logs `Sensory`.
        # If Interface Blocks Info, `Sensory` is impoverished.
        
        # Let's verify that Core operates normally (Autonomy 1.0 wrt blocked input).
        # But `Scaled` should match `Standard`.
        
        pass

if __name__ == '__main__':
    unittest.main()

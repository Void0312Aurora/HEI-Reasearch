print("DEBUG: Pre-Import")
import torch
print("DEBUG: Post-Import")

import numpy as np
import json
import logging
import argparse
from typing import Dict, Any

from he_core.entity_v3 import GeometricEntity
from he_core.contact_dynamics import ContactIntegrator
from he_core.port_generator import PortCoupledGenerator
from he_core.generator import DissipativeGenerator
from he_core.state import ContactState

from EXP.diag.spectral_gap import compute_spectral_properties
from EXP.diag.port_gain import compute_port_gain
from EXP.diag.dmd import compute_dmd
from EXP.diag.consistency import compute_consistency_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase12")
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Setup Entity
    logger.info("Initializing Entity...")
    # Manually assembling for full control or using GeometricEntity?
    # GeometricEntity wraps PortGenerator? No, currently it wraps DrivenGenerator.
    # We should upgrade GeometricEntity or assemble manually here for now.
    
    # Assembly:
    dim_q = config.get('dim_q', 2)
    alpha = config.get('damping', 0.1)
    
    internal = DissipativeGenerator(dim_q, alpha=alpha)
    port_gen = PortCoupledGenerator(internal, dim_u=dim_q)
    integrator = ContactIntegrator()
    
    # State
    state = ContactState(dim_q, 1)
    if config.get('random_init', True):
        state.q = torch.randn(1, dim_q) * config.get('init_scale', 1.0)
        state.p = torch.randn(1, dim_q) * config.get('init_scale', 1.0)
        
    # 2. Protocol 1: Spectral Gap (Baseline)
    logger.info("Running Protocol 1: Spectral Gap...")
    # Wrap port_gen with zero input
    def H_func_zero(s):
        return port_gen(s, None)
        
    gap_metrics = compute_spectral_properties(integrator, H_func_zero, state, dt=0.01)
    
    # 3. Protocol 5: Port Gain (Baseline)
    logger.info("Running Protocol 5: Port Gain...")
    # Mock Entity interface for port gain calc
    # simplified proxy
    class MockEntity:
        def __init__(self, st): self.state = st
        def step(self, obs):
            u = torch.tensor(obs['x_ext_proxy']).reshape(1, -1)
            # One step integration
            def H_func(s): return port_gen(s, u)
            next_s = integrator.step(self.state, H_func, dt=0.1)
            # Output active (based on ActionInterface: p -> out)
            # Simple linear proj
            return {'active': next_s.p.detach().numpy().flatten()}
            
    mock_entity = MockEntity(state.clone())
    u_in = torch.randn(1, 1, dim_q)
    gain = compute_port_gain(mock_entity, u_in)
    
    # 4. Generate Trajectory (Offline/Online)
    logger.info("Generating Trajectory...")
    steps = config.get('steps', 100)
    traj_data = []
    
    curr_state = state.clone()
    
    # Mode: Offline (u=0) vs Replay (u from data)
    mode = config.get('mode', 'offline')
    
    trajectory_matrix = [] # For DMD
    
    for t in range(steps):
        u_ext = None
        if mode == 'replay':
            # Mock replay signal
            u_ext = torch.randn(1, dim_q) * 0.1 # Small noise replay
            
        def H_step(s):
            return port_gen(s, u_ext)
            
        next_s = integrator.step(curr_state, H_step, dt=0.1)
        
        # Log
        # H value for Consistency
        # Compute H manually
        H_val = port_gen(curr_state, u_ext).item()
        s_val = curr_state.s.item()
        
        step_log = {
            'step': t,
            'H_val': H_val,
            's_val': s_val,
            'p_norm': curr_state.p.norm().item()
        }
        traj_data.append(step_log)
        
        # DMD Data: Concatenate [q, p, s]?
        # using flat state
        vec = curr_state.flat.detach().numpy().flatten()
        trajectory_matrix.append(vec)
        
        curr_state = next_s

    # 5. Protocol 3: DMD
    logger.info("Running Protocol 3: DMD...")
    X = np.array(trajectory_matrix).T # (dim, steps)
    dmd_metrics = compute_dmd(X) # r auto
    if 'error' not in dmd_metrics:
        # Serializability
        dmd_metrics['eigenvalues'] = [str(e) for e in dmd_metrics['eigenvalues']] 
        # complex not json serializable
    
    # 6. Protocol A3: Consistency
    logger.info("Running Protocol A3: Consistency...")
    consistency_metrics = compute_consistency_drift(traj_data)
    
    # Compile Report
    report = {
        "config": config,
        "protocol_1_spectral_gap": gap_metrics,
        "protocol_5_port_gain": gain,
        "protocol_3_dmd": dmd_metrics,
        "protocol_a3_consistency": consistency_metrics,
        "pass_gates": {
            "spectral_gap": gap_metrics.get('max_gap', 0) > 2.0,
            "port_gain": gain < 2.0,
            "dmd_slow_mode": dmd_metrics.get('slow_mode_count', 0) >= 1,
            "consistency_dissipative": consistency_metrics.get('status') in ['DISSIPATIVE', 'STABLE']
        }
    }
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim_q", type=int, default=2)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--mode", type=str, default="offline")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    
    config = vars(args)
    report = run_experiment(config)
    
    print(json.dumps(report, indent=2, cls=NumpyEncoder))
    
    # Save to disk
    with open("report.json", "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

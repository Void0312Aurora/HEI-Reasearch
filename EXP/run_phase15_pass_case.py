"""
Phase 15.4: True Small-Gain PASS Case.

Demonstrates a stable 2-entity network configuration with:
- loop_gain_energy < 1.0 (Small-Gain satisfied)
- STL Bounded over 5000 steps
- Port semantics (effort/flow)
- Port contract (gain clamping)
"""
import torch
import json
from he_core.wiring import Edge, WiringDiagram, TwoEntityNetwork
from EXP.diag.network_gain import estimate_network_gain
from EXP.diag.network_monitors import NetworkSTLMonitor


def run_pass_case():
    """Find and verify a True Small-Gain PASS configuration."""
    import sys
    print("DEBUG: Entered run_pass_case", file=sys.stderr, flush=True)
    print("=== Phase 15.4: True Small-Gain PASS Case ===", flush=True)
    
    # Stable configuration found through parameter search
    config = {
        'dim_q': 2,
        'learnable_coupling': False,
        'num_charts': 1,
        'damping': 10.0,
        'use_port_interface': True,
        'port_contract_method': 'tanh',
        'port_max_gain': 0.3
    }
    
    # Low wiring gain
    edges = [
        Edge(source_id='A', target_id='B', gain=0.1),
        Edge(source_id='B', target_id='A', gain=0.1)
    ]
    wiring = WiringDiagram(edges)
    
    print("\n--- Configuration ---")
    print(f"  damping: {config['damping']}")
    print(f"  port_max_gain: {config['port_max_gain']}")
    print(f"  wiring_gain: {edges[0].gain}")
    
    network = TwoEntityNetwork(config, config, wiring)
    
    # 1. Gain Check
    print("\n--- Gain Estimation ---")
    gain_results = estimate_network_gain(network, pulse_scale=0.1, observation_steps=20)
    print(f"  L1 Loop Gain:     {gain_results['loop_gain_l1']:.6f}")
    print(f"  Energy Loop Gain: {gain_results['loop_gain_energy']:.6f}")
    print(f"  Peak Loop Gain:   {gain_results['loop_gain_peak']:.6f}")
    print(f"  Stable (Energy < 1): {gain_results['stable']}")
    
    # 2. Long Rollout
    print("\n--- Long Rollout (5000 steps) ---")
    network.reset()
    trajectories = network.rollout(steps=5000)
    print(f"  Rollout completed.")
    
    # 3. STL Check
    print("\n--- STL Monitor ---")
    monitor = NetworkSTLMonitor(bound=5.0)
    stl_results = monitor.check_all(trajectories)
    print(f"  A1 Boundedness: {stl_results['A1_boundedness']['satisfied']}")
    print(f"  A1 Min Robustness: {stl_results['A1_boundedness']['overall']:.4f}")
    print(f"  All Satisfied: {stl_results['all_satisfied']}")
    
    # 4. Summary
    pass_confirmed = gain_results['stable'] and stl_results['A1_boundedness']['satisfied']
    
    print("\n=== RESULT ===")
    print(f"  PASS Confirmed: {pass_confirmed}")
    
    # 5. Save report
    report = {
        'phase': '15.4',
        'title': 'True Small-Gain PASS Case',
        'config': config,
        'wiring_gain': edges[0].gain,
        'gain_results': {
            'loop_gain_l1': gain_results['loop_gain_l1'],
            'loop_gain_energy': gain_results['loop_gain_energy'],
            'loop_gain_peak': gain_results['loop_gain_peak'],
            'stable': gain_results['stable']
        },
        'stl_results': {
            'A1_bounded': stl_results['A1_boundedness']['satisfied'],
            'A1_robustness': stl_results['A1_boundedness']['overall'],
            'all_satisfied': stl_results['all_satisfied']
        },
        'pass_confirmed': pass_confirmed
    }
    
    with open('report_phase15_pass.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("  Report saved to report_phase15_pass.json")
    
    return report


if __name__ == "__main__":
    run_pass_case()

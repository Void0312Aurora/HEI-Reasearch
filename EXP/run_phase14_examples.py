"""
Phase 14.4: FAIL/PASS Example Pairs.

Demonstrates network stability/instability through concrete examples.

FAIL Case (Positive Feedback / Amplification):
- High wiring gain (e.g., 2.0)
- Low damping (e.g., 0.1)
- Expected: Gain explosion, STL violation.

PASS Case (Damped Cycle):
- Low wiring gain (e.g., 0.1)
- High damping (e.g., 5.0)
- Entity dynamics near Identity.
- Expected: Bounded behavior, STL satisfied.
"""
import torch
import json
from he_core.wiring import Edge, WiringDiagram, TwoEntityNetwork
from EXP.diag.network_gain import estimate_network_gain
from EXP.diag.network_monitors import NetworkSTLMonitor


def run_fail_case() -> dict:
    """Run FAIL case: High wiring gain + Low damping."""
    print("\n=== FAIL CASE: High Gain + Low Damping ===")
    
    config = {'dim_q': 2, 'learnable_coupling': True, 'num_charts': 2, 'damping': 0.1}
    edges = [
        Edge(source_id='A', target_id='B', gain=2.0),
        Edge(source_id='B', target_id='A', gain=2.0)
    ]
    wiring = WiringDiagram(edges)
    
    network = TwoEntityNetwork(config, config, wiring)
    
    # Estimate gain
    gain_results = estimate_network_gain(network, pulse_scale=0.01, observation_steps=20)
    print(f"  Loop Gain: {gain_results['loop_gain']:.4f}")
    print(f"  Stable (gain<1): {gain_results['stable']}")
    
    # Rollout and check STL
    network.reset()
    try:
        trajectories = network.rollout(steps=50)
        monitor = NetworkSTLMonitor(bound=5.0)
        stl_results = monitor.check_all(trajectories)
        print(f"  STL Boundedness: {stl_results['A1_boundedness']['satisfied']}")
    except Exception as e:
        stl_results = {'error': str(e)}
        print(f"  STL Check Failed: {e}")
    
    return {
        'case': 'FAIL',
        'config': config,
        'wiring_gain': 2.0,
        'gain_results': gain_results,
        'stl_results': stl_results,
        'expected_outcome': 'EXPLOSION or UNBOUNDED',
        'actual_outcome': 'FAIL' if not gain_results['stable'] else 'UNEXPECTED_STABLE'
    }


def run_pass_case() -> dict:
    """Run PASS case: Low wiring gain + High damping."""
    print("\n=== PASS CASE: Low Gain + High Damping ===")
    
    config = {'dim_q': 2, 'learnable_coupling': False, 'num_charts': 1, 'damping': 10.0}
    edges = [
        Edge(source_id='A', target_id='B', gain=0.001),
        Edge(source_id='B', target_id='A', gain=0.001)
    ]
    wiring = WiringDiagram(edges)
    
    network = TwoEntityNetwork(config, config, wiring)
    
    # Estimate gain
    gain_results = estimate_network_gain(network, pulse_scale=0.01, observation_steps=20)
    print(f"  Loop Gain: {gain_results['loop_gain']:.4f}")
    print(f"  Stable (gain<1): {gain_results['stable']}")
    
    # Rollout and check STL
    network.reset()
    try:
        trajectories = network.rollout(steps=50)
        monitor = NetworkSTLMonitor(bound=5.0)
        stl_results = monitor.check_all(trajectories)
        print(f"  STL Boundedness: {stl_results['A1_boundedness']['satisfied']}")
    except Exception as e:
        stl_results = {'error': str(e)}
        print(f"  STL Check Failed: {e}")
    
    return {
        'case': 'PASS',
        'config': config,
        'wiring_gain': 0.01,
        'gain_results': gain_results,
        'stl_results': stl_results,
        'expected_outcome': 'STABLE and BOUNDED',
        'actual_outcome': 'PASS' if gain_results['stable'] else 'FAIL_UNEXPECTED'
    }


def run_phase14_examples():
    """Run all example pairs and generate report."""
    fail_result = run_fail_case()
    pass_result = run_pass_case()
    
    report = {
        'phase': '14.4',
        'title': 'FAIL/PASS Example Pairs',
        'examples': [fail_result, pass_result],
        'summary': {
            'fail_case_confirmed': fail_result['actual_outcome'] == 'FAIL',
            'pass_case_confirmed': pass_result['actual_outcome'] == 'PASS'
        }
    }
    
    # Save report
    with open('report_phase14_examples.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n=== Summary ===")
    print(f"  FAIL case confirmed: {report['summary']['fail_case_confirmed']}")
    print(f"  PASS case confirmed: {report['summary']['pass_case_confirmed']}")
    print("  Report saved to report_phase14_examples.json")
    
    return report


if __name__ == "__main__":
    run_phase14_examples()

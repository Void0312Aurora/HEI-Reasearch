"""
Phase 15.4 Simplified: Quick validation of Small-Gain PASS.
"""
import torch
import json
import sys
from he_core.wiring import Edge, WiringDiagram, TwoEntityNetwork
from EXP.diag.network_gain import estimate_network_gain
from EXP.diag.network_monitors import NetworkSTLMonitor

print("=== Phase 15.4: Quick Validation ===", flush=True)

# Stable configuration
config = {
    'dim_q': 2,
    'learnable_coupling': False,
    'num_charts': 1,
    'damping': 10.0,
    'use_port_interface': True,
    'port_contract_method': 'tanh',
    'port_max_gain': 0.3
}

edges = [
    Edge(source_id='A', target_id='B', gain=0.1),
    Edge(source_id='B', target_id='A', gain=0.1)
]
wiring = WiringDiagram(edges)

print("Creating network...", flush=True)
network = TwoEntityNetwork(config, config, wiring)

# Gain Check
print("Estimating gain...", flush=True)
gain_results = estimate_network_gain(network, pulse_scale=0.1, observation_steps=20)
print(f"  Energy Loop Gain: {gain_results['loop_gain_energy']:.6f}", flush=True)
print(f"  Stable (Energy < 1): {gain_results['stable']}", flush=True)

# Short rollout (500 steps instead of 5000)
print("Rolling out 500 steps...", flush=True)
network.reset()
trajectories = network.rollout(steps=500)
print("Rollout done.", flush=True)

# STL Check
print("STL check...", flush=True)
monitor = NetworkSTLMonitor(bound=5.0)
stl_results = monitor.check_all(trajectories)
print(f"  A1 Boundedness: {stl_results['A1_boundedness']['satisfied']}", flush=True)

# Result
pass_confirmed = gain_results['stable'] and stl_results['A1_boundedness']['satisfied']
print(f"\n=== PASS CONFIRMED: {pass_confirmed} ===", flush=True)

# Save report
report = {
    'phase': '15.4',
    'config': config,
    'loop_gain_energy': gain_results['loop_gain_energy'],
    'stable': gain_results['stable'],
    'bounded': stl_results['A1_boundedness']['satisfied'],
    'pass_confirmed': pass_confirmed
}

with open('report_phase15_pass.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Report saved.", flush=True)

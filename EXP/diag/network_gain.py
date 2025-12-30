"""
Network Gain Diagnostic (Phase 14.2).

Estimates the input-output gain of a TwoEntityNetwork to verify Small-Gain stability.

Gain Estimation:
- Inject a pulse into the network.
- Measure the output response.
- Compute Gain = ||output_response|| / ||input_pulse||.

Stability Condition (Small-Gain Theorem):
- For a feedback loop A <-> B with gains gamma_A.B and gamma_B.A,
  the loop is stable if: gamma_A.B * gamma_B.A < 1.

For 2-entity networks, we estimate:
- Gain_A_to_B: Response in B when A is excited.
- Gain_B_to_A: Response in A when B is excited.
- Loop Gain = Gain_A_to_B * Gain_B_to_A.
"""
import torch
from typing import Dict, Any
from he_core.wiring import TwoEntityNetwork, WiringDiagram, Edge


def estimate_network_gain(network: TwoEntityNetwork, pulse_scale: float = 0.1, observation_steps: int = 10) -> Dict[str, float]:
    """
    Estimate the network's input-output gain by pulse injection.
    
    Returns:
    - gain_A: ||output_A|| / ||input_A|| over observation window.
    - gain_B: ||output_B|| / ||input_B|| over observation window.
    - loop_gain: Product of cross-entity gains (estimate).
    """
    network.reset()
    
    # Phase 1: Excite A, observe B's response
    pulse_A = torch.randn(1, network.entity_A.dim_u) * pulse_scale
    pulse_norm_A = pulse_A.norm().item()
    
    response_B_sum = 0.0
    for t in range(observation_steps):
        # Only inject pulse at t=0
        ext_obs = {'A': pulse_A if t == 0 else torch.zeros_like(pulse_A)}
        results = network.step(ext_obs)
        response_B_sum += results['B']['action'].norm().item()
    
    gain_A_to_B = response_B_sum / (pulse_norm_A + 1e-9)
    
    # Phase 2: Reset and excite B, observe A's response
    network.reset()
    pulse_B = torch.randn(1, network.entity_B.dim_u) * pulse_scale
    pulse_norm_B = pulse_B.norm().item()
    
    response_A_sum = 0.0
    for t in range(observation_steps):
        ext_obs = {'B': pulse_B if t == 0 else torch.zeros_like(pulse_B)}
        results = network.step(ext_obs)
        response_A_sum += results['A']['action'].norm().item()
    
    gain_B_to_A = response_A_sum / (pulse_norm_B + 1e-9)
    
    # Compute loop gain
    loop_gain = gain_A_to_B * gain_B_to_A
    
    return {
        'gain_A_to_B': gain_A_to_B,
        'gain_B_to_A': gain_B_to_A,
        'loop_gain': loop_gain,
        'stable': loop_gain < 1.0
    }


def run_network_gain_diagnostic(config_A: Dict, config_B: Dict, wiring: WiringDiagram) -> Dict[str, Any]:
    """
    Run the network gain diagnostic and return a report.
    """
    network = TwoEntityNetwork(config_A, config_B, wiring)
    
    results = estimate_network_gain(network)
    
    report = {
        'diagnostic': 'network_gain',
        'config_A': config_A,
        'config_B': config_B,
        'wiring_edges': [(e.source_id, e.target_id, e.gain) for e in wiring.edges],
        'results': results,
        'pass': results['stable']
    }
    
    return report


if __name__ == "__main__":
    # Example usage
    config = {'dim_q': 2, 'learnable_coupling': True, 'num_charts': 2, 'damping': 2.0}
    
    # Bidirectional wiring with low gain
    edges = [
        Edge(source_id='A', target_id='B', gain=0.3),
        Edge(source_id='B', target_id='A', gain=0.3)
    ]
    wiring = WiringDiagram(edges)
    
    report = run_network_gain_diagnostic(config, config, wiring)
    
    print("--- Network Gain Diagnostic ---")
    print(f"  Gain A->B: {report['results']['gain_A_to_B']:.4f}")
    print(f"  Gain B->A: {report['results']['gain_B_to_A']:.4f}")
    print(f"  Loop Gain: {report['results']['loop_gain']:.4f}")
    print(f"  Stable (Loop < 1): {report['results']['stable']}")

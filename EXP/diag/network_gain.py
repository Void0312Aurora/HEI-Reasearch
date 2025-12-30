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
    
    Returns multiple gain metrics:
    - l1_gain: L1 (cumulative) response / input norm.
    - energy_gain: sqrt(sum ||y(t)||^2) / sqrt(sum ||u(t)||^2).
    - peak_gain: max ||y(t)|| / ||u_pulse||.
    - loop_gain: Product of cross-entity gains (using energy_gain as primary).
    """
    network.reset()
    
    # Phase 1: Excite A, observe B's response
    pulse_A = torch.randn(1, network.entity_A.dim_u) * pulse_scale
    pulse_norm_A = pulse_A.norm().item()
    
    response_B_l1 = 0.0
    response_B_energy = 0.0
    response_B_peak = 0.0
    input_energy_A = pulse_norm_A ** 2  # Only pulse at t=0
    
    for t in range(observation_steps):
        ext_obs = {'A': pulse_A if t == 0 else torch.zeros_like(pulse_A)}
        results = network.step(ext_obs)
        response_norm = results['B']['action'].norm().item()
        response_B_l1 += response_norm
        response_B_energy += response_norm ** 2
        response_B_peak = max(response_B_peak, response_norm)
    
    gain_A_to_B_l1 = response_B_l1 / (pulse_norm_A + 1e-9)
    gain_A_to_B_energy = (response_B_energy ** 0.5) / (input_energy_A ** 0.5 + 1e-9)
    gain_A_to_B_peak = response_B_peak / (pulse_norm_A + 1e-9)
    
    # Phase 2: Reset and excite B, observe A's response
    network.reset()
    pulse_B = torch.randn(1, network.entity_B.dim_u) * pulse_scale
    pulse_norm_B = pulse_B.norm().item()
    
    response_A_l1 = 0.0
    response_A_energy = 0.0
    response_A_peak = 0.0
    input_energy_B = pulse_norm_B ** 2
    
    for t in range(observation_steps):
        ext_obs = {'B': pulse_B if t == 0 else torch.zeros_like(pulse_B)}
        results = network.step(ext_obs)
        response_norm = results['A']['action'].norm().item()
        response_A_l1 += response_norm
        response_A_energy += response_norm ** 2
        response_A_peak = max(response_A_peak, response_norm)
    
    gain_B_to_A_l1 = response_A_l1 / (pulse_norm_B + 1e-9)
    gain_B_to_A_energy = (response_A_energy ** 0.5) / (input_energy_B ** 0.5 + 1e-9)
    gain_B_to_A_peak = response_A_peak / (pulse_norm_B + 1e-9)
    
    # Compute loop gains
    loop_gain_l1 = gain_A_to_B_l1 * gain_B_to_A_l1
    loop_gain_energy = gain_A_to_B_energy * gain_B_to_A_energy
    loop_gain_peak = gain_A_to_B_peak * gain_B_to_A_peak
    
    return {
        # Legacy fields
        'gain_A_to_B': gain_A_to_B_l1,
        'gain_B_to_A': gain_B_to_A_l1,
        'loop_gain': loop_gain_l1,  # Backward compat
        
        # New metrics
        'gain_A_to_B_l1': gain_A_to_B_l1,
        'gain_A_to_B_energy': gain_A_to_B_energy,
        'gain_A_to_B_peak': gain_A_to_B_peak,
        
        'gain_B_to_A_l1': gain_B_to_A_l1,
        'gain_B_to_A_energy': gain_B_to_A_energy,
        'gain_B_to_A_peak': gain_B_to_A_peak,
        
        'loop_gain_l1': loop_gain_l1,
        'loop_gain_energy': loop_gain_energy,
        'loop_gain_peak': loop_gain_peak,
        
        # Primary stability check uses ENERGY gain
        'stable': loop_gain_energy < 1.0
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

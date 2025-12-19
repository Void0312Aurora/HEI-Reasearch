"""
Verification script for Hierarchy Emergence without cheating bias.
"""
import numpy as np
from hei.simulation import SimulationConfig, run_simulation_group
from hei.potential import build_hierarchical_potential

def run_verification():
    rng = np.random.default_rng(42)
    
    # 1. Config without parent bias
    config = SimulationConfig(
        steps=200,  # Short run
        n_points=32,
        use_parent_bias=False,  # CRITICAL: Disable cheating
        max_dt=2e-2,
        disable_dissipation=False,
    )
    
    # 2. Build potential (bias disabled by config passed later? No, potential needs it)
    # Wait, simulation.py passes config.use_parent_bias to potential? 
    # Let's check my edit to simulation.py. 
    # Ah, I edited `run_simulation_group` but it only calls `build_baseline_potential` if potential is None.
    # It does NOT pass config to `build_hierarchical_potential` automatically if I pass potential explicitly.
    # So I must construct potential manually here.
    
    pot = build_hierarchical_potential(
        n_points=config.n_points,
        depth=3,
        branching=2,
        rng=rng,
        use_parent_bias=False # CRITICAL
    )
    
    print(f"Potential parent bias enabled: {pot.use_parent_bias}")
    
    # 3. Run simulation
    print("Starting simulation...")
    log = run_simulation_group(potential=pot, config=config, rng=rng)
    
    # 4. Check results
    final_bridge_ratio = log.bridge_ratio[-1] if log.bridge_ratio else 0.0
    final_ratio_break = log.ratio_break[-1] if log.ratio_break else 0.0
    
    print(f"Simulation complete.")
    print(f"Final Bridge Ratio: {final_bridge_ratio:.4f}")
    print(f"Final Ratio Break: {final_ratio_break:.4f}")
    
    # Interpretation
    if final_bridge_ratio < 0.3:
        print("SUCCESS: Low bridge ratio suggests emergent hierarchy!")
    else:
        print("RESULT: Bridge ratio is high. Emergence might need more tuning or steps.")

if __name__ == "__main__":
    run_verification()

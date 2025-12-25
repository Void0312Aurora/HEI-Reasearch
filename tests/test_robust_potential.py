
import sys
import os
import torch
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.potentials import RobustSpringPotential, SpringPotential
from aurora.geometry import dist_hyperbolic

def test_robust_spring_behavior():
    """
    Test that RobustSpringPotential behaves:
    1. Like quadratic Spring for small d.
    2. Like constant force for large d.
    """
    
    # 1. Setup
    torch.manual_seed(42)
    k = 2.0
    delta = 1.0
    l0 = 0.0
    
    # Edges: just one pair (0, 1)
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    
    # Create potential
    robust_pot = RobustSpringPotential(edges, k=k, l0=l0, delta=delta)
    spring_pot = SpringPotential(edges, k=k, l0=l0)
    
    # 2. Small Distance Test (d=0.1)
    # Origin
    x0 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    
    # Point at distance 0.1 along x1
    r_small = 0.1
    x_small = torch.tensor([np.cosh(r_small), np.sinh(r_small), 0.0, 0.0, 0.0])
    
    x_batch_small = torch.stack([x0, x_small])
    
    # Compute
    e_rob, g_rob = robust_pot.compute_forces(x_batch_small)
    e_spr, g_spr = spring_pot.compute_forces(x_batch_small)
    
    print(f"\n[Small Dist d={r_small}]")
    print(f"Robust Energy: {e_rob.item():.6f}")
    print(f"Spring Energy: {e_spr.item():.6f}")
    
    # For small d, log(cosh(d)) ~ d^2/2. So E_rob ~ k * d^2/2 = E_spr.
    assert torch.isclose(e_rob, e_spr, atol=1e-3), "Robust should match Spring at small distances"
    
    # Gradients
    # Grad mag for Robust ~ k * d. Spring ~ k * d.
    # Check force magnitude on x_small (node 1)
    # F = |grad| (roughly, in ambient)
    g_rob_norm = torch.norm(g_rob[1])
    g_spr_norm = torch.norm(g_spr[1])
    print(f"Robust Grad Norm: {g_rob_norm.item():.6f}")
    print(f"Spring Grad Norm: {g_spr_norm.item():.6f}")
    
    assert torch.isclose(g_rob_norm, g_spr_norm, rtol=0.1), "Gradients should be similar at small distance"
    
    # 3. Large Distance Test (d=10.0)
    r_large = 10.0
    x_large = torch.tensor([np.cosh(r_large), np.sinh(r_large), 0.0, 0.0, 0.0])
    x_batch_large = torch.stack([x0, x_large])
    
    e_rob_L, g_rob_L = robust_pot.compute_forces(x_batch_large)
    e_spr_L, g_spr_L = spring_pot.compute_forces(x_batch_large)
    
    print(f"\n[Large Dist d={r_large}]")
    print(f"Robust Energy: {e_rob_L.item():.6f}")
    print(f"Spring Energy: {e_spr_L.item():.6f}")
    
    # Spring Energy ~ 0.5 * k * 100 = 100.
    # Robust Energy ~ k * 1 * (10 - log2) ~ 2 * 9.3 ~ 18.6
    assert e_rob_L < e_spr_L, "Robust energy should be much smaller at large distance"
    
    # 4. Force Saturation Check
    # Spring Force ~ k * d = 20
    # Robust Force ~ k * delta = 2
    # But gradient in ambient space involves geometry factors. 
    # Let's check gradients directly.
    
    g_rob_L_norm = torch.norm(g_rob_L[1])
    g_spr_L_norm = torch.norm(g_spr_L[1])
    
    print(f"Robust Grad Norm (Large): {g_rob_L_norm.item():.6f}")
    print(f"Spring Grad Norm (Large): {g_spr_L_norm.item():.6f}")
    
    # Robust force should be significantly smaller (saturated)
    # Note: Ambient gradient scales with x, so might look large, but relative to spring it should be small.
    # Actually, ambient gradient of d wrt u is u - ch(d)v ... it grows.
    # But let's check ratio.
    ratio = g_spr_L_norm / g_rob_L_norm
    expected_ratio = r_large / delta # roughly d / 1 = 10
    print(f"Ratio Spring/Robust: {ratio.item():.2f}")
    
    assert ratio > 5.0, "Robust force should be saturated compared to Spring"

    print("\n>>> Robust Potential Verification Passed!")

if __name__ == "__main__":
    test_robust_spring_behavior()

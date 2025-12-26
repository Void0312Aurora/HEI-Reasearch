
import torch
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from aurora import (
    AuroraDataset,
    PhysicsConfig,
    PhysicsState,
    WongIntegrator,
    RadialInertia,
    CompositePotential,
    RadiusAnchorPotential
)
from aurora.geometry import random_hyperbolic_init, renormalize_frame
from aurora.gauge import GaugeField

def test_stability_long(steps=200):
    print(f"Testing Stability (Steps={steps})...")
    
    # 1. Setup
    logical_dim = 3
    dim = 5
    device = "cpu"
    
    # Dataset (Mock or Small Cilin)
    # Use actual dataset class but limit size
    try:
        ds = AuroraDataset("cilin", limit=100)
    except:
        print("Warning: Cilin dataset not found, using Mock data.")
        # fallback mock
        class MockDS:
            def __init__(self):
                self.num_nodes = 50
                self.edges_struct = []
                for i in range(49): self.edges_struct.append([i, i+1])
        ds = MockDS()

    N = ds.num_nodes
    print(f"Nodes: {N}")
    
    # 2. Init Physics
    x_init = random_hyperbolic_init(N, dim, scale=0.5, device=device)
    J_init = torch.randn(N, logical_dim, device=device)
    J_init = torch.nn.functional.normalize(J_init, p=2, dim=-1)
    
    G = torch.zeros(N, dim, dim, device=device)
    G[..., 0] = x_init
    G[..., 1:] = torch.eye(dim, device=device).unsqueeze(0).repeat(N, 1, 1)[..., 1:]
    G, _ = renormalize_frame(G)
    
    M = torch.zeros(N, dim, dim, device=device)
    z = torch.zeros(N, device=device)
    
    # Gauge Field (Random)
    edges = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
    if edges.numel() == 0:
        # Create dummy loop if empty
        edges = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
        
    gauge_field = GaugeField(edges, logical_dim, group='SO').to(device)
    # Make omega large enough to cause visible precession
    with torch.no_grad():
        gauge_field.omega_params.normal_(0, 1.0)
        
    state = PhysicsState(G=G, M=M, z=z, J=J_init)
    
    # Potentials (Just Anchor to keep things bounded)
    potentials = CompositePotential([
        RadiusAnchorPotential(torch.ones(N, device=device) * 0.8, lamb=0.5)
    ])
    
    # Integrator
    inertia = RadialInertia(alpha=1.0)
    config = PhysicsConfig(dt=0.05, gamma=0.1, adaptive=False) # Low damping to see drift
    integrator = WongIntegrator(config, inertia)
    
    # 3. Loop
    J_norms_history = []
    Energy_history = []
    
    for i in range(steps):
        state = integrator.step(state, potentials, gauge_field=gauge_field)
        
        # Check conservation
        # J should have norm 1.0 (or whatever initial was)
        J_norms = torch.norm(state.J, dim=-1)
        mean_norm = J_norms.mean().item()
        std_norm = J_norms.std().item()
        
        # Energy
        E_pot, _ = potentials.compute_forces(state.x)
        E_kin = inertia.kinetic_energy(state.M, state.x).sum()
        E_tot = E_pot + E_kin
        
        J_norms_history.append(mean_norm)
        Energy_history.append(E_tot.item())
        
        if i % 50 == 0:
            print(f"Step {i}: E={E_tot.item():.2f}, J_norm={mean_norm:.6f} (+/- {std_norm:.6f})")
            
    # 4. Analysis
    J_drift = np.max(np.abs(np.array(J_norms_history) - 1.0))
    print(f"Max J Norm Drift: {J_drift:.2e}")
    
    if J_drift < 1e-4:
        print(">>> PASS: Logical Charge Norm Conserved.")
    else:
        print(">>> FAIL: Logical Charge Norm Drifted.")
        
    # Energy shouldn't explode (monotonic decrease due to gamma, or oscillation)
    # If gamma=0.1, should decrease.
    E_start = Energy_history[0]
    E_end = Energy_history[-1]
    print(f"Energy: {E_start:.2f} -> {E_end:.2f}")
    
    if E_end < E_start + 1.0: # Allow small fluctuation but mostly dissipative
         print(">>> PASS: Energy Stable.")
    else:
         print(">>> FAIL: Energy Exploded.")

if __name__ == "__main__":
    test_stability_long()

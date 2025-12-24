"""
Diagnostic Script for Aurora V2.
================================

Runs training with detailed force analysis to diagnose "Frozen Semantics".
Logs:
1. Force Magnitudes (Struct, Sem, Repulsion, Anchor).
2. Adaptive DT.
3. Energy Components.

Based on train_aurora_v2.py.
"""

import sys
import os
import argparse
import torch
import numpy as np
import time
import pickle

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora import (
    AuroraDataset,
    PhysicsConfig,
    PhysicsState,
    ContactIntegrator,
    CompositePotential,
    SpringPotential,
    RadiusAnchorPotential,
    GatedRepulsionPotential,
    RadialInertia
)
from aurora.geometry import random_hyperbolic_init, project_to_tangent

def diagnose_forces(state, potentials, names, device):
    """
    Compute and log force norms for each potential component.
    """
    x = state.x
    
    headers = ["Component", "Energy", "GradNorm(Mean)", "GradNorm(Max)", "Collision(%)"]
    print(f"\n--- Force Diagnosis (Step {state.step if hasattr(state, 'step') else '?'}) ---")
    print(f"{headers[0]:<15} | {headers[1]:<10} | {headers[2]:<15} | {headers[3]:<15} | {headers[4]:<12}")
    print("-" * 80)
    
    total_grad = torch.zeros_like(x)
    
    for name, pot in zip(names, potentials):
        e, g = pot.compute_forces(x)
        
        f_tangent = -project_to_tangent(x, g)
        f_mags = torch.norm(f_tangent, dim=-1)
        mean_f = f_mags.mean().item()
        max_f = f_mags.max().item()
        energy = e.item()
        
        collision_rate = "-"
        # Special check for Gated Repulsion
        if isinstance(pot, GatedRepulsionPotential):
            # Re-run dist calculation to count active contacts
            # This is duplicate work but fine for debug script
            N = x.shape[0]
            # Use same seed? No, but stochastic estimate is fine
            u_idx = torch.randint(0, N, (N * pot.num_neg,), device=device)
            v_idx = torch.randint(0, N, (N * pot.num_neg,), device=device)
            xu = x[u_idx]; xv = x[v_idx]
            J = torch.ones(x.shape[-1], device=device); J[0] = -1.0
            inner = (xu * xv * J).sum(dim=-1)
            inner = torch.clamp(inner, max=-1.0 - 1e-7)
            dist = torch.acosh(-inner)
            mask = dist < pot.epsilon
            rate = mask.float().mean().item() * 100.0
            collision_rate = f"{rate:.2f}%"
        
        print(f"{name:<15} | {energy:<10.1f} | {mean_f:<15.2e} | {max_f:<15.2e} | {collision_rate:<12}")
        
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cilin")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200, help="Diagnostic run length")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--semantic_path", type=str, default=None)
    
    # Staged Training Args (Diagnostic Mode: Usually run specific snapshot)
    parser.add_argument("--k_struct", type=float, default=1.0)
    parser.add_argument("--k_sem", type=float, default=0.5)
    parser.add_argument("--k_rep", type=float, default=5.0) # Repulsion A
    parser.add_argument("--epsilon", type=float, default=0.1) # Repulsion Eps
    
    # Test Modes
    parser.add_argument("--sem_amp_test", action="store_true", help="Run Semantic Amplification Test")
    
    args = parser.parse_args()
    
    # Amplification Test Overrides
    if args.sem_amp_test:
        print(">>> MODE: Semantic Amplification Test (struct=0.01, sem=5.0)")
        args.k_struct = 0.01
        args.k_sem = 5.0
        args.steps = 200
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Aurora V2 Diagnostic on {device} (Epsilon={args.epsilon})")
    
    # 1. Load Data
    ds = AuroraDataset(args.dataset, limit=args.limit)
    print(f"  Nodes: {ds.num_nodes}")
    
    # 2. Physics Init
    N = ds.num_nodes
    dim = 5
    
    x_init = random_hyperbolic_init(N, dim, scale=0.1, device=device)
    G = torch.zeros(N, dim, dim, device=device)
    G[..., 0] = x_init
    G[..., 1:] = torch.eye(dim, device=device).unsqueeze(0).repeat(N, 1, 1)[..., 1:]
    
    from aurora.geometry import renormalize_frame
    G, _ = renormalize_frame(G)
    
    M = torch.zeros(N, dim, dim, device=device)
    z = torch.zeros(N, device=device)
    
    state = PhysicsState(G=G, M=M, z=z)
    
    # 3. Potentials
    potentials = []
    names = []
    
    # Structural
    edges_struct_t = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
    pot_struct = SpringPotential(edges_struct_t, k=args.k_struct, l0=0.0)
    potentials.append(pot_struct)
    names.append("Structure")
    
    # Semantic
    pot_sem = None
    if args.semantic_path:
        sem_edges = ds.load_semantic_edges(args.semantic_path)
        if sem_edges:
            sem_indices = [(u, v) for u, v, w in sem_edges]
            sem_t = torch.tensor(sem_indices, dtype=torch.long, device=device)
            pot_sem = SpringPotential(sem_t, k=args.k_sem)
            potentials.append(pot_sem)
            names.append("Semantic")
            
    # Volume Control
    depths = torch.tensor(ds.depths, dtype=torch.float32, device=device)
    target_radii = 0.5 + depths * 0.5
    potentials.append(RadiusAnchorPotential(target_radii, lamb=0.1))
    names.append("Anchor")
    
    # Repulsion (Gated)
    potentials.append(GatedRepulsionPotential(A=args.k_rep, epsilon=args.epsilon, num_neg=10))
    names.append("GatedRep")
    
    oracle = CompositePotential(potentials)
    
    # 4. Integrator
    inertia = RadialInertia(alpha=5.0)
    config = PhysicsConfig(dt=0.05, gamma=0.5, adaptive=True, solver_iters=10)
    integrator = ContactIntegrator(config, inertia)

    # 5. Loop
    print(f"Starting Diagnostic Simulation (Steps={args.steps})...")
    
    for i in range(args.steps):
        state.step = i # Hack to pass step info
        state = integrator.step(state, oracle)
        
        if i % 20 == 0: 
            avg_r = torch.mean(torch.acosh(state.x[:, 0])).item()
            dt = state.diagnostics.get('dt', config.dt)
            print(f"Step {i}: R={avg_r:.2f}, dt={dt:.1e}, E={state.diagnostics.get('energy',0):.1f}")
            
            # Run Force Diagnosis
            diagnose_forces(state, potentials, names, device)
            
            # --- Additional Metrics (Min Dist & Force Ratio) ---
            # Use random sampling for dist stats to save time
            idx_sample = torch.randint(0, state.x.shape[0], (1000,), device=device)
            idx_sample_2 = torch.randint(0, state.x.shape[0], (1000,), device=device)
            xu = state.x[idx_sample]; xv = state.x[idx_sample_2]
            J = torch.ones(state.x.shape[-1], device=device); J[0] = -1.0
            inner = (xu * xv * J).sum(dim=-1)
            dist_sample = torch.acosh(torch.clamp(-inner, min=1.0001))
            d_p1 = torch.quantile(dist_sample, 0.01).item()
            d_min = dist_sample.min().item()
            print(f">>> Dist Stats: Min={d_min:.4f}, P1={d_p1:.4f} (Target Eps={args.epsilon})")
            
            # Force Ratio
            # We need to capture mean_f from diagnose_forces.
             # Ideally diagnose_forces should return a dict.
            # But calculating here roughly is fine.
            pass
            
    # Debug Save
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/aurora_diag.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'x': state.x.detach().cpu().numpy(),
            'nodes': ds.nodes,
            'vocab': ds.vocab.word_to_id
        }, f)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()

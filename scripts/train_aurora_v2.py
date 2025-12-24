"""
Train Aurora V2 (Clean Architecture).
=====================================

New entry point for training the Aurora Interaction Engine (HEI Phase II).
Uses the clean `src/aurora` package with decoupled data loading.

Usage:
    python scripts/train_aurora_v2.py --dataset cilin --limit 5000 --device cuda
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
    RepulsionPotential,
    RadialInertia
)
from aurora.geometry import random_hyperbolic_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cilin", choices=["cilin", "openhow"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--semantic_path", type=str, default=None, help="Path to str-based semantic edges pickle")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Aurora V2 Training on {device}")
    
    # 1. Load Data
    print("Loading Dataset...")
    ds = AuroraDataset(args.dataset, limit=args.limit)
    print(f"  Nodes: {ds.num_nodes}")
    print(f"  Struct Edges: {len(ds.edges_struct)}")
    
    # 2. Physics Init
    N = ds.num_nodes
    dim = 5
    
    # Init Frame (G)
    # x on H^n, M=0
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
    
    # Structural
    edges_struct_t = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
    potentials.append(SpringPotential(edges_struct_t, k=1.0, l0=0.0))
    
    # Semantic (Optional)
    if args.semantic_path:
        sem_edges = ds.load_semantic_edges(args.semantic_path)
        if sem_edges:
            sem_indices = [(u, v) for u, v, w in sem_edges]
            sem_t = torch.tensor(sem_indices, dtype=torch.long, device=device)
            potentials.append(SpringPotential(sem_t, k=0.5))
            print(f"  Added {len(sem_indices)} Semantic Edges.")
            
    # Volume Control
    depths = torch.tensor(ds.depths, dtype=torch.float32, device=device)
    target_radii = 0.5 + depths * 0.5
    potentials.append(RadiusAnchorPotential(target_radii, lamb=0.1))
    
    # Repulsion
    potentials.append(RepulsionPotential(A=5.0, sigma=1.0))
    
    oracle = CompositePotential(potentials)
    
    # 4. Integrator
    inertia = RadialInertia(alpha=5.0)
    config = PhysicsConfig(dt=0.05, gamma=0.5, adaptive=True, solver_iters=10)
    integrator = ContactIntegrator(config, inertia)

    # 5. Loop
    print("Starting Simulation...")
    start_t = time.time()
    for i in range(args.steps):
        state = integrator.step(state, oracle)
        
        if i % 20 == 0: # More frequent logs for short run
            avg_r = torch.mean(torch.acosh(state.x[:, 0])).item()
            dt = state.diagnostics.get('dt', config.dt)
            print(f"Step {i}: R={avg_r:.2f}, dt={dt:.1e}, E={state.diagnostics.get('energy',0):.1f}")
            
    end_t = time.time()
    print(f"Done. {args.steps} steps in {end_t - start_t:.1f}s")
    
    # Debug Save
    print(f"DEBUG: Saving State. X shape: {state.x.shape}, Nodes: {len(ds.nodes)}")
    
    # Save (Simple)
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/aurora_v2_eval.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'x': state.x.detach().cpu().numpy(),
            'nodes': ds.nodes,
            'vocab': ds.vocab.word_to_id
        }, f)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()

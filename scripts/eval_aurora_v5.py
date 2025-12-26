
"""
Evaluation Script for Aurora v2.0 (Theory 5)
============================================

Analyzes the checkpoint produced by `train_aurora_v2.py` with Logic Dynamics enabled.
Metrics:
1. Gauge Alignment (Link Prediction proxy via J).
2. Curvature Distribution (Holonomy).
3. Structural Fidelity.
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.gauge import GaugeField
from aurora.geometry import minkowski_inner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .pkl checkpoint")
    parser.add_argument("--semantic_path", type=str, default=None, help="Path to semantic edges (must match training)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        data = pickle.load(f)
        
    x = torch.tensor(data['x'], device=device)
    J = torch.tensor(data['J'], device=device) if data['J'] is not None else None
    
    print(f"Loaded State: N={x.shape[0]}")
    if J is None:
        print("WARNING: Checkpoint does not contain Logical Charge J.")
        return

    print(f"Logical Charge J: Shape {J.shape}")
    
    # 1. Load Dataset to get edges
    from aurora import AuroraDataset
    dataset_name = data['config'].get('dataset', 'cilin')
    print(f"Loading Dataset {dataset_name} for Structural Edges...")
    ds = AuroraDataset(dataset_name, limit=data['config'].get('limit'))
    edges_struct = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
    
    # 2. Reconstruct GaugeField Topology (Union)
    sem_path = args.semantic_path
    edges_all = edges_struct
    
    if sem_path:
        sem_edges = ds.load_semantic_edges(sem_path, split=data['config'].get('split', 'all'))
        if sem_edges:
            sem_indices = [(u, v) for u, v, w in sem_edges]
            sem_t = torch.tensor(sem_indices, dtype=torch.long, device=device)
            edges_all = torch.cat([edges_struct, sem_t], dim=0)
            print(f"Loaded {len(sem_indices)} semantic edges. Total Gauge Edges: {edges_all.shape[0]}")
            
    logical_dim = data['config'].get('logical_dim', 3)
    gauge_field = GaugeField(edges_all, logical_dim, group='SO').to(device)
    
    if 'gauge_field' in data and data['gauge_field'] is not None:
        print("Loading Gauge Field parameters from checkpoint...")
        gauge_field.load_state_dict(data['gauge_field'])
    else:
        print("WARNING: Checkpoint missing 'gauge_field' state. Using random initialization.")
    
    # 3. Compute Alignment Score
    # Alignment = Mean <J_v, U_uv J_u>
    with torch.no_grad():
        u = edges_all[:, 0]
        v = edges_all[:, 1]
        U_all = gauge_field.get_U()
        
        J_u = J[u]
        J_v = J[v]
        
        # Transport J_u
        J_u_trans = torch.matmul(U_all, J_u.unsqueeze(-1)).squeeze(-1)
        
        # Dot product
        alignment = torch.sum(J_v * J_u_trans, dim=-1) # (E,)
        mean_align = torch.mean(alignment).item()
        
        # Reference (No U):
        raw_align = torch.mean(torch.sum(J_v * J_u, dim=-1)).item()
        
        print(f"\n--- Gauge Alignment Results ---")
        print(f"Mean Gauge Alignment: {mean_align:.4f} (Ideally close to 1.0)")
        print(f"Raw J Correlation:    {raw_align:.4f} (Baseline without U)")
        print(f"Gauge Gain:           {mean_align - raw_align:+.4f}")
    
    # 4. Compute Curvature (Holonomy)
    print(f"\n--- Curvature Analysis ---")
    with torch.no_grad():
        Omega, triangles = gauge_field.compute_curvature()
        if triangles.shape[0] > 0:
            omega_norms = torch.norm(Omega.reshape(Omega.shape[0], -1), dim=-1)
            print(f"Triangles Found: {triangles.shape[0]}")
            print(f"Mean Curvature Norm (||Ω||): {omega_norms.mean().item():.4e}")
            print(f"Max Curvature Norm:          {omega_norms.max().item():.4e}")
            print(f"Curvature Density:           {(omega_norms > 1e-4).float().mean().item()*100:.2f}% (>1e-4)")
        else:
            print("No triangles found in the graph structure. (Curvature is 0 by topology).")

    # 5. J-Distribution Statistics
    print(f"\n--- J-Vector Statistics ---")
    j_norms = torch.norm(J, dim=-1)
    print(f"J Norms: Mean={j_norms.mean():.4f}, Std={j_norms.std():.4f}")
    j_global_mean = torch.mean(J, dim=0)
    print(f"Global J Mean: {j_global_mean.cpu().numpy()}")
    
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()

"""
Aurora Logic Stress Test - Paradox Evaluation.
==============================================

Detects "Logical Frustration" (Curvature) in 2-cycles.
Hypothesis: 
- Consistent relations (A=B, B=A) have Holonomy ~ Identity.
- Paradoxical/Asymmetric relations (A->B, B!->A) have High Holonomy.

Since we don't have explicit "Not" relations trained, we analyze the
distribution of 2-cycle frustration in the existing Semantic Graph.
High frustration edges indicate "Non-commutative" or "Complex" relationships.
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    print("Loading Checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    # J not strictly needed for Holonomy, just U.
    gf_state = ckpt['gauge_field']
    
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural', input_dim=5).to(device)
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # 2. Sample Edges (Semantic)
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    print(f"Sampling {args.samples} edges from {len(sem_edges)}...")
    
    indices = np.random.choice(len(sem_edges), min(args.samples, len(sem_edges)), replace=False)
    sample_edges = [sem_edges[i] for i in indices]
    
    frustrations = []
    
    print("Computing 2-Cycle Holonomy (Reciprocity)...")
    
    # Batch Compute? Or Loop? Loop is safer for logic clarity.
    # Edge: u -> v.
    # Cycle: u -> v -> u.
    # H = U_{vu} * U_{uv}
    
    u_list = [e[0] for e in sample_edges]
    v_list = [e[1] for e in sample_edges]
    
    u_tensor = torch.tensor(u_list, dtype=torch.long, device=device)
    v_tensor = torch.tensor(v_list, dtype=torch.long, device=device)
    
    # Forward: u -> v
    edges_fwd = torch.stack([u_tensor, v_tensor], dim=1)
    U_fwd = gauge_field.get_U(x=x, edges=edges_fwd) # (B, k, k)
    
    # Backward: v -> u
    edges_bwd = torch.stack([v_tensor, u_tensor], dim=1)
    U_bwd = gauge_field.get_U(x=x, edges=edges_bwd) # (B, k, k)
    
    # Holonomy H = U_bwd @ U_fwd
    H = torch.matmul(U_bwd, U_fwd)
    
    # Frustration = || I - H ||
    I = torch.eye(3, device=device).unsqueeze(0)
    diff = torch.norm(I - H, dim=(1,2))
    frustrations = diff.cpu().numpy()
    
    mean_f = np.mean(frustrations)
    max_f = np.max(frustrations)
    
    print(f"\nMean Frustration: {mean_f:.4f}")
    print(f"Max Frustration:  {max_f:.4f}")
    
    # Analyze Top Frustrated Pairs (Paradox Candidates)
    idxs = np.argsort(frustrations)[::-1][:10]
    print("\nTop 10 Most Frustrated 'Paradox' Pairs:")
    for i in idxs:
        u_id = u_list[i]
        v_id = v_list[i]
        u_str = ds.nodes[u_id].split(":")[1] if ":" in ds.nodes[u_id] else ds.nodes[u_id]
        v_str = ds.nodes[v_id].split(":")[1] if ":" in ds.nodes[v_id] else ds.nodes[v_id]
        print(f"{u_str} <-> {v_str} : F={frustrations[i]:.4f}")
        
    # Validation: Compare with Random Pairs (Non-Edges)
    # Hypothesis: Random pairs might have RANDOM holonomy, 
    # but connected pairs should have LOW holonomy unless paradoxical.
    
    print("\nControl: Random Non-Edge Pairs...")
    rand_u = torch.randint(0, len(ds.nodes), (100,), device=device)
    rand_v = torch.randint(0, len(ds.nodes), (100,), device=device)
    
    e_f = torch.stack([rand_u, rand_v], dim=1)
    e_b = torch.stack([rand_v, rand_u], dim=1)
    
    Uf = gauge_field.get_U(x=x, edges=e_f)
    Ub = gauge_field.get_U(x=x, edges=e_b)
    Hr = torch.matmul(Ub, Uf)
    fr_rand = torch.norm(I[:100] - Hr, dim=(1,2)).cpu().numpy()
    
    print(f"Random Pair Mean Frustration: {np.mean(fr_rand):.4f}")
    
    if np.mean(frustrations) < np.mean(fr_rand):
        print("\nResult: Semantic Edges fit Gauge Field better than Random (Lower Frustration).")
        print("High Frustration Edges are likely 'Wormholes' or 'Paradoxes'.")
    else:
        print("\nResult: Semantic Edges are highly frustrated (Complex Topology).")

if __name__ == "__main__":
    main()

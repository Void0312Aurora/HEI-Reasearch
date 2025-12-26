"""
Script: Audit Global Cycle Consistency (Gate A+).
Checks if "Wormholes" introduce global topological frustration.
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField
from aurora.topology import GlobalCycleMonitor
from aurora.geometry import dist_hyperbolic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dataset", type=str, default="cilin")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    print(f"Loading checkpoint {args.checkpoint}...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
        
    x = torch.tensor(ckpt['x'], dtype=torch.float32, device=device)
    
    # Reconstruct Gauge (Neural)
    gf_state = ckpt['gauge_field']
    dummy_edges = torch.zeros((1, 2), dtype=torch.long, device=device)
    gauge_field = GaugeField(dummy_edges, logical_dim=3, input_dim=5, backend_type='neural').to(device)
    # Filter buffers
    filtered_state = {k: v for k, v in gf_state.items() if not (k.startswith('edges') or k.startswith('tri_'))}
    gauge_field.load_state_dict(filtered_state, strict=False)
    
    # 2. Reconstruct Graph to find cycles
    # We need the graph structure used in training.
    # Checkpoint doesn't store edge list explicitly for reconstruction?
    # It stores 'config' -> split.
    # We should reconstruct from dataset.
    
    ds = AuroraDataset(args.dataset, limit=args.limit)
    
    # Base Graph: Structural Edges
    print("Building Base Graph...")
    adj = {}
    for u, v in ds.edges_struct:
        if u not in adj: adj[u] = set()
        if v not in adj: adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
        
    # We should also add Semantic Edges that were in the training set?
    # Cycle Monitor usually checks if NEW edges conflict with EXISTING graph.
    # So "Existing Graph" = Struct + Pre-existing Sem.
    # For this audit, let's use Struct + ALL Semantics as the 'Fabric'.
    # Actually, Wormholes are the edges we want to TEST. They shouldn't be in the pathfinding graph ideally.
    # Step:
    # 1. Identify Wormholes (Dist > 3.0, Align > 0.9).
    # 2. Remove them from graph (if present).
    # 3. Find path in remaining graph.
    # 4. Compute Holonomy of loop.
    
    # Let's verify Mined Edges from Cycle 3.
    # We don't have the list of "Just Mined" edges easily unless we diff.
    # But we can screen all semantic edges.
    
    # Load Semantic Edges used in Cycle 3
    sem_path = args.checkpoint.replace("checkpoint_cycle_", "semantic_state_").replace(".pkl", ".pkl")
    # Actually logic might differ, let's assume standard path or reconstruct
    # Simpler: Screen ALL pairs in the semantic dataset using the checkpoint.
    
    # Let's load the semantic edges from the workspace file matching the cycle
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    
    print(f"Loading semantic edges from {sem_file}...")
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    # 3. Identify Wormholes
    print("Identifying Wormholes...")
    wormholes = []
    
    sem_u = torch.tensor([e[0] for e in sem_edges], dtype=torch.long, device=device)
    sem_v = torch.tensor([e[1] for e in sem_edges], dtype=torch.long, device=device)
    
    # Calc Dist
    xu = x[sem_u]
    xv = x[sem_v]
    dists = dist_hyperbolic(xu, xv)
    
    # Calc Align
    edges_t = torch.stack([sem_u, sem_v], dim=1)
    U_pred = gauge_field.get_U(x=x, edges=edges_t)
    J = torch.tensor(ckpt['J'], device=device)
    Ju = J[sem_u]
    Jv = J[sem_v]
    Ju_trans = torch.matmul(U_pred, Ju.unsqueeze(-1)).squeeze(-1)
    aligns = torch.sum(Jv * Ju_trans, dim=-1)
    
    # Filter
    mask = (dists > 3.0) & (aligns > 0.9)
    wormhole_indices = torch.nonzero(mask).squeeze()
    
    if len(wormhole_indices.shape) == 0:
         # scalar
         wormhole_indices = wormhole_indices.unsqueeze(0)
         
    print(f"Found {len(wormhole_indices)} Wormholes for Audit.")
    
    if len(wormhole_indices) == 0:
        return

    # 4. Monitor
    monitor = GlobalCycleMonitor(adj, gauge_field, device=device)
    
    candidates = edges_t[wormhole_indices]
    
    # Find Cycles
    cycles = monitor.find_cycles_for_candidates(candidates, max_depth=args.depth, sample_size=args.samples)
    
    if not cycles:
        print("No cycles found (Wormholes connect disconnected components or path > depth).")
        return
        
    print(f"Found {len(cycles)} cycles. Computing Holonomy...")
    frustrations = monitor.compute_holonomy(cycles, x)
    
    mean_f = frustrations.mean().item()
    max_f = frustrations.max().item()
    
    print("\n=== Global Cycle Audit (Gate A+) ===")
    print(f"Depth Limit: {args.depth}")
    print(f"Analyzed Cycles: {len(cycles)}")
    print(f"Mean Frustration (||Log H||): {mean_f:.4f}")
    print(f"Max Frustration:              {max_f:.4f}")
    
    # Interpretation
    # If Mean < 0.1, Topological Conflict is Low.
    # If Mean > 0.5, Significant Frustration.
    
    if mean_f < 0.2:
        print(">>> RESULT: LOW Global Frustration. System is topologically consistent.")
    else:
        print(">>> RESULT: HIGH Global Frustration. Topology conflict detected.")

if __name__ == "__main__":
    main()

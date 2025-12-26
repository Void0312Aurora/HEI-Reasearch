"""
Aurora Long-Range Reasoning Stress Test.
========================================

Measures Semantic Coherence (Alignment) over long logic chains.
Hypothesis: 
- Geometric Transport maintains coherence.
- Alignment should decay slowly with depth.

Metric:
- Select pairs (u, v) at distance k.
- Find shortest path P = [u, ..., v].
- Transport J_u along P to get J_pred.
- Score = <J_pred, J_v>.
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
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    print("Loading Checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    gf_state = ckpt['gauge_field']
    
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural', input_dim=5).to(device)
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # 2. Build Graph
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    G = nx.Graph()
    for u, v in ds.edges_struct: G.add_edge(u, v)
    for u, v, w in sem_edges: G.add_edge(u, v)
    
    # 3. Sample Pairs by Distance
    print("Sampling paths...")
    depths = [2, 4, 6]
    results = {k: [] for k in depths}
    
    # Sample random nodes
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    N = min(100, len(nodes))
    
    for start_node in nodes[:N]:
        # BFS to find nodes at depths
        lengths = nx.single_source_shortest_path_length(G, start_node, cutoff=max(depths))
        
        for k in depths:
            targets = [n for n, d in lengths.items() if d == k]
            if not targets: continue
            
            # Pick one target per depth per start node
            target_node = np.random.choice(targets)
            
            # Get Path
            path = nx.shortest_path(G, start_node, target_node)
            
            # Transport
            J_curr = J[start_node].clone() # (3,)
            
            # Iterate path edges
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                
                # Get U
                edges_t = torch.tensor([[u, v]], dtype=torch.long, device=device)
                U = gauge_field.get_U(x=x, edges=edges_t)[0] # (3, 3)
                
                J_curr = torch.matmul(U, J_curr)
                
            # Alignment with target
            J_target = J[target_node]
            alignment = torch.sum(J_curr * J_target).item()
            results[k].append(alignment)
            
    print("\n--- Long-Range Coherence Results ---")
    for k in depths:
        aligns = results[k]
        if not aligns:
            print(f"Depth {k}: No samples.")
        else:
            mean_a = np.mean(aligns)
            std_a = np.std(aligns)
            print(f"Depth {k}: Mean Alignment = {mean_a:.4f} +/- {std_a:.4f}")
            
    # Baseline comparison? Ideally we compare to random.
    # Max possible alignment is 1.0 (if normalized).
    # Since J is not strictly normalized to 1 in the training (Potentials usually normalize, but let's check).
    # Alignment might be unnormalized.
    
if __name__ == "__main__":
    main()

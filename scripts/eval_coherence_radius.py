"""
Aurora Coherence Radius Systematization.
========================================

Systematically measures coherence decay over depth to define R_c.
Also validates Wormhole compression hypothesis.
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
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    gf_state = ckpt['gauge_field']
    
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5).to(device)
    
    # Patch Backend to match Checkpoint Architecture (Legacy)
    from aurora.gauge import GaugeConnectionBackend
    import torch.nn as nn
    from aurora.geometry import log_map
    from typing import Optional
    
    class LegacyNeuralBackend(GaugeConnectionBackend):
        def __init__(self, input_dim=5, logical_dim=3, hidden_dim=64):
            super().__init__(logical_dim)
            input_size = 3 * input_dim
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), # No expansion to 128
                nn.Tanh(),
                nn.Linear(hidden_dim, logical_dim * logical_dim)
            )
            self.rel_embed = None
            
        def get_omega(self, edges=None, x=None, relation_ids=None):
             # Wrapper
             return self.forward(x=x, edges_uv=edges, relation_ids=relation_ids)
             
        def forward(self, edge_indices=None, x=None, edges_uv=None, relation_ids=None):
            if x is None: raise ValueError("Need x")
            u, v = edges_uv[:, 0], edges_uv[:, 1]
            swap_mask = u > v
            u_canon = torch.where(swap_mask, v, u)
            v_canon = torch.where(swap_mask, u, v)
            xu = x[u_canon]
            xv = x[v_canon]
            v_uv = log_map(xu, xv)
            feat = torch.cat([xu, xv, v_uv], dim=-1)
            
            out = self.net(feat)
            out = 3.0 * torch.tanh(out)
            
            # Reshape and Skew-Symmetric
            out = out.view(-1, self.logical_dim, self.logical_dim)
            omega_canon = 0.5 * (out - out.transpose(1, 2))
            
            negation = torch.where(swap_mask, -1.0, 1.0).view(-1, 1, 1)
            return omega_canon * negation
            
    # Inject Legacy Backend
    gauge_field.backend = LegacyNeuralBackend().to(device)
    
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # 2. Build Graph (Structural + Semantic)
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    G = nx.Graph()
    for u, v in ds.edges_struct: G.add_edge(u, v)
    G_sem = G.copy()
    for u, v, w in sem_edges: G_sem.add_edge(u, v)
    
    # 3. Measure Coherence vs Depth (on G_struct only, to isolate 'Logical Depth')
    # Because G_sem has shortcuts (Wormholes), we want to measure decay on the "Long Path".
    # Wait, if we measure on G_struct, we might find pairs that are far apart.
    # But usually we want to measure how the *trained* system behaves.
    # The trained system uses G_sem for transport?
    # No, Gauge Field is defined on edges. Transport happens along a PATH.
    # We can choose a path in G_struct (Logical Hierarchy) and see if Coherence survives.
    
    print("Measuring Coherence Decay on Structural Tree (Sample=20)...")
    depths = list(range(1, 9))
    coherence_stats = {d: [] for d in depths}
    
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    
    for idx, start_node in enumerate(nodes[:20]):
        if idx % 5 == 0: print(f"Processing node {idx}...")
        try:
            paths = nx.single_source_shortest_path(G, start_node, cutoff=8)
        except Exception:
            continue # Disconnected?
            
        # Group by depth
        by_depth = {d: [] for d in range(1, 9)}
        for target, path in paths.items():
            d = len(path) - 1
            if 1 <= d <= 8:
                by_depth[d].append((target, path))
                
        # Sample 5 per depth per start node
        for d in range(1, 9):
            candidates = by_depth[d]
            if not candidates: continue
            
            # Subsample
            indices = np.random.choice(len(candidates), min(5, len(candidates)), replace=False)
            selected = [candidates[i] for i in indices]
            
            for target, path in selected:
                # Transport
                J_curr = J[start_node].clone()
                valid_path = True
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    edges_t = torch.tensor([[u, v]], dtype=torch.long, device=device)
                    U = gauge_field.get_U(x=x, edges=edges_t)[0]
                    J_curr = torch.matmul(U, J_curr)
                    
                if valid_path:
                    align = torch.sum(J_curr * J[target]).item()
                    coherence_stats[d].append(align)
                
    # 4. Results
    print("\n--- Coherence Radius Analysis ---")
    vals = []
    errs = []
    for d in depths:
        data = coherence_stats[d]
        if data:
            mean = np.mean(data)
            std = np.std(data)
            print(f"Depth {d}: {mean:.4f} +/- {std:.4f}")
            vals.append(mean)
            errs.append(std)
        else:
            vals.append(0)
            errs.append(0)
            
    # Find Radius
    R_c = 0
    for d, v in zip(depths, vals):
        if v < 0.5: # Threshold
             R_c = d - 1 + (vals[d-1-1] - 0.5)/(vals[d-1-1] - v) # Linear interp?
             # Simple: Last depth > 0.5.
             if v > 0.5: R_c = d
             else: break
             
    print(f"\nEstimated Coherence Radius R_c: ~{R_c} hops")
    
    # 5. Wormhole Validation
    # Identify Semantic Edges (Wormholes)
    # Check graph distance of these pairs in G_struct (without wormholes).
    # Check if they are compressed to 1 hop (Direct Edge) which is < R_c.
    
    print("\n--- Wormhole Utility Validation ---")
    wormhole_dists = []
    for u, v, w in sem_edges[:200]:
        try:
            d_struct = nx.shortest_path_length(G, u, v)
            wormhole_dists.append(d_struct)
        except:
            pass
            
    avg_shortcut = np.mean(wormhole_dists)
    print(f"Average Logical Distance of Wormholes: {avg_shortcut:.2f} hops")
    print(f"Wormholes compress {avg_shortcut:.2f} hops -> 1 hop.")
    
    if avg_shortcut > R_c:
        print("Result: Wormholes bridge distances greater than Coherence Radius!")
    else:
        print("Result: Wormholes strictly optimize efficiency within Radius.")

if __name__ == "__main__":
    main()

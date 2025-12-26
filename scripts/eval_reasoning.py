"""
Geometric Reasoning Benchmark (Phase 12).
========================================

Tests the reasoning capabilities of the Neural Gauge Field.
1. **Analogy**: A:B :: C:? (Transport Vector v_AB from C to find D)
2. **Pathfinding**: Compare Tree Distance vs Gauge Distance (Shortest Path on Graph).

Ref: `docs/plan/理论基础-5.md`
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse
import networkx as nx
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField
from aurora.geometry import dist_hyperbolic

def build_graph(ds, sem_edges):
    G = nx.Graph()
    # Add structural edges
    for u, v in ds.edges_struct:
        G.add_edge(u, v, weight=1.0)
    # Add semantic edges
    for u, v, w in sem_edges:
        G.add_edge(u, v, weight=0.1) # Prefer semantic shortcuts?
    return G

def get_analogy_triplets(ds, num_samples=100):
    """
    Generate synthetic analogy tasks from Hierarchy.
    Sibling Analogy: A:Parent :: B:Parent (A and B are siblings)
    Or Cousin Analogy: A:B :: C:D (A,B are siblings; C,D are siblings)
    """
    # Simple strategy: Cousin Analogy
    # Find two pairs of siblings (a1, a2) and (b1, b2)
    # Task: a1 : a2 :: b1 : ? (Target b2)
    # Ideally a1 and b1 should be 'related' (e.g. same category)
    # But for geometry test, random independent pairs are fine to test transport mechanics.
    
    # 1. Build Sibling Map
    parent_map = {} # u -> p
    children_map = {} # p -> [c1, c2...]
    for u, v in ds.edges_struct:
        # Assuming tree structure where parent < child? No guarantees.
        # But Cilin has ordering?
        # Let's just treat neighbors as siblings if they share a neighbor.
        # This is expensive.
        # Use structure: Cilin tree is loaded.
        pass
    
    # Fallback: Just pick random semantic edges (synonyms) as "A:B"
    # A:B (Synonym) :: C:D (Synonym)
    # If A close to B, and C close to D.
    # Transport (B shift) to C should land near D?
    # In Hyperbolic, vector A->B is specific to tangent space at A.
    # Transport A->C, then apply.
    pass
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint {args.checkpoint}...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
        
    x = torch.tensor(ckpt['x'], device=device)
    
    # Reconstruct Gauge
    gf_state = ckpt['gauge_field']
    dummy_edges = torch.zeros((1, 2), dtype=torch.long, device=device)
    gauge_field = GaugeField(dummy_edges, logical_dim=3, input_dim=5, backend_type='neural').to(device)
    filtered_state = {k: v for k, v in gf_state.items() if not (k.startswith('edges') or k.startswith('tri_'))}
    gauge_field.load_state_dict(filtered_state, strict=False)
    
    # Load Data
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # Identify Wormholes for Testing
    # We want to test pathfinding THROUGH wormholes.
    # Pick random distant pairs.
    # Path 1: Shortest Path in Tree (Structure Only).
    # Path 2: Shortest Path in Graph (Structure + Wormholes).
    
    # Load Wormholes (Semantics from latest cycle)
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    # 1. Build Graphs
    print("Building Graphs...")
    G_tree = nx.Graph()
    G_tree.add_edges_from(ds.edges_struct)
    
    G_aug = nx.Graph()
    G_aug.add_edges_from(ds.edges_struct)
    for u, v, w in sem_edges:
        G_aug.add_edge(u, v)
        
    print(f"Tree Edges: {G_tree.number_of_edges()}")
    print(f"Augmented Edges: {G_aug.number_of_edges()}")
    
    # 2. Pathfinding Test
    # Sample random pairs
    pairs = []
    print(f"Sampling {args.samples} pairs...")
    for _ in range(args.samples):
        u = np.random.randint(0, ds.num_nodes)
        v = np.random.randint(0, ds.num_nodes)
        if u != v:
            pairs.append((u, v))
            
    # Compare Path Lengths (Hops)
    # Ideally: Augmented graph has significantly shorter paths for some pairs.
    improved = 0
    total_reduction = 0
    
    # Energy Calculation requires Transport.
    # Path Energy = Sum (1 - <v_next, U v_curr>) ?
    # Actually, "Geometric Energy" of a path = Length?
    # Or "Transport Cost".
    # Let's stick to HOP Count for now as proxy for "Connectivity".
    
    print("\n=== Pathfinding Benchmark ===")
    
    wormhole_hits = 0
    
    for u, v in tqdm(pairs):
        try:
            len_tree = nx.shortest_path_length(G_tree, u, v)
        except:
            len_tree = 999
            
        try:
            path_aug = nx.shortest_path(G_aug, u, v)
            len_aug = len(path_aug) - 1
            
            # Check if path uses semantic edges
            uses_sem = False
            for i in range(len(path_aug)-1):
                p1, p2 = path_aug[i], path_aug[i+1]
                if G_aug.has_edge(p1, p2) and not G_tree.has_edge(p1, p2):
                     uses_sem = True
                     break
            
            if uses_sem:
                wormhole_hits += 1
            
        except:
            len_aug = 999
            
        if len_aug < len_tree:
            improved += 1
            total_reduction += (len_tree - len_aug)
            
    print(f"Pairs Improved: {improved}/{args.samples}")
    print(f"Wormhole Usage freq: {wormhole_hits}/{args.samples}")
    print(f"Avg Hop Reduction (for improved): {total_reduction/improved if improved else 0:.2f}")

if __name__ == "__main__":
    main()

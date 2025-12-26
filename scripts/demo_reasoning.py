"""
Demo: Geometric Reasoning Engine (Track A).
===========================================

Demonstrates the 'Explainable Inference' capability.
1. Finds path between two distant words using ReasoningEngine.
2. Classifies edges as 'Tree-Consistent' or 'Tunnel'.
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
from aurora.reasoning import ReasoningEngine, RelationClassifier
from aurora.geometry import dist_hyperbolic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--u", type=str, help="Start word", default=None)
    parser.add_argument("--v", type=str, help="End word", default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    print("Loading Checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    gf_state = ckpt['gauge_field']
    
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural').to(device)
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    # 2. Data & Graph
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # Build Graph (Struct + Sem)
    # Load Cycle 3 Semantics
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    G = nx.Graph()
    G_tree = nx.Graph()
    # Struct
    for u, v in ds.edges_struct:
        G.add_edge(u, v, type='tree', weight=1.0)
        G_tree.add_edge(u, v)
        
    # Sem
    for u, v, w in sem_edges:
        G.add_edge(u, v, type='sem', weight=0.1) # Shortcut weight
        
    # 3. Initialize Engines
    engine = ReasoningEngine(gauge_field, G, x, device=device)
    classifier = RelationClassifier(gauge_field, device=device)
    
    # 4. Select Nodes
    if args.u and args.v:
        try:
            start = ds.vocab.word_to_id[args.u]
            end = ds.vocab.word_to_id[args.v]
        except KeyError:
            print("Words not found in vocabulary.")
            return
    else:
        # Random distant pair
        while True:
            u_idx = np.random.randint(0, ds.num_nodes)
            v_idx = np.random.randint(0, ds.num_nodes)
            try:
                # Ensure they are connected in tree but distant
                dist = nx.shortest_path_length(G, u_idx, v_idx)
                if dist > 4:
                    start, end = u_idx, v_idx
                    break
            except:
                continue
                
    u_word = ds.nodes[start]
    v_word = ds.nodes[end]
    print(f"\nTask: Pathfinding {u_word} -> {v_word}")
    
    # 5. Run Pathfinding
    result = engine.find_path(start, end)
    path = result['path']
    cost = result['cost']
    
    if not path:
        print("No path found.")
        return
        
    print(f"Path Cost: {cost:.4f}")
    print(f"Hops: {len(path)-1}")
    
    print("\n--- Explainable Path Trace ---")
    for i in range(len(path)-1):
        p1, p2 = path[i], path[i+1]
        w1 = ds.nodes[p1]
        w2 = ds.nodes[p2]
        
        edge_data = G.get_edge_data(p1, p2)
        edge_source = edge_data.get('type', 'unknown')
        
        # Classify
        # Use simple Tree check to classify 'Tunnel' for demo
        # Because Classifier requires J and Tree object passed weirdly
        # Improve Classifier API later
        
        # Classifier call
        # Mocking Tree Graph for classifier (using simple structure)
        # Actually Classifier needs G_tree to check shortcuts.
        # We can pass G as proxy if we assume 'tree' labeled edges.
        
        cls = classifier.classify_edge(p1, p2, J[p1], J[p2], G_tree, x)
        
        arrow = "->"
        if edge_source == 'sem':
            arrow = "=> (Tunnel)"
            
        print(f"{w1:<20} {arrow:<15} {w2:<20} [{cls}]")
        
    print("------------------------------")

if __name__ == "__main__":
    main()

"""
Demo: Native Geometric Language Model (GeometricLM).
====================================================

Showcases "Language without LLMs".
1. **Stream of Consciousness**: Generates associative chains via Gauge Transport.
2. **Cloze Test**: Solves [MASK] using bidirectional transport.
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
from aurora.language import GeometricLM

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
    
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural').to(device)
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    # 2. Data & Graph
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # Identify Cycle idx
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    G = nx.Graph()
    for u, v in ds.edges_struct:
        G.add_edge(u, v, type='tree')
    for u, v, w in sem_edges:
        G.add_edge(u, v, type='sem')
        
    print(f"Graph Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    lm = GeometricLM(gauge_field, G, x, J, device=device)
    
    # 3. Stream Demo
    print("\n=== Demo 1: Stream of Consciousness (Associative Chain) ===")
    seeds = ["Code:Ba01", "Code:Aa01", "Code:Ka01"] # Man, Thing, Activity?
    # Better: Use actual words if possible.
    # Cilin words: '太阳' (Sun), '人' (Man).
    # Find IDs.
    
    seed_words = ['太阳', '英雄', '科学']
    
    for word in seed_words:
        try:
            seed_id = ds.vocab.word_to_id[word]
        except KeyError:
            # Fallback to finding by code
            # Iterate
            seed_id = np.random.randint(0, 1000)
            word = ds.nodes[seed_id]
            
        print(f"\nSeed: {word}")
        traj = lm.generate_stream(seed_id, length=12, temperature=0.2)
        
        # Print Chain
        chain = []
        for item in traj:
            w_idx = item['word_idx']
            w_str = ds.nodes[w_idx]
            if "C:" in w_str: w_str = w_str.split(":")[1]
            chain.append(w_str)
            
        print(" -> ".join(chain))
        
    # 4. Cloze Test Demo
    print("\n=== Demo 2: Cloze Test (Geometric Filling) ===")
    # Format: A [MASK] B
    # Pick A and B s.t. A->M->B exists.
    # We can use the graph to find such pairs.
    
    pairs = []
    # Find 2-hop paths
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    
    count = 0
    for n in nodes:
        neighbors = list(G.neighbors(n))
        if len(neighbors) < 2: continue
        
        # Pick 2 distinct neighbors
        u = neighbors[0]
        v = neighbors[1]
        
        # Mask is n. Task: u [MASK] v -> ?
        pairs.append((u, n, v))
        count += 1
        if count >= 5: break
        
    for u, target, v in pairs:
        u_str = ds.nodes[u].split(":")[1] if "C:" in ds.nodes[u] else ds.nodes[u]
        v_str = ds.nodes[v].split(":")[1] if "C:" in ds.nodes[v] else ds.nodes[v]
        tgt_str = ds.nodes[target].split(":")[1] if "C:" in ds.nodes[target] else ds.nodes[target]
        
        print(f"\nQuery: {u_str} [MASK] {v_str}")
        print(f"Target: {tgt_str}")
        
        pred_idx, score = lm.cloze_test(u, v)
        
        if pred_idx != -1:
            pred_str = ds.nodes[pred_idx]
            if "C:" in pred_str: pred_str = pred_str.split(":")[1]
            print(f"Prediction: {pred_str} (Score: {score:.4f})")
            
            # Check correctness (Exact or Neighbor?)
            if pred_idx == target:
                print("Result: EXACT MATCH")
            elif G.has_edge(pred_idx, target):
                 print("Result: NEAR MATCH (Neighbor)")
            else:
                 print("Result: MISMATCH")
        else:
            print("Prediction: [None]")

if __name__ == "__main__":
    main()

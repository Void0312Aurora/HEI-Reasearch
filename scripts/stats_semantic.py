"""
Semantic Connectivity Analysis.
===============================

Calculates graph statistics for the Semantic Layer.
"""

import sys
import os
import pickle
import numpy as np
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from aurora import AuroraDataset

def analyze_connectivity(dataset_name, semantic_path):
    print("Loading Dataset...")
    ds = AuroraDataset(dataset_name)
    vocab = ds.vocab.word_to_id
    
    print(f"Loading semantic edges from {semantic_path}...")
    sem_edges = ds.load_semantic_edges(semantic_path) # Dataset handles raw map now? 
    # Wait, AuroraDataset.load_semantic_edges was updated in Step 583 to do fuzzy matching?
    # Let's verify. Yes, it was.
    
    if not sem_edges:
        print("No edges loaded.")
        return

    print(f"Loaded {len(sem_edges)} effective edges.")
    
    # Build Graph
    adj = {}
    degrees = np.zeros(ds.num_nodes, dtype=int)
    
    for u, v, w in sem_edges:
        degrees[u] += 1
        degrees[v] += 1
        
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u) # Undirected
        
    # Stats 1: Degree Distribution
    active_nodes = np.sum(degrees > 0)
    print(f"\n--- Degree Statistics ---")
    print(f"Total Nodes: {ds.num_nodes}")
    print(f"Active Semantic Nodes: {active_nodes} ({active_nodes/ds.num_nodes*100:.1f}%)")
    print(f"Mean Degree (global): {np.mean(degrees):.4f}")
    
    if active_nodes > 0:
        active_degs = degrees[degrees > 0]
        print(f"Mean Degree (active): {np.mean(active_degs):.2f}")
        print(f"Median Degree (active): {np.median(active_degs)}")
        print(f"Max Degree: {np.max(active_degs)}")
        print(f"P95 Degree: {np.percentile(active_degs, 95):.1f}")
        
    # Stats 2: Connected Components (BFS)
    print(f"\n--- Connected Components ---")
    visited = set()
    components = []
    
    nodes_with_edges = list(adj.keys())
    
    for start_node in nodes_with_edges:
        if start_node in visited: continue
        
        # BFS
        q = [start_node]
        visited.add(start_node)
        size = 0
        
        idx = 0
        while idx < len(q):
            u = q[idx]; idx+=1
            size += 1
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        components.append(size)
        
    components.sort(reverse=True)
    if components:
        print(f"Number of Components: {len(components)}")
        print(f"Largest Component Size: {components[0]} ({components[0]/ds.num_nodes*100:.2f}%)")
        print(f"Top 5 Component Sizes: {components[:5]}")
        print(f"Isolated Dyads (Size 2): {components.count(2)}")
    else:
        print("No components found.")

if __name__ == "__main__":
    analyze_connectivity("cilin", "checkpoints/semantic_edges_wiki.pkl")

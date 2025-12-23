"""
Audit Semantic Hubs.
====================

Loads the Wiki PMI edges and proper vocabulary to identify and analyze
the top semantic hubs (highest degree nodes).

Helps distinguish between "True Semantic Cores" (e.g., Happy, Animal)
and "Toxic Structural Hubs" (e.g., Is, The, Stuff).
"""

import pickle
import os
import sys
import collections
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.concept_mapper import ConceptMapper

def main():
    PMI_FILE = "checkpoints/semantic_edges_wiki.pkl"
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    
    if not os.path.exists(PMI_FILE) or not os.path.exists(CHECKPOINT):
        print("Error: Files not found.")
        return

    # 1. Load Edges
    print(f"Loading Semantic Edges from {PMI_FILE}...")
    with open(PMI_FILE, 'rb') as f:
        edges = pickle.load(f)
    print(f"Loaded {len(edges)} edges.")
    
    # 2. Calculate Degrees
    degree = collections.defaultdict(int)
    for u, v, w, t in edges:
        degree[u] += 1
        degree[v] += 1
        
    # 3. Load Vocab
    print(f"Loading Vocab from {CHECKPOINT}...")
    mapper = ConceptMapper(checkpoint_path=CHECKPOINT)
    
    # 4. Top Hubs Analysis
    sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== TOP 50 SEMANTIC HUBS ===")
    print(f"{'Rank':<5} {'ID':<10} {'Word':<20} {'Degree':<10}")
    print("-" * 50)
    
    for i, (node_id, deg) in enumerate(sorted_nodes[:50]):
        word = mapper.id_to_word.get(node_id, f"<UNK:{node_id}>")
        print(f"{i+1:<5} {node_id:<10} {word:<20} {deg:<10}")
        
    # Stats
    degrees = list(degree.values())
    p50 = np.percentile(degrees, 50)
    p90 = np.percentile(degrees, 90)
    p99 = np.percentile(degrees, 99)
    max_d = np.max(degrees)
    
    print("\n=== DEGREE STATISTICS ===")
    print(f"P50: {p50}")
    print(f"P90: {p90}")
    print(f"P99: {p99}")
    print(f"Max: {max_d}")

if __name__ == "__main__":
    main()

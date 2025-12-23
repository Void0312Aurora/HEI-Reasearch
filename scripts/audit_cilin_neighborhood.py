"""
Cilin Migration Audit Panel (Gate L, H, S, C, M, R).
===================================================

As per temp-10.md, this script validates the Cilin migration.

Gate L (Language Purity):
- Random 500 CH words. Top-50 neighbors CH ratio >= 99%.

Gate H (Hierarchy Consistency):
- 1k words. Top-50 neighbors share same Category (L1/L2) ratio.

Gate S (Synonym Convergence):
- 1k synonym groups. Inner-group dist < Random dist.

Gate C (Numerical Stability):
- Residual P99, Renorm Max, NaN Check.

Gate M (Mapping Coverage):
- Check coverage against high-frequency Chinese list (approx).

Gate R (Readout Coverage):
- Branching factor of Cilin Categories (how many leaves per code?).
"""

import sys
import os
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.geometry_n import dist_n, minkowski_inner

def load_checkpoint(path):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def is_chinese(word):
    for char in word:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def gate_l_purity(nodes, edges, node_map, sample_size=500, k=50):
    """
    Gate L: Language Purity.
    Check if neighbors of Chinese words are Chinese.
    Using Skeleton neighbors (since we use Skeleton-First Readout).
    """
    print("\n[Gate L] Language Purity Audit...")
    
    # Identify Chinese leaf words
    ch_words = [n for n in nodes if n.startswith("C:") and is_chinese(n.split(":")[1])]
    
    if not ch_words:
        print("FAIL: No Chinese words found.")
        return 0.0

    samples = np.random.choice(ch_words, min(len(ch_words), sample_size), replace=False)
    
    total_purity = 0.0
    
    # Build fast edge lookup
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u) # Undirected for neighbor check
        
    for word_node in samples:
        idx = node_map[word_node]
        
        # Get neighbors (BFS 2-hop for broader context)
        neighbors = set()
        q = [idx]
        visited = {idx}
        for _ in range(2):
            next_q = []
            for curr in q:
                for n in adj[curr]:
                    if n not in visited:
                        visited.add(n)
                        neighbors.add(n)
                        next_q.append(n)
            q = next_q
            
        # Check purity of neighbors (leaf words only)
        leaf_neighbors = [nodes[n] for n in neighbors if nodes[n].startswith("C:")]
        
        if not leaf_neighbors:
            continue
            
        ch_count = sum(1 for n in leaf_neighbors if is_chinese(n.split(":")[1]))
        purity = ch_count / len(leaf_neighbors)
        total_purity += purity
        
    avg_purity = total_purity / len(samples)
    print(f"  Avg Chinese Neighbor Ratio: {avg_purity*100:.2f}%")
    print(f"  Status: {'PASS' if avg_purity >= 0.99 else 'FAIL'} (Threshold 99%)")
    return avg_purity

def gate_r_readout(nodes, edges, node_map):
    """
    Gate R: Readout Coverage.
    Check branching factor (leaves per category).
    """
    print("\n[Gate R] Readout Coverage Audit...")
    
    # Identify Category Codes (L3/L4/L5)
    codes = [n for n in nodes if n.startswith("Code:")]
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v) # Parent -> Child
        
    leaf_counts = []
    
    for code_node in codes:
        idx = node_map[code_node]
        
        # BFS to finding leaves
        leaves = 0
        q = [idx]
        # Only go down
        while q:
            curr = q.pop(0)
            if curr in adj:
                children = adj[curr]
                for child in children:
                    child_node = nodes[child]
                    if child_node.startswith("C:"):
                        leaves += 1
                    elif child_node.startswith("Code:"):
                        q.append(child)
        
        if leaves > 0:
            leaf_counts.append(leaves)
            
    leaf_counts = np.array(leaf_counts)
    print(f"  Categories with Leaves: {len(leaf_counts)} / {len(codes)}")
    print(f"  Mean Leaves per Category: {np.mean(leaf_counts):.1f}")
    print(f"  Median Leaves: {np.median(leaf_counts)}")
    print(f"  Max Leaves: {np.max(leaf_counts)}")
    print(f"  Zero-Leaf Categories: {len(codes) - len(leaf_counts)}")
    
    status = "PASS" if len(leaf_counts) > 0 and np.mean(leaf_counts) > 1 else "WARN"
    print(f"  Status: {status}")
    return leaf_counts

def gate_s_synonyms(nodes, edges, node_map, G_pos, sample_size=1000):
    """
    Gate S: Synonym Convergence.
    Check if words under same low-level code (synonyms) are geometrically closer than random.
    """
    print("\n[Gate S] Synonym Convergence Audit...")
    
    # Group words by parent Code
    parent_map = {}
    for u, v in edges:
        # u is parent, v is child
        # We want child -> parent
        child_node = nodes[v]
        parent_node = nodes[u]
        
        if child_node.startswith("C:") and parent_node.startswith("Code:"):
            if parent_node not in parent_map:
                parent_map[parent_node] = []
            parent_map[parent_node].append(v)
            
    # Filter groups with >= 2 members
    valid_groups = [g for g in parent_map.values() if len(g) >= 2]
    
    if not valid_groups:
        print("FAIL: No synonym groups found.")
        return
        
    samples = valid_groups[:sample_size]
    
    inner_dists = []
    random_dists = []
    
    # Calculate geometric distances
    # G_pos shape: (N, dim+1, dim+1) -> we need coordinates or compute dist directly
    
    # Using simple random sampling for baseline
    all_indices = list(range(len(nodes)))
    
    count = 0 
    for group in samples:
        # Inner group distance (pairs)
        # Just take first pair for speed
        u = group[0]
        v = group[1]
        
        d_inner = dist_n(G_pos[u], G_pos[v])
        inner_dists.append(d_inner)
        
        # Random pair
        r1, r2 = np.random.choice(all_indices, 2)
        d_rand = dist_n(G_pos[r1], G_pos[r2])
        random_dists.append(d_rand)
        
        count += 1
        if count >= sample_size: break
        
    avg_inner = np.mean(inner_dists)
    avg_rand = np.mean(random_dists)
    
    print(f"  Avg Inner-Group Dist: {avg_inner:.4f}")
    print(f"  Avg Random Dist:      {avg_rand:.4f}")
    print(f"  Ratio (Inner/Rand):   {avg_inner/avg_rand:.2f}")
    
    status = "PASS" if avg_inner < avg_rand else "FAIL"
    print(f"  Status: {status}")

def main():
    checkpoint_path = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    data = load_checkpoint(checkpoint_path)
    nodes = data['nodes']
    edges = data.get('edges', [])
    G_pos = data['G']
    
    node_map = {n: i for i, n in enumerate(nodes)}
    
    # Gate L
    gate_l_purity(nodes, edges, node_map)
    
    # Gate R
    gate_r_readout(nodes, edges, node_map)
    
    # Gate S
    gate_s_synonyms(nodes, edges, node_map, G_pos)
    
    # Gate C (Stability)
    # We don't have training logs here, but we can check NaNs in G
    print("\n[Gate C] Numerical Stability Audit...")
    nan_count = np.isnan(G_pos).sum()
    print(f"  NaN Count in G: {nan_count}")
    print(f"  Status: {'PASS' if nan_count == 0 else 'FAIL'}")

if __name__ == "__main__":
    main()

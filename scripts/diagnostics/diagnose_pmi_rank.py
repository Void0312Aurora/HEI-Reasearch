"""
PMI Rank & Volume Diagnostic.
=============================

Per temp-25.md:
1. Tracks Volume State: KNN Top-1 Distance P50 (Target > 0.15).
2. Tracks PMI Penetration:
   - Rank Distribution (P50, P90)
   - Hit Rates (K=50, 100, 500, 2000)
   - Distance Ratio: d(PMI_Top1) / d(KNN_Top50)

This reveals if PMI is improving relative to local geometry, even if A(q)=0.
"""

import pickle
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.concept_mapper import ConceptMapper

def hyperbolic_distance_np(v1, v2):
    """Time-First Hyperboloid: d = arccosh(-<v1, v2>_L)"""
    inner = -v1[0] * v2[0] + np.sum(v1[1:] * v2[1:])
    return np.arccosh(max(-inner, 1.0 + 1e-15))

def main():
    PMI_FILE = "checkpoints/semantic_edges_wiki.pkl"
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    
    print(f"Loading PMI Edges from {PMI_FILE}...")
    with open(PMI_FILE, 'rb') as f:
        edges_raw = pickle.load(f)
    print(f"Loaded {len(edges_raw)} edges.")
    
    # Build Adjacency for Word nodes
    # Only keep type=2 (PMI)
    # We need a quick lookup: u -> list of v
    pmi_adj = {}
    for u, v, w, t in edges_raw:
        if t == 2:
            if u not in pmi_adj: pmi_adj[u] = []
            if v not in pmi_adj: pmi_adj[v] = []
            pmi_adj[u].append((v, w))
            pmi_adj[v].append((u, w))
            
    print(f"Loading Checkpoint {CHECKPOINT}...")
    with open(CHECKPOINT, 'rb') as f:
        data = pickle.load(f)
        
    embedding = data['x']
    node_list = data['nodes']
    
    # Filter Word Indices (Exclude Code nodes)
    word_indices = [i for i, n in enumerate(node_list) if not n.startswith("Code:")]
    word_indices_set = set(word_indices)
    print(f"Word Nodes: {len(word_indices)}")
    
    # Sample Probes (ensure they have PMI neighbors)
    # Pick random 100 word nodes that have PMI edges
    candidates = [i for i in word_indices if i in pmi_adj]
    if not candidates:
        print("No word nodes have PMI edges?")
        return
        
    np.random.seed(42)
    probes = np.random.choice(candidates, size=min(100, len(candidates)), replace=False)
    
    ranks = []
    ratios = []
    knn1_dists = []
    
    print(f"\nRunning Diagnostic on {len(probes)} probes...")
    
    for u_idx in tqdm(probes):
        u_vec = embedding[u_idx]
        
        # Get PMI neighbors
        pmi_nbrs = pmi_adj.get(u_idx, [])
        if not pmi_nbrs:
            continue
            
        # Get strongest PMI neighbor
        pmi_nbrs.sort(key=lambda x: x[1], reverse=True)
        top_pmi_idx, top_pmi_w = pmi_nbrs[0]
        
        # Calculate d_PMI
        if top_pmi_idx >= len(embedding): continue
        d_pmi = hyperbolic_distance_np(u_vec, embedding[top_pmi_idx])
        
        # Calculate distances to ALL Word nodes (brute force for accuracy on 100 probes)
        # In production this might need Faiss/HNSW, but 100 * 140k is doable ~14M ops
        
        # Vectorized distance calculation
        # X: (N_words, D), U: (D,)
        # <U, X> = -u0*x0 + u1..*x1..
        
        # Extract word embeddings matrix
        # Doing this per probe is slow, let's optimize outside loop if needed, 
        # but for 100 probes it's okay to just loop.
        
        # Optimization: matrix mult
        # We need a subset of embedding corresponding to word_indices
        pass 
        
    # Optimized Calculation
    word_emb = embedding[word_indices]
    
    for step_i, u_idx in enumerate(tqdm(probes)):
        u_vec = embedding[u_idx]
        
        # Get strongest PMI
        pmi_nbrs = pmi_adj.get(u_idx, [])
        pmi_nbrs.sort(key=lambda x: x[1], reverse=True)
        top_pmi_idx, _ = pmi_nbrs[0]
        
        # d_pmi
        v_vec = embedding[top_pmi_idx]
        d_pmi = hyperbolic_distance_np(u_vec, v_vec)
        
        # All distances
        # inner = -u0*x0 + us*xs
        # u0 shape (1,), x0 shape (N,)
        inner = -u_vec[0] * word_emb[:, 0] + np.dot(word_emb[:, 1:], u_vec[1:])
        clipped = np.maximum(-inner, 1.0 + 1e-15)
        dists = np.arccosh(clipped)
        
        # Remove self (d=0)
        # Find index of u_idx in word_indices
        # It's messy to map back. Just set 0 distance to infinity?
        # Or sorting handles it (it will be index 0).
        
        sorted_dists = np.sort(dists)
        
        # KNN Top-1 (exclude self at 0.0)
        # sorted_dists[0] is self ~ 0.0
        d_knn1 = sorted_dists[1]
        d_knn50 = sorted_dists[50]
        
        knn1_dists.append(d_knn1)
        ratios.append(d_pmi / d_knn50)
        
        # Find Rank of PMI neighbor
        # We computed known d_pmi. Count how many are smaller.
        # This is rank.
        rank = np.sum(dists < d_pmi) # includes self, so actual rank is sum - 1 + 1 (1-based) = sum
        # if dists < d_pmi checks <, so if equal it doesn't count.
        # If strict inequality:
        count_smaller = np.sum(dists < d_pmi)
        # If self is smaller (0 < d_pmi), it counts. 
        # So rank (1-based) among neighbors is count_smaller.
        ranks.append(count_smaller)
        
    ranks = np.array(ranks)
    ratios = np.array(ratios)
    knn1_dists = np.array(knn1_dists)
    
    print("\n=== VOLUME & RANK DIAGNOSTIC ===")
    print(f"Probes: {len(ranks)}")
    
    print("\n[Volume / Crowding Metric]")
    print(f"KNN Top-1 Dist P50: {np.percentile(knn1_dists, 50):.4f} (Target > 0.15)")
    print(f"KNN Top-1 Dist P90: {np.percentile(knn1_dists, 90):.4f}")
    
    print("\n[PMI Penetration]")
    print(f"PMI Top-1 Rank P50: {np.percentile(ranks, 50):.1f}")
    print(f"PMI Top-1 Rank P90: {np.percentile(ranks, 90):.1f}")
    print(f"Ratio d(PMI)/d(KNN50) P50: {np.percentile(ratios, 50):.2f}")
    
    print("\n[Hit Rates]")
    for k in [50, 100, 500, 2000, 10000]:
        hit_rate = np.mean(ranks <= k) * 100
        print(f"Hit@{k}: {hit_rate:.1f}%")
        
    print("================================")

if __name__ == "__main__":
    main()

"""
Aurora Evaluation Script.
=========================

Performs academic evaluation of the trained Hyperbolic embedding.

Metrics:
1. **Ultrametricity Score**: Measures how well the embedding preserves hierarchical tree structure. 
   Lower is better (0 = Perfect Tree).
2. **Semantic Alignment (MAP/Rank)**: Measures if high-PMI words are geometrically closer.
   Checks if true semantic neighbors are ranked higher than random nodes.

Visualization:
- Projects $H^n$ to Poincaré Disk (2D/3D).
- Visualizes semantic clusters (e.g. root categories).
"""

import sys
import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.geometry import dist_hyperbolic
# We need to manually implement numpy versions of geometry for eval to avoid torch overhead/conflict
# or just wrap torch.

def minkowski_inner_np(u, v):
    # u, v: (Dim,) or (N, Dim)
    # <u, v> = -u0v0 + u1v1 + ...
    if u.ndim == 1 and v.ndim == 1:
         return -u[0]*v[0] + np.sum(u[1:]*v[1:])
    elif u.ndim == 2 and v.ndim == 1:
         # Broadcast v to u
         return -u[:, 0]*v[0] + np.sum(u[:, 1:]*v[1:], axis=1)
    elif u.ndim == 1 and v.ndim == 2:
         return -v[:, 0]*u[0] + np.sum(v[:, 1:]*u[1:], axis=1)
    else:
         return -u[:, 0]*v[:, 0] + np.sum(u[:, 1:]*v[:, 1:], axis=1)

def dist_hyperbolic_np(u, v):
    inner = minkowski_inner_np(u, v)
    # Clamp for numerical safety (-1.0 max)
    inner = np.minimum(inner, -1.0 - 1e-7)
    return np.arccosh(-inner)

def project_to_poincare_ball(h):
    """
    Map Hyperboloid (t, x, y, z...) -> Poincare Ball (u, v, w...)
    Formula: v = x / (1 + t)
    """
    # h: (N, dim)
    t = h[:, 0:1]
    x_space = h[:, 1:]
    return x_space / (1.0 + t)

def eval_ultrametricity(h_data, sample_size=10000):
    """
    Score = Mean (|d_AC - d_BC|) / d_AB for triangles where AB is smallest side?
    Standard def: For any x, y, z, d(x,z) <= max(d(x,y), d(y,z)).
    Strong Triangle Inequality.
    Gromov's delta is better for general hyperbolicity.
    
    Here we use the relative delta score from standard literature:
    Sample triples. Check deviations from isosceles with long legs.
    """
    N = h_data.shape[0]
    indices = np.random.choice(N, size=(sample_size, 3), replace=True)
    
    scores = []
    
    for i, j, k in indices:
        if i == j or j == k or i == k: continue
        
        u, v, w = h_data[i], h_data[j], h_data[k]
        
        d_ij = dist_hyperbolic_np(u, v)
        d_jk = dist_hyperbolic_np(v, w)
        d_ik = dist_hyperbolic_np(u, w)
        
        # Sort edges: a <= b <= c
        edges = sorted([d_ij, d_jk, d_ik])
        s, m, l = edges
        
        # Ultrametric condition: l <= m (should be equal)
        # Violation: l - m
        # Score = (l - m) / l
        if l > 1e-9:
            score = (l - m) / l
            scores.append(score)
            
    return np.mean(scores) if scores else 0.0

def eval_semantic_rank(h_data, semantic_path, vocab, sample_limit=1000):
    """
    For a subset of semantic edges (u, v), calculate the rank of v relative to u 
    among all nodes (or a negative sample set).
    """
    if not os.path.exists(semantic_path):
        return None
        
    with open(semantic_path, 'rb') as f:
        edges = pickle.load(f) # List[(u_str, v_str, w, ...)]
        
    # Filter edges to those in vocab
    # [FIX] Handle mismatch between Raw Words (edges) and Node IDs (vocab)
    # Reconstruct raw map from vocab keys
    raw_to_id = {}
    for node_str, idx in vocab.items():
        # Node format: C:word:id
        if node_str.startswith("C:"):
            parts = node_str.split(":")
            if len(parts) >= 2:
                raw_word = parts[1]
                if raw_word not in raw_to_id:
                    raw_to_id[raw_word] = idx
                    
    valid_pairs = []
    match_count = 0
    total_edges = 0
    
    for item in edges:
        try:
            u_s, v_s = item[0], item[1]
            total_edges += 1
            
            # Try exact match first
            u_id = vocab.get(u_s)
            v_id = vocab.get(v_s)
            
            # Try raw match
            if u_id is None: u_id = raw_to_id.get(u_s)
            if v_id is None: v_id = raw_to_id.get(v_s)
            
            if u_id is not None and v_id is not None:
                valid_pairs.append((u_id, v_id))
                match_count += 1
        except:
            continue
            
    print(f"Matched {match_count}/{total_edges} edges using fuzzy lookup.")
            
    if not valid_pairs:
        print("No valid semantic pairs found for evaluation.")
        return None
        
    print(f"Evaluating alignment on substring of {len(valid_pairs)} semantic edges (sampling {sample_limit})...")
    
    # Subsample
    if len(valid_pairs) > sample_limit:
        import random
        random.shuffle(valid_pairs)
        valid_pairs = valid_pairs[:sample_limit]
        
    # Collect active node set for Active-Only Negative Sampling
    active_nodes = set()
    for u, v in valid_pairs:
        active_nodes.add(u)
        active_nodes.add(v)
    active_node_list = list(active_nodes)
    print(f"Active Nodes for Evaluation: {len(active_node_list)}")
    
    ranks_active = []
    ranks_global = []
    
    # Global distance calculation is expensive (N*N).
    # Use negative sampling: compare v against 100 random negatives.
    # Score = Rank among 101 candidates.
    
    N = h_data.shape[0]
    
    for u_idx, v_idx in tqdm(valid_pairs, desc="Semantic Rank"):
        u_vec = h_data[u_idx]
        v_vec = h_data[v_idx]
        d_pos = dist_hyperbolic_np(u_vec, v_vec)
        
        # 1. Active Subgraph Rank (Harder, Main Metric)
        neg_indices_act = np.random.choice(active_node_list, size=100)
        neg_vecs_act = h_data[neg_indices_act]
        
        inner_act = minkowski_inner_np(neg_vecs_act, u_vec)
        inner_act = np.minimum(inner_act, -1.0 - 1e-7)
        d_negs_act = np.arccosh(-inner_act)
        
        rank_act = np.sum(d_negs_act < d_pos) + 1
        ranks_active.append(rank_act)
        
        # 2. Global Rank (Reference)
        neg_indices_glob = np.random.randint(0, N, size=100)
        neg_vecs_glob = h_data[neg_indices_glob]
        
        inner_glob = minkowski_inner_np(neg_vecs_glob, u_vec)
        inner_glob = np.minimum(inner_glob, -1.0 - 1e-7)
        d_negs_glob = np.arccosh(-inner_glob)
        
        rank_glob = np.sum(d_negs_glob < d_pos) + 1
        ranks_global.append(rank_glob)
        
    mean_rank_active = np.mean(ranks_active)
    mean_rank_global = np.mean(ranks_global)
    
    print(f">>> Active Subgraph Mean Rank: {mean_rank_active:.2f} (Primary)")
    print(f">>> Global Mean Rank: {mean_rank_global:.2f} (Reference)")
    
    return mean_rank_active

def clean_and_map_labels(vocab, max_labels=50):
    # Reverse vocab
    id_to_word = {v: k for k, v in vocab.items()}
    # Filter for visualizing only interesting nodes (e.g. short words, meaningful ones)
    return id_to_word

def visualize_embedding(h_data, vocab, save_path="aurora_viz.png"):
    """
    Project to Poincare Disk (2D) and plot.
    If dim > 3 (after projection to 2D disk), use PCA to reduce to 2 components.
    """
    print("Projecting to Poincaré Ball...")
    ball_data = project_to_poincare_ball(h_data) # (N, dim-1)
    
    # PCA to 2D if needed
    dim_ball = ball_data.shape[1]
    if dim_ball > 2:
        print(f"Reducing {dim_ball}D -> 2D via PCA...")
        # Center data? Poincare ball origin is meaningful (root).
        # Standard PCA might distort hyperbolic geometry.
        # But for visualization 'academic behavior' allows PCA on the tangent or ball coords for overview.
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        viz_data = pca.fit_transform(ball_data)
        var_ratio = pca.explained_variance_ratio_
        print(f"PCA Variance Explained: {var_ratio}")
    else:
        viz_data = ball_data
        
    plt.figure(figsize=(12, 12))
    plt.scatter(viz_data[:, 0], viz_data[:, 1], s=1, alpha=0.3, c='blue', edgecolors='none')
    
    # Annotate some hubs
    # How to find hubs? Low radius (high hierarchy).
    radii = h_data[:, 0] # cosh(r)
    # smallest radii indices
    hub_indices = np.argsort(radii)[:30]
    
    id_to_word = {v: k for k, v in vocab.items()}
    
    texts = []
    # Annotate some hubs
    # How to find hubs? Low radius (high hierarchy).
    radii = h_data[:, 0] # cosh(r)
    # smallest radii indices
    hub_indices = np.argsort(radii)[:30]
    
    # Highlight hubs in red without text labels (cleaner academic view)
    plt.scatter(viz_data[hub_indices, 0], viz_data[hub_indices, 1], c='red', s=50, label='Hubs (Roots)')
    
    plt.legend(loc='upper left')
        
    plt.title("Aurora V2 Semantic Embedding (Poincaré Projection)")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--semantic_path", type=str, default=None)
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}...")
    with open(args.checkpoint, 'rb') as f:
        data = pickle.load(f)
        
    h_data = data['x'] # (N, dim)
    vocab = data['vocab']
    nodes = data['nodes']
    
    print(f"Embedding Shape: {h_data.shape}")
    
    # 1. Radius Statistics (Crucial for Volume Control Audit)
    print("Calculating Radius Statistics...")
    # h_data is (N, dim). x0 = h_data[:, 0]
    x0 = h_data[:, 0]
    # Safety clamp
    x0 = np.maximum(x0, 1.0 + 1e-7)
    radii = np.arccosh(x0)
    
    r_mean = np.mean(radii)
    r_median = np.median(radii)
    r_p95 = np.percentile(radii, 95)
    r_max = np.max(radii)
    
    print(f">>> Radius Stats: Mean={r_mean:.2f}, Median={r_median:.2f}, P95={r_p95:.2f}, Max={r_max:.2f}")
    if r_mean > 20.0:
        print("WARNING: High mean radius detected. Check Volume Control or Repulsion Force.")

    # 2. Ultrametricity
    print("Calculating Ultrametricity Score...")
    print("Def: Mean((longest - middle) / longest) for random triangles.")
    um_score = eval_ultrametricity(h_data)
    print(f">>> Ultrametricity Score: {um_score:.4f} (Lower=Better, 0=Tree)")
    
    # 3. Semantic Rank
    if args.semantic_path:
        rank = eval_semantic_rank(h_data, args.semantic_path, vocab)
        if rank:
            print(f">>> Mean Rank (vs 100 Negs): {rank:.2f} (1.0 = Perfect)")
            
    # 3. Visual
    visualize_embedding(h_data, vocab, save_path="checkpoints/aurora_viz.png")

if __name__ == "__main__":
    main()

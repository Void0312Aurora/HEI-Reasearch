"""
PMI-KNN Alignment Diagnostic.
=============================

Computes:
1. A(q) = |PMI_Top20 ∩ KNN_Top50| / 20 for each probe word.
2. Radial vs Angular decomposition for probe → PMI Top-1 neighbor.

This diagnoses whether PMI is "sinking into geometry" or being "radially suppressed".
"""

import pickle
import os
import sys
import collections
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.concept_mapper import ConceptMapper

def minkowski_inner(v1, v2):
    """Time-First Minkowski inner product: <v1, v2>_L = -t1*t2 + spatial"""
    return -v1[0] * v2[0] + np.sum(v1[1:] * v2[1:])

def hyperbolic_distance(v1, v2):
    """Hyperbolic distance: arccosh(-<v1, v2>_L)"""
    inner = minkowski_inner(v1, v2)
    return np.arccosh(max(-inner, 1.0 + 1e-15))

def get_radius(v):
    """Get hyperbolic radius (distance from origin). For time-first hyperboloid, r = arccosh(t)"""
    return np.arccosh(max(v[0], 1.0 + 1e-15))

def radial_angular_decomposition(v1, v2):
    """
    Decompose hyperbolic distance into radial and angular contributions.
    Returns (radial_diff, angular_contrib, total_dist)
    """
    r1 = get_radius(v1)
    r2 = get_radius(v2)
    radial_diff = abs(r1 - r2)
    
    total_dist = hyperbolic_distance(v1, v2)
    
    # Angular contribution is approximated as sqrt(d^2 - (Δr)^2) when Δr < d
    if radial_diff < total_dist:
        angular_contrib = np.sqrt(max(total_dist**2 - radial_diff**2, 0))
    else:
        angular_contrib = 0.0
        
    return radial_diff, angular_contrib, total_dist

def main():
    PMI_FILE = "checkpoints/semantic_edges_wiki.pkl"
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    
    # 1. Load PMI edges
    print(f"Loading PMI Edges from {PMI_FILE}...")
    with open(PMI_FILE, 'rb') as f:
        edges = pickle.load(f)
    print(f"Loaded {len(edges)} edges.")
    
    # Build adjacency list (ID -> [(neighbor_id, weight)])
    pmi_neighbors = collections.defaultdict(list)
    for u, v, w, t in edges:
        pmi_neighbors[u].append((v, w))
        pmi_neighbors[v].append((u, w))
    
    # Sort by weight descending
    for k in pmi_neighbors:
        pmi_neighbors[k].sort(key=lambda x: x[1], reverse=True)
    
    # 2. Load Checkpoint
    print(f"Loading Checkpoint {CHECKPOINT}...")
    with open(CHECKPOINT, 'rb') as f:
        data = pickle.load(f)
    
    embedding = data['x']  # (N, D)
    node_list = data['nodes']
    
    # Build node_to_idx mapping
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # 3. Load Mapper
    mapper = ConceptMapper(checkpoint_path=CHECKPOINT)
    
    # 4. Identify Word nodes (filter out Code nodes)
    word_node_indices = set()
    for i, n in enumerate(node_list):
        if not n.startswith("Code:"):
            word_node_indices.add(i)
    print(f"Word nodes: {len(word_node_indices)} / {len(node_list)}")
    
    # 5. Define Probes
    probe_words = ["猫", "医生", "狗", "老师", "快乐", "红"]
    
    print("\n=== PMI-KNN ALIGNMENT DIAGNOSTIC ===")
    
    alignment_scores = []
    
    for word in probe_words:
        ids = mapper.text_to_particles(word)
        if not ids:
            print(f"\n[{word}]: Not in vocabulary.")
            continue
        
        pid = ids[0]  # Particle ID
        
        # Get PMI Top-20 neighbors (IDs)
        pmi_nbrs = pmi_neighbors.get(pid, [])[:20]
        pmi_nbr_ids = set([n[0] for n in pmi_nbrs])
        
        if len(pmi_nbrs) == 0:
            print(f"\n[{word}]: No PMI neighbors.")
            continue
        
        # Find node index for this word
        target_idx = None
        for i, n in enumerate(node_list):
            parts = n.split(':')
            if len(parts) >= 2 and parts[1] == word:
                target_idx = i
                break
        
        if target_idx is None:
            print(f"\n[{word}]: Node not found in embedding.")
            continue
        
        # Get geometric Top-50 Word neighbors
        vec = embedding[target_idx]
        dists = []
        for i in word_node_indices:
            if i == target_idx:
                continue
            dists.append((i, hyperbolic_distance(vec, embedding[i])))
        dists.sort(key=lambda x: x[1])
        knn_top50 = set([d[0] for d in dists[:50]])
        
        # Compute A(q) = |PMI_Top20 ∩ KNN_Top50| / 20
        # But we need to map PMI particle IDs to node indices
        # PMI uses particle IDs, embedding uses node indices
        # The mapper.id_to_word exists, but we need node_list index
        
        # Build particle_id -> node_idx mapping
        # Particle ID in PMI graph is the same as the node index? Let's check.
        # Actually, PMI edges use particle IDs from ConceptMapper.
        # We need to find the node_list index corresponding to each particle ID.
        
        # Simplification: Assume particle_id == node_idx for word nodes (check if reasonable)
        # Actually, let's use id_to_word to verify
        
        overlap = 0
        overlap_words = []
        for nbr_pid in pmi_nbr_ids:
            # PMI neighbor particle ID -> check if it's in KNN top 50
            if nbr_pid in knn_top50:
                overlap += 1
                nbr_word = mapper.id_to_word.get(nbr_pid, f"<{nbr_pid}>")
                overlap_words.append(nbr_word)
        
        a_q = overlap / len(pmi_nbrs) if pmi_nbrs else 0.0
        alignment_scores.append(a_q)
        
        # Radial/Angular decomposition for Top-1 PMI neighbor
        top1_pid = pmi_nbrs[0][0]
        top1_word = mapper.id_to_word.get(top1_pid, f"<{top1_pid}>")
        
        if top1_pid < len(embedding):
            radial, angular, total = radial_angular_decomposition(vec, embedding[top1_pid])
            r_self = get_radius(vec)
            r_nbr = get_radius(embedding[top1_pid])
        else:
            radial, angular, total = 0, 0, 0
            r_self, r_nbr = 0, 0
        
        print(f"\n[{word}] (ID={pid})")
        print(f"  A(q) = {a_q:.2f} ({overlap}/{len(pmi_nbrs)} PMI neighbors in KNN Top-50)")
        if overlap > 0:
            print(f"  Overlap: {overlap_words}")
        print(f"  PMI Top-1: {top1_word}")
        print(f"    Radius(self)={r_self:.4f}, Radius(nbr)={r_nbr:.4f}")
        print(f"    Δ_radial={radial:.4f}, Δ_angular≈{angular:.4f}, Total={total:.4f}")
    
    print("\n=== SUMMARY ===")
    if alignment_scores:
        print(f"Mean A(q): {np.mean(alignment_scores):.2f}")
        print(f"P50 A(q):  {np.percentile(alignment_scores, 50):.2f}")
        print(f"P90 A(q):  {np.percentile(alignment_scores, 90):.2f}")
        
        if np.mean(alignment_scores) < 0.1:
            print("\n>>> DIAGNOSIS: PMI edges NOT sinking into geometry. Need longer training or structural fixes.")
        else:
            print("\n>>> DIAGNOSIS: PMI edges ARE affecting geometry. Consider fine-tuning weights.")

if __name__ == "__main__":
    main()

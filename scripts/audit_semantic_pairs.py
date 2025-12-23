"""
Audit Semantic Pairs (Phase II Verification).
=============================================

Checks distances of semantic pairs (PMI/Def) in the trained embedding.
"""

import pickle
import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.geometry_n import dist_n

def main():
    checkpoint_path = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    edges_path = "checkpoints/semantic_edges.pkl"
    
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        return
        
    print(f"Loading checkpoint {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
        
    G = data['G'] # (N, dim+1, dim+1) or (N, dim+1, dim+1) ?
    # Actually training script saves G as (N, dim+1, dim+1).
    # We need positions x = G[:, :, 0].
    
    # Check shape
    if len(G.shape) == 3:
        positions = G[:, :, 0]
    else:
        positions = G
        
    print(f"Positions shape: {positions.shape}")
    
    # Load semantic edges
    print(f"Loading Semantic Edges from {edges_path}...")
    with open(edges_path, 'rb') as f:
        semantic_edges = pickle.load(f)
        
    # Calculate Distances
    dists = []
    
    print("\n--- Semantic Pair Distances ---")
    print(f"{'Pair':<30} | {'Type':<5} | {'Distance':<10}")
    print("-" * 55)
    
    # Load mapper for text
    from hei_n.concept_mapper import ConceptMapper
    mapper = ConceptMapper(checkpoint_path=checkpoint_path)
    
    pmi_dists = []
    def_dists = []
    
    for u, v, w, type_id in semantic_edges:
        pos_u = positions[u]
        pos_v = positions[v]
        
        # Calculate distance
        # dist_n expects np arrays
        d = dist_n(pos_u, pos_v)
        
        u_text = mapper.particles_to_text([u])[0]
        v_text = mapper.particles_to_text([v])[0]
        
        pair_str = f"{u_text}-{v_text}"
        type_str = "PMI" if type_id == 2 else "Def"
        
        print(f"{pair_str:<30} | {type_str:<5} | {d:.4f}")
        
        if type_id == 2:
            pmi_dists.append(d)
        else:
            def_dists.append(d)
            
    print("\n--- Summary ---")
    print(f"PMI Mean Dist: {np.mean(pmi_dists):.4f} (Count: {len(pmi_dists)})")
    if def_dists:
        print(f"Def Mean Dist: {np.mean(def_dists):.4f} (Count: {len(def_dists)})")
    
    # Random Baseline comparison
    # Sample 100 random pairs
    N = positions.shape[0]
    rand_dists = []
    for _ in range(100):
        idx = np.random.choice(N, 2, replace=False)
        d = dist_n(positions[idx[0]], positions[idx[1]])
        rand_dists.append(d)
        
    print(f"Random Baseline Mean Dist: {np.mean(rand_dists):.4f}")
    
    # Success Criteria: Semantic < Random * 0.5?
    if np.mean(pmi_dists) < np.mean(rand_dists) * 0.8:
        print("RESULT: PASS (Semantic pairs are closer than random)")
    else:
        print("RESULT: FAIL/WEAK (Semantic pairs not significantly closer)")

if __name__ == "__main__":
    main()

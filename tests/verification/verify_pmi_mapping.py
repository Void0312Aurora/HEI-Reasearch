"""
PMI Edge Closed-Loop Verification.
===================================

Tests whether PMI edge particle IDs correctly map to node indices in the embedding.
Per temp-23.md recommendations:
1. Select specific PMI edge (猫 → 波斯猫)
2. Verify particle_id matches node_list index
3. Compute d(u, v) directly from embedding
4. Check rank(v | u) in KNN

If rank is high but d(u,v) is small → KNN candidate/distance mismatch.
If d(u,v) is large → ID mapping error.
"""

import pickle
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.concept_mapper import ConceptMapper

def hyperbolic_distance(v1, v2):
    """Time-First Hyperboloid: d = arccosh(-<v1, v2>_L)"""
    inner = -v1[0] * v2[0] + np.sum(v1[1:] * v2[1:])
    return np.arccosh(max(-inner, 1.0 + 1e-15))

def main():
    PMI_FILE = "checkpoints/semantic_edges_wiki.pkl"
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    
    # 1. Load PMI edges
    print(f"Loading PMI Edges from {PMI_FILE}...")
    with open(PMI_FILE, 'rb') as f:
        edges = pickle.load(f)
    print(f"Loaded {len(edges)} edges.")
    
    # 2. Load Checkpoint
    print(f"Loading Checkpoint {CHECKPOINT}...")
    with open(CHECKPOINT, 'rb') as f:
        data = pickle.load(f)
    
    embedding = data['x']  # (N, D)
    node_list = data['nodes']
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # 3. Load Mapper (for word → particle_id)
    mapper = ConceptMapper(checkpoint_path=CHECKPOINT)
    
    # 4. Test specific PMI edges
    test_pairs = [
        ("猫", "波斯猫"),
        ("医生", "护士"),
        ("狗", "看门"),
        ("老师", "班主任"),
        ("红", "绿"),
    ]
    
    print("\n=== PMI EDGE CLOSED-LOOP VERIFICATION ===")
    
    for word_u, word_v in test_pairs:
        print(f"\n--- Testing: [{word_u}] → [{word_v}] ---")
        
        # Get particle IDs from mapper
        ids_u = mapper.text_to_particles(word_u)
        ids_v = mapper.text_to_particles(word_v)
        
        if not ids_u:
            print(f"  [{word_u}]: Not in mapper vocabulary.")
            continue
        if not ids_v:
            print(f"  [{word_v}]: Not in mapper vocabulary.")
            continue
            
        pid_u = ids_u[0]
        pid_v = ids_v[0]
        print(f"  Particle IDs: {word_u}={pid_u}, {word_v}={pid_v}")
        
        # Check if PMI edge exists
        edge_found = False
        edge_weight = 0.0
        for u, v, w, t in edges:
            if t != 2:  # Only PMI edges
                continue
            if (u == pid_u and v == pid_v) or (u == pid_v and v == pid_u):
                edge_found = True
                edge_weight = w
                break
        
        if edge_found:
            print(f"  PMI Edge EXISTS: weight={edge_weight:.4f}")
        else:
            print(f"  PMI Edge NOT FOUND in graph!")
            continue
        
        # Check if particle IDs are valid indices
        if pid_u >= len(embedding) or pid_v >= len(embedding):
            print(f"  ERROR: Particle ID out of bounds! Max idx={len(embedding)-1}")
            continue
        
        # Compute distance directly from embedding
        vec_u = embedding[pid_u]
        vec_v = embedding[pid_v]
        d_uv = hyperbolic_distance(vec_u, vec_v)
        print(f"  Direct Distance d({word_u}, {word_v}) = {d_uv:.4f}")
        
        # Compute rank of v in KNN of u (Word nodes only)
        # Gather all Word node distances from u
        word_dists = []
        for i, node in enumerate(node_list):
            if node.startswith("Code:"):
                continue
            if i == pid_u:
                continue
            d = hyperbolic_distance(vec_u, embedding[i])
            word_dists.append((i, d))
        
        # Sort by distance
        word_dists.sort(key=lambda x: x[1])
        
        # Find rank of v
        rank_v = -1
        for rank, (idx, d) in enumerate(word_dists):
            if idx == pid_v:
                rank_v = rank + 1  # 1-indexed
                break
        
        if rank_v > 0:
            print(f"  Rank of [{word_v}] in KNN of [{word_u}]: #{rank_v}")
            print(f"  KNN Top-5 for comparison:")
            for k in range(min(5, len(word_dists))):
                idx, d = word_dists[k]
                parts = node_list[idx].split(':')
                name = parts[1] if len(parts) >= 2 else node_list[idx]
                print(f"    #{k+1}: {name} (d={d:.4f})")
        else:
            print(f"  [{word_v}] NOT FOUND in Word node list!")
            # Check if it's a Code node
            target_node = node_list[pid_v] if pid_v < len(node_list) else "OOB"
            print(f"  node_list[{pid_v}] = {target_node}")
    
    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()

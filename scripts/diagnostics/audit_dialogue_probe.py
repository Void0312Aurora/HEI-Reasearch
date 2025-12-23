"""
Audit Dialogue Probes.
======================

Qualitative evaluation of the HEI embedding by querying nearest neighbors
for a set of probe words relevant to dialogue (Emotion, Logic, Concrete).

Goal: Verify "Association" capability (e.g., Sad -> Cry) vs "Taxonomy" (Sad -> Emotion).
"""

import pickle
import os
import sys
import numpy as np
import collections

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.concept_mapper import ConceptMapper
from hei_n.concept_mapper import ConceptMapper

def compute_distances(vec, mat):
    """
    Compute hyperbolic distance between a vector (D,) and matrix (N, D).
    Uses TIME-FIRST convention: x[0] is time-like, x[1:] are spatial.
    d = arccosh(-<x, y>_L)
    <x, y>_L = -x[0]*y[0] + sum(x[1:]*y[1:])
    """
    # Minkowski dot product (Time-First): -t*t + spatial*spatial
    inner = -vec[0] * mat[:, 0] + np.sum(vec[1:] * mat[:, 1:], axis=-1)
    
    # Clip for numerical stability (-inner >= 1)
    val = np.maximum(-inner, 1.0 + 1e-15)
    return np.arccosh(val)

def main():
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    DEPTHS_FILE = "checkpoints/aurora_base_gpu_cilin_full_depths.npy"
    
    if not os.path.exists(CHECKPOINT):
        print("Error: Checkpoint not found.")
        return

    # 1. Load Model
    print(f"Loading Checkpoint {CHECKPOINT}...")
    with open(CHECKPOINT, 'rb') as f:
        data = pickle.load(f)
        
    # embedding = data['embedding'] # (N, D) <-- WRONG
    # Checkpoint uses 'x' for positions (Hyperboloid coordinates)
    embedding = data['x']
    if not isinstance(embedding, np.ndarray): embedding = np.array(embedding)
    node_list = data['nodes']
    word_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # 2. Load Cilin Mapper (for nice printing)
    mapper = ConceptMapper(checkpoint_path=CHECKPOINT)
    
    # 3. Define Probes
    probes = [
        "快乐", "悲伤", "愤怒", # Emotion
        "如果", "因为", "所以", # Logic
        "猫", "狗", "医生", "老师", # Concrete
        "红", "圆" # Abstract
    ]
    
    print("\n=== DIALOGUE PROBE AUDIT ===")
    
    for word in probes:
        # Find ID
        ids = mapper.text_to_particles(word)
        if not ids:
            print(f"Probe '{word}': Not found.")
            continue
            
        # Try finding the 'word' type node
        target_idx = -1
        found = False
        
        # Heuristic: Search in node_list
        # Try constructing exact node name if we knew the format 
        # But scanning is safer
        for node_str in node_list:
            # Format is usually 'C:word:idx' or similar
            parts = node_str.split(':')
            if len(parts) >= 2 and parts[1] == word:
                 target_idx = word_to_idx[node_str]
                 found = True
                 break
        
        if not found:
             print(f"Probe '{word}': Node not found in embedding.")
             continue
             
        # Calculate Distances
        vec = embedding[target_idx]
        dists = compute_distances(vec, embedding)
        
        # Build (index, distance) pairs, filtering out Code nodes
        word_dists = []
        for i, node in enumerate(node_list):
            if node.startswith("Code:"):
                continue  # Skip Code nodes
            if i == target_idx:
                continue  # Skip self
            word_dists.append((i, float(dists[i])))
        
        # Sort by distance and take top-15
        word_dists.sort(key=lambda x: x[1])
        top_k = word_dists[:15]
        
        print(f"\nProbe: [{word}]")
        for k_int, dist in top_k:
            node_name = node_list[k_int]
            # Extract clean word (Format: C:word:count or similar)
            parts = node_name.split(':')
            clean_word = parts[1] if len(parts) >= 2 else node_name
            
            print(f"  {dist:.4f} : {clean_word}")

if __name__ == "__main__":
    main()

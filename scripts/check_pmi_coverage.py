"""
Check PMI Coverage for Probe Words.
====================================

Directly inspect the PMI graph to see if 猫/医生/红 have strong PMI neighbors.
This diagnoses whether the issue is "PMI coverage" or "geometric suppression".
"""

import pickle
import os
import sys
import collections

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.concept_mapper import ConceptMapper

def main():
    PMI_FILE = "checkpoints/semantic_edges_wiki.pkl"
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    
    # 1. Load PMI edges
    print(f"Loading PMI Edges from {PMI_FILE}...")
    with open(PMI_FILE, 'rb') as f:
        edges = pickle.load(f)
    print(f"Loaded {len(edges)} edges.")
    
    # 2. Build adjacency list
    neighbors = collections.defaultdict(list)
    for u, v, w, t in edges:
        neighbors[u].append((v, w))
        neighbors[v].append((u, w))
        
    # 3. Load Mapper
    mapper = ConceptMapper(checkpoint_path=CHECKPOINT)
    
    # 4. Check specific probe words
    probe_words = ["猫", "医生", "红", "快乐", "狗", "老师"]
    
    print("\n=== PMI COVERAGE CHECK ===")
    for word in probe_words:
        ids = mapper.text_to_particles(word)
        if not ids:
            print(f"\n[{word}]: Not in vocabulary.")
            continue
            
        # Get first particle ID
        pid = ids[0]
        
        # Get PMI neighbors
        nbrs = neighbors.get(pid, [])
        nbrs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n[{word}] (ID={pid}): {len(nbrs)} PMI neighbors")
        
        # Print Top-20
        for nbr_id, weight in nbrs[:20]:
            nbr_word = mapper.id_to_word.get(nbr_id, f"<UNK:{nbr_id}>")
            print(f"  {weight:.4f} : {nbr_word}")
            
        if len(nbrs) == 0:
            print("  >>> NO PMI NEIGHBORS! This word is isolated in PMI graph.")

if __name__ == "__main__":
    main()

"""
Gate N Audit: Neighborhood Semantics.
=====================================

Evaluates the local neighborhood quality of the embedding.
Metric: Category Consistency (Purity of Neighborhood).

For a set of probes, retrieves Top-K neighbors and calculates:
1. L1 Match Rate: % neighbors sharing same Level 1 Code (e.g. 'A' - Human).
2. L2 Match Rate: % neighbors sharing same Level 2 Code (e.g. 'Aa' - People).

High Consistency = Preserved Structure.
Drastic Drop = Semantic Entropy / Collapse.

Target:
- L1 Match > 80% (Broad category stable)
- L2 Match > 50% (Fine category stable)
"""

import pickle
import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import collections

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.local_force_field import LocalForceField

def main():
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    PROBE_COUNT = 500
    TOP_K = 50
    
    if not os.path.exists(CHECKPOINT):
        print("Checkpoint not found.")
        return
        
    print(f"Loading {CHECKPOINT} for Gate N Audit...")
    with open(CHECKPOINT, 'rb') as f:
        data = pickle.load(f)
        
    G_tensor = torch.tensor(data['G'], device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
    positions = G_tensor[:, :, 0] # (N, dim)
    
    # Needs Cilin Code Mapping
    # Creating a map: Particle ID -> Cilin Code
    # The checkpoint keys 'nodes' matches the order.
    nodes = data['nodes'] # List of node strings "Code:Aa01" or "C:Word:0"
    
    # We actually need to know the Cilin Code for each WORD particle.
    # Words are children of codes.
    # Reconstructing simple map from the Node String if possible?
    # No, words are "C:Word:ID". They don't contain the code.
    # We need the graph edges? 'edges' is in checkpoint.
    edges = data['edges'] # (E, 2)
    
    # Build Parent Map
    print("Building Parent Map...")
    parent_map = {} # Child ID -> Parent ID
    adj = collections.defaultdict(list)
    for u, v in edges:
        # Edge direction? Checkpoint edges are usually Code -> Word?
        # In `train.py`: `G.add_edge(leaf_code_node, word_node)`.
        # So edges[0] is Parent, edges[1] is Child (Word).
        parent_map[v] = u
        
    # Build ID -> Code String map
    # A word might be child of "Code:Aa01".
    # So we map WordID -> ParentID -> ParentName("Code:Aa01") -> "Aa01"
    
    id_to_code = {}
    valid_probes = []
    
    for i, node_str in enumerate(nodes):
        if node_str.startswith("C:"):
            # It's a word/concept
            pid = i
            if pid in parent_map:
                parent_id = parent_map[pid]
                parent_str = nodes[parent_id]
                if parent_str.startswith("Code:"):
                    code = parent_str.split(":")[1]
                    id_to_code[pid] = code
                    valid_probes.append(pid)
                    
    print(f"Found {len(valid_probes)} valid word particles with Cilin Codes.")
    
    # Sample Probes
    np.random.seed(42)
    if len(valid_probes) > PROBE_COUNT:
        probes = np.random.choice(valid_probes, PROBE_COUNT, replace=False)
    else:
        probes = valid_probes
        
    print(f"Auditing {len(probes)} probes (Top-{TOP_K} neighbors)...")
    
    # Init ForceField (for KNN)
    lff = LocalForceField(positions.cpu().numpy(), device='cuda') # Moves back to GPU inside
    
    l1_matches = []
    l2_matches = []
    
    probe_tensor = positions[probes] # (P, dim)
    
    # Batch query? LFF supports batch if implemented.
    # get_neighbors takes (dim,).
    # Let's loop or vectorise. `get_neighbors` uses `index.search(query, k)`
    # query can be (B, dim).
    
    # LFF API: `query = cursor_pos.detach().cpu().numpy().reshape(1, -1)`
    # It strictly reshapes to 1,-1. 
    # Modify LFF? No, just loop or use faiss directly if needed.
    # Doing loop for 500 probes is fast.
    
    for i in tqdm(range(len(probes))):
        pid = probes[i]
        pos = positions[pid]
        
        # Get Code
        my_code = id_to_code[pid]
        
        # Get Neighbors
        n_ids, _ = lff.get_neighbors(pos, k=TOP_K+1) # +1 includes self
        
        # Calculate Purity
        hits_l1 = 0
        hits_l2 = 0
        valid_n = 0
        
        for nid in n_ids:
            if nid == pid: continue
            if nid not in id_to_code: continue # Skip non-word neighbors (codes themselves)
            
            n_code = id_to_code[nid]
            valid_n += 1
            
            # L1 Match (Char 0)
            if n_code[0] == my_code[0]:
                hits_l1 += 1
                
            # L2 Match (Char 0-1)
            if len(n_code) >= 2 and len(my_code) >= 2:
                if n_code[:2] == my_code[:2]:
                    hits_l2 += 1
                    
        if valid_n > 0:
            l1_matches.append(hits_l1 / valid_n)
            l2_matches.append(hits_l2 / valid_n)
            
    # Report
    mean_l1 = np.mean(l1_matches)
    mean_l2 = np.mean(l2_matches)
    
    print("\n=== GATE N REPORT ===")
    print(f"L1 Consistency (Broad): {mean_l1*100:.2f}% (Target: >80%)")
    print(f"L2 Consistency (Fine):  {mean_l2*100:.2f}% (Target: >50%)")
    
    status = "PASS" if mean_l1 > 0.80 else "FAIL"
    print(f"Status: {status}")
    print("=====================")

if __name__ == "__main__":
    main()

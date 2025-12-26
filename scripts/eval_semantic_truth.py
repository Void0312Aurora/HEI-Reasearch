"""
Gate B: Semantic Truth Validator (Active Inference Phase 11)

Purpose:
Validate statistically if mined edges connect words that share the same Semantic Category (Synonyms).
Uses the Structural Tree (Cilin) to determine ground truth categories.
"""

import sys
import os
import torch
import pickle
import argparse
import numpy as np

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset

def build_parent_map(dataset):
    """
    Build Child -> Parent map from structural edges.
    """
    parent_map = {}
    for u, v in dataset.edges_struct:
        # Edge direction in Cilin loader: Parent -> Child?
        # Usually tree edges are encoded as (u, v).
        # Need to verify direction. 
        # Cilin loader: `edge_arr` comes from `G.edges()`.
        # Usually Parent -> Child.
        # But let's assume if (u, v) is edge, and depths[u] < depths[v], then u is parent.
        if dataset.depths[u] < dataset.depths[v]:
            parent_map[v] = u
        else:
            parent_map[u] = v
    return parent_map

def get_synonym_group(node_idx, parent_map, nodes):
    """
    Find the potential 'Synonym Group' node (Code node).
    For Cilin: Word -> SynonymGroup(Code) -> ...
    """
    if node_idx not in parent_map:
        return None
    
    parent_idx = parent_map[node_idx]
    parent_str = nodes[parent_idx]
    
    # Check if parent is a Code node (Cilin Category)
    # Cilin format: Code:Aa01
    return parent_idx

def validate_edges(edges, dataset):
    """
    Compute Precision of Synonymy.
    edges: List of (u, v) or (u, v, w) tuples (indices).
    """
    # 1. Build Parent Map
    parent_map = build_parent_map(dataset)
    
    total = 0
    matches = 0
    missed_info = 0
    
    print(f"Validating {len(edges)} edges against Cilin Ontology...")
    
    for item in edges:
        u, v = item[0], item[1]
        
        # Determine Synonym Groups
        group_u = get_synonym_group(u, parent_map, dataset.nodes)
        group_v = get_synonym_group(v, parent_map, dataset.nodes)
        
        if group_u is None or group_v is None:
            missed_info += 1
            continue
            
        total += 1
        
        # Validation Logic:
        # Strict: Same Group ID
        if group_u == group_v:
            matches += 1
            
    if total == 0:
        return 0.0
        
    precision = matches / total
    print(f"Analyzed {total} pairs (skipped {missed_info} root/orphan nodes).")
    print(f"Matches (Shared Parent): {matches}")
    print(f"Precision: {precision:.4f}")
    
    return precision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cilin")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--edges_path", type=str, required=True, help="Path to mined edges pickle")
    args = parser.parse_args()
    
    # Load Dataset
    print(f"Loading {args.dataset} (limit={args.limit})...")
    ds = AuroraDataset(args.dataset, limit=args.limit)
    
    # Load Edges
    print(f"Loading edges from {args.edges_path}...")
    with open(args.edges_path, 'rb') as f:
        edges = pickle.load(f)
    
    # If edges are tensor, convert to list
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy().tolist()
        
    # If edges are strings (raw format), mapping is needed. 
    # But usually we valid mined indices.
    # If file contains strings:
    if len(edges) > 0 and isinstance(edges[0][0], str):
        print("Input edges are strings. Mapping to indices...")
        mapped = []
        for item in edges:
            u_s, v_s = item[0], item[1]
            u = ds.vocab[u_s]
            v = ds.vocab[v_s]
            if u is not None and v is not None:
                mapped.append((u, v))
        edges = mapped
        
    validate_edges(edges, ds)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Base Neighborhood Audit Script.
================================

Runs 3 audits based on temp-09.md recommendations:
1. Sememe Overlap: Check if geometric neighbors share sememes with anchor
2. Skeleton Edge P@K: Check if geometric neighbors are skeleton neighbors
3. Language Consistency: Check language mixing in neighborhoods
"""

import sys
import os
import pickle
import numpy as np
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.local_force_field import LocalForceField


def load_checkpoint(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def parse_node(node):
    """Parse node to extract info."""
    info = {'raw': node, 'type': 'unknown', 'en': None, 'zh': None}
    
    if node.startswith('C:'):
        # Concept: C:word:id
        parts = node.split(':')
        if len(parts) >= 2:
            info['type'] = 'concept'
            info['en'] = parts[1]
    elif '|' in node:
        # Sememe: English|Chinese
        parts = node.split('|')
        info['type'] = 'sememe'
        info['en'] = parts[0]
        info['zh'] = parts[1] if len(parts) > 1 else None
    else:
        info['type'] = 'other'
        info['en'] = node
        
    return info


def is_chinese(text):
    """Check if text contains Chinese characters."""
    if text is None:
        return False
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def extract_sememes_for_concept(node_idx, nodes, edges):
    """Extract sememes connected to a concept via edges."""
    sememes = set()
    # This is a placeholder - need actual edge data
    # For now, return empty set
    return sememes


def run_audit_1_sememe_overlap(nodes, positions, test_indices, k=50):
    """
    Audit 1: Sememe Overlap Check
    For each concept, check if top-K neighbors share sememes.
    """
    print("\n" + "=" * 60)
    print("AUDIT 1: Sememe Overlap Check")
    print("=" * 60)
    
    # Build force field for KNN
    ff = LocalForceField(positions, device='cpu')
    
    # Parse all nodes
    node_info = [parse_node(n) for n in nodes]
    
    # Get sememes (nodes with | separator)
    sememe_indices = [i for i, info in enumerate(node_info) if info['type'] == 'sememe']
    
    print(f"Total nodes: {len(nodes)}")
    print(f"Sememes: {len(sememe_indices)}")
    print()
    
    # For each test concept, check neighbor types
    for idx in test_indices:
        info = node_info[idx]
        print(f"[Concept {idx}] {info['raw']}")
        
        # Get neighbors
        pos = positions[idx:idx+1].astype(np.float32)
        pos_tensor = __import__('torch').tensor(pos)
        neighbor_ids, _ = ff.get_neighbors(pos_tensor[0], k=k)
        
        # Count neighbor types
        type_counts = Counter()
        same_type_count = 0
        
        for nid in neighbor_ids:
            n_info = node_info[nid]
            type_counts[n_info['type']] += 1
            
            # Check if same sememe domain (simplified: same first letter of sememe)
            if info['type'] == 'sememe' and n_info['type'] == 'sememe':
                if info['en'] and n_info['en'] and info['en'].lower() in n_info['en'].lower():
                    same_type_count += 1
                    
        print(f"  Top-{k} neighbor types: {dict(type_counts)}")
        print(f"  Sample neighbors: {[node_info[nid]['raw'][:30] for nid in neighbor_ids[:5]]}")
        print()
        
    return True


def run_audit_2_skeleton_edge_hit(nodes, positions, edges, test_indices, k=50):
    """
    Audit 2: Skeleton Edge P@K
    Check if geometric neighbors are skeleton neighbors.
    """
    print("\n" + "=" * 60)
    print("AUDIT 2: Skeleton Edge P@K (Placeholder)")
    print("=" * 60)
    
    # Need edge data from training - placeholder for now
    print("NOTE: Need skeleton edge data to compute P@K.")
    print("Will check if edge file exists...")
    
    edge_files = [
        "data/openhownet_skeleton_edges.npy",
        "data/hownet_edges.npy",
        "data/edges.npy"
    ]
    
    for ef in edge_files:
        if os.path.exists(ef):
            print(f"Found: {ef}")
        else:
            print(f"Not found: {ef}")
            
    print("\nSkipping P@K calculation (need edge data).")
    return False


def run_audit_3_language_consistency(nodes, positions, test_indices, k=50):
    """
    Audit 3: Language Consistency
    Check Chinese/English mixing in neighborhoods.
    """
    print("\n" + "=" * 60)
    print("AUDIT 3: Language Consistency")
    print("=" * 60)
    
    ff = LocalForceField(positions, device='cpu')
    node_info = [parse_node(n) for n in nodes]
    
    results = []
    
    for idx in test_indices:
        info = node_info[idx]
        anchor_has_zh = is_chinese(info['zh'] or info['en'] or info['raw'])
        
        # Get neighbors
        pos = positions[idx:idx+1].astype(np.float32)
        pos_tensor = __import__('torch').tensor(pos)
        neighbor_ids, _ = ff.get_neighbors(pos_tensor[0], k=k)
        
        # Count language
        zh_count = 0
        en_count = 0
        other_count = 0
        
        for nid in neighbor_ids:
            n_info = node_info[nid]
            n_text = n_info['zh'] or n_info['en'] or n_info['raw']
            
            if is_chinese(n_text):
                zh_count += 1
            elif n_text and n_text.isascii():
                en_count += 1
            else:
                other_count += 1
                
        anchor_lang = "ZH" if anchor_has_zh else "EN"
        print(f"[{anchor_lang}] {info['raw'][:40]}")
        print(f"  Neighbors: ZH={zh_count}/{k} ({100*zh_count/k:.0f}%), "
              f"EN={en_count}/{k} ({100*en_count/k:.0f}%), "
              f"Other={other_count}/{k}")
        
        results.append({
            'anchor_lang': anchor_lang,
            'zh_ratio': zh_count / k,
            'en_ratio': en_count / k,
        })
        print()
        
    # Summary
    print("=" * 60)
    print("Summary:")
    zh_anchors = [r for r in results if r['anchor_lang'] == 'ZH']
    en_anchors = [r for r in results if r['anchor_lang'] == 'EN']
    
    if zh_anchors:
        avg_zh_in_zh = np.mean([r['zh_ratio'] for r in zh_anchors])
        print(f"  ZH anchors: Avg ZH neighbors = {avg_zh_in_zh:.1%}")
    if en_anchors:
        avg_en_in_en = np.mean([r['en_ratio'] for r in en_anchors])
        print(f"  EN anchors: Avg EN neighbors = {avg_en_in_en:.1%}")
        
    return True


def main():
    checkpoint_path = "checkpoints/aurora_base_gpu_100000.pkl"
    
    print("=" * 60)
    print("BASE NEIGHBORHOOD AUDIT")
    print("=" * 60)
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    data = load_checkpoint(checkpoint_path)
    
    G = data['G']
    nodes = data.get('nodes', [])
    positions = G[:, :, 0]  # (N, dim)
    
    N, dim = positions.shape
    print(f"Loaded {N} particles in H^{dim-1}")
    
    # Test indices: food, animal, human, and some random
    node_info = [parse_node(n) for n in nodes]
    
    # Find specific concepts
    test_targets = ['食物', '动物', 'human', 'food']
    test_indices = []
    
    for target in test_targets:
        for i, n in enumerate(nodes):
            if target in n:
                test_indices.append(i)
                break
                
    # Add random samples
    np.random.seed(42)
    random_samples = np.random.choice(N, size=5, replace=False)
    test_indices.extend(random_samples.tolist())
    
    test_indices = list(set(test_indices))[:10]  # Limit to 10
    
    print(f"\nTest concepts ({len(test_indices)}):")
    for idx in test_indices:
        print(f"  [{idx}] {nodes[idx][:50]}")
    
    # Run audits
    run_audit_1_sememe_overlap(nodes, positions, test_indices)
    run_audit_2_skeleton_edge_hit(nodes, positions, None, test_indices)
    run_audit_3_language_consistency(nodes, positions, test_indices)
    
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Aurora Data: Dataset Management.
================================

Handles loading of structure (Cilin/HowNet) and semantic edges (PMI).
Crucially, creates a runtime Vocabulary and maps all external data (string edges)
to runtime indices, solving the index coupling defect.
"""

import os
import json
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
import pickle

class Vocabulary:
    def __init__(self, words: List[str]):
        self.id_to_word = list(words)
        self.word_to_id = {w: i for i, w in enumerate(words)}
        
    def __len__(self):
        return len(self.id_to_word)
        
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.word_to_id.get(idx, None)
        return self.id_to_word[idx]
        
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.id_to_word, f, ensure_ascii=False)

class AuroraDataset:
    def __init__(self, name: str, limit: Optional[int] = None):
        self.name = name
        self.limit = limit
        
        # Graph Structure
        self.nodes: List[str] = []
        self.edges_struct: List[Tuple[int, int]] = [] # (u, v) indices for Tree
        self.depths: Optional[np.ndarray] = None
        self.root_idx: int = 0
        
        # Load Base Structure
        if name == 'cilin':
            self._load_cilin()
        elif name == 'openhow':
            self._load_openhow()
        else:
            raise ValueError(f"Unknown dataset: {name}")
            
        self.vocab = Vocabulary(self.nodes)
        self.num_nodes = len(self.nodes)
        
    def _load_cilin(self):
        # Re-implement simple loader or import?
        # Let's import the legacy loader to save space, but wrap result
        # Check src/hei_n/datasets/cilin_loader.py
        from hei_n.datasets.cilin_loader import load_cilin_dataset
        nodes, edge_arr, depths, root = load_cilin_dataset(self.limit)
        
        self.nodes = list(nodes)
        self.edges_struct = [(int(u), int(v)) for u, v in edge_arr]
        self.depths = depths
        self.root_idx = root
        
    def _load_openhow(self):
        from hei_n.datasets.hownet_loader import load_dataset
        nodes, edge_arr, depths, root = load_dataset(self.limit)
        
        self.nodes = list(nodes)
        self.edges_struct = [(int(u), int(v)) for u, v in edge_arr]
        self.depths = depths
        self.root_idx = root
        
    def load_semantic_edges(self, path: str) -> List[Tuple[int, int, float]]:
        """
        Load semantic edges from pickle containing (str, str, float) tuples.
        Maps them to current indices.
        """
        if not os.path.exists(path):
            print(f"Warning: Semantic edge file {path} not found.")
            return []
            
        print(f"Loading semantic edges from {path}...")
        with open(path, 'rb') as f:
            raw_edges = pickle.load(f)
            
        # raw_edges format: List[Tuple[u, v, w, type]]?
        # Or just checking what build_wiki_pmi currently outputs?
        # Current build_wiki_pmi outputs ints. 
        # But our plan was to UPDATE it to output strings.
        # Assuming we have valid data (or we will handle int legacy if needed, but implementation plan said fix generation).
        
        valid_edges = []
        miss_count = 0
        
        # Heuristic check: is the first item int or str?
        if raw_edges and isinstance(raw_edges[0][0], int):
            print("WARNING: Loaded edges are Integers (Legacy Format). Using strict index.")
            # This is dangerous if vocab mismatch.
            # But for migration, maybe we allow it? No, force safety.
            print("Error: Cannot safely load Integer edges with decoupled dataset. Please regenerate edges as Strings.")
            return []
            
        for item in raw_edges:
            # Expected: (u_str, v_str, w, type)
            try:
                u_s, v_s = item[0], item[1]
                w = item[2]
                
                u_id = self.vocab[u_s]
                v_id = self.vocab[v_s]
                
                if u_id is not None and v_id is not None:
                    valid_edges.append((u_id, v_id, w))
                else:
                    miss_count += 1
            except:
                continue
                
        print(f"Loaded {len(valid_edges)} semantic edges (Skipped {miss_count} unknown words).")
        return valid_edges

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
        
        # [NEW] Build Raw Word Map for Loose Matching (Index Decoupling Support)
        # Maps "apple" -> candidate index (e.g. index of "C:apple:01")
        self.raw_map = {}
        for idx, node_str in enumerate(self.nodes):
            # Parse C:word:id or Code:code
            if node_str.startswith("C:"):
                # Format C:word:id
                parts = node_str.split(":")
                if len(parts) >= 2:
                    raw_word = parts[1]
                    # Strategy: First Come First Serve (matches ConceptMapper)
                    if raw_word not in self.raw_map:
                        self.raw_map[raw_word] = idx
                        
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
        
    def load_semantic_edges(self, path: str, split: str = "all") -> List[Tuple[int, int, float]]:
        """
        Load semantic edges from pickle containing (str, str, float) tuples.
        Maps them to current indices.
        
        Args:
            split: "all", "train" (90%), or "holdout" (10%). Deterministic split based on index.
        """
        if not os.path.exists(path):
            print(f"Warning: Semantic edge file {path} not found.")
            return []
            
        print(f"Loading semantic edges from {path} (Split: {split})...")
        with open(path, 'rb') as f:
            raw_edges = pickle.load(f)
            
        valid_edges = []
        miss_count = 0
        
        # Heuristic check
        if raw_edges and isinstance(raw_edges[0][0], int):
            print("WARNING: Loaded edges are Integers (Legacy Format). Cannot apply splits safely.")
            return []
            
        # Process edges
        mapped_edges = []
        for item in raw_edges:
            try:
                u_s, v_s = item[0], item[1]
                w = item[2]
                
                # Resolving
                u_id = self.vocab[u_s]
                v_id = self.vocab[v_s]
                
                if u_id is None: u_id = self.raw_map.get(u_s)
                if v_id is None: v_id = self.raw_map.get(v_s)
                
                if u_id is not None and v_id is not None:
                    if u_id != v_id:
                        mapped_edges.append((u_id, v_id, w))
                else:
                    miss_count += 1
            except:
                continue
                
        # Apply deterministic split
        # We'll use a simple modulo on the edge index (assuming random order in file, or shuffle)
        # To be safe, let's shuffle deterministically.
        rng = np.random.RandomState(42) # Fixed seed for splitting
        rng.shuffle(mapped_edges)
        
        total = len(mapped_edges)
        split_idx = int(total * 0.9) # 90/10
        
        if split == "train":
            final_edges = mapped_edges[:split_idx]
        elif split == "holdout":
            final_edges = mapped_edges[split_idx:]
        else:
            final_edges = mapped_edges
            
        print(f"Loaded {len(final_edges)} semantic edges (Split: {split}, Total Pool: {total}, Skipped {miss_count}).")
        return final_edges

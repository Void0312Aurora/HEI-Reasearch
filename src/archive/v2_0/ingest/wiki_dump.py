"""
Wikipedia Scalable Ingestor.
============================

Handles large-scale ingestion of Wikipedia dumps.
Features:
1. Stream Processing (Generator-based).
2. Negative Sampling (Hard Negatives).
3. Statistical Reporting.
"""

import collections
import random
import numpy as np
from typing import List, Tuple, Dict, Generator
from ..data import AuroraDataset
from .wikipedia import WikipediaIngestor

class WikiDumpIngestor(WikipediaIngestor):
    def __init__(self, ds: AuroraDataset, neg_ratio: int = 5):
        super().__init__(ds)
        self.neg_ratio = neg_ratio
        
        # Stats
        self.stats = {
            'flow_edges': 0,
            'phrase_edges': 0,
            'dep_edges': 0,
            'negatives': 0
        }
        self.cached_keys = list(ds.vocab.word_to_id.values())
        
    def stream_dump(self, file_path: str, batch_size: int = 256):
        """
        Stream dump file and yield batches of (tokens, negative_samples).
        Supports:
        - .txt: Raw text, one sentence/doc per line.
        - .json: JSON Lines, expects 'text' field (e.g. WikiExtractor output).
        """
        import json
        is_json = file_path.endswith('.json')
        
        current_batch = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if not text: continue
                
                if is_json:
                    try:
                        obj = json.loads(text)
                        text = obj.get('text', '')
                    except json.JSONDecodeError:
                        continue # Skip bad lines
                
                # Tokenize
                tokens = self.tokenize(text)
                if len(tokens) < 2: continue

                # Positive Edges
                pos_edges = self.ingest_document(text)
                
                # Negative Sampling
                neg_edges = self._sample_negatives(pos_edges, text)
                
                # Add to batch
                for u, v, r in pos_edges:
                    current_batch.append((u, v, r, 1))
                    self._update_stats(r, 1)
                    
                for u, v, r in neg_edges:
                    current_batch.append((u, v, r, 0))
                    self._update_stats(r, 0)
                    
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
                    
        if current_batch:
            yield current_batch
            
    def _sample_negatives(self, pos_edges: List[Tuple[int, int, int]], text: str) -> List[Tuple[int, int, int]]:
        """
        Generate Hard Negatives.
        Strategies:
        1. Flow: Replace v with random word from vocab (Easy) or same sentence (Hard).
        2. Phrase: Reverse order or random word.
        3. Dep: Random Dependent.
        """
        negatives = []
        tokens = self.tokenize(text)
        token_ids = [self.resolve_token(t) for t in tokens]
        valid_ids = [i for i in token_ids if i is not None]
        
        if not valid_ids: return []
        
        if not valid_ids: return []
        
        # vocab_keys = list(self.ds.vocab.word_to_id.values()) # Optim: Cache this
        
        for u, v, r in pos_edges:
            for _ in range(self.neg_ratio):
                # Strategy: 50% Hard (Same Context), 50% Random
                if random.random() < 0.5 and len(valid_ids) > 1:
                    v_neg = random.choice(valid_ids)
                else:
                    v_neg = random.choice(self.cached_keys)
                    
                if v_neg != v:
                    negatives.append((u, v_neg, r))
                    
        return negatives

    def _update_stats(self, r: int, label: int):
        if label == 0:
            self.stats['negatives'] += 1
            return
            
        if r == self.TYPE_FLOW: self.stats['flow_edges'] += 1
        elif r == self.TYPE_PHRASE: self.stats['phrase_edges'] += 1
        else: self.stats['dep_edges'] += 1
        
    def print_stats(self):
        print("\n--- Ingestion Statistics ---")
        print(f"Flow Edges:   {self.stats['flow_edges']}")
        print(f"Phrase Edges: {self.stats['phrase_edges']}")
        print(f"Dep Edges:    {self.stats['dep_edges']}")
        print(f"Negatives:    {self.stats['negatives']}")
        total_pos = self.stats['flow_edges'] + self.stats['phrase_edges'] + self.stats['dep_edges']
        if total_pos > 0:
            print(f"Neg/Pos Ratio: {self.stats['negatives'] / total_pos:.2f}")

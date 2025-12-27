"""
Wikipedia Ingestion Pipeline.
=============================

Converts raw text into Geometric Edges:
1. Flow Edges (Linear Coherence).
2. Phrase Edges (Compound Concepts).
3. Dependency Edges (Syntactic Structure).
"""

import re
import collections
from typing import List, Tuple, Dict, Set
from .dependency import DependencyGeometer
from ..data import AuroraDataset

class WikipediaIngestor:
    def __init__(self, ds: AuroraDataset):
        self.ds = ds
        self.dep_parser = DependencyGeometer() # Reuses the mock parser
        self.vocab_set = set(ds.vocab.word_to_id.keys())
        
        # Edge Types
        self.TYPE_FLOW = 100
        self.TYPE_PHRASE = 101
        
    def resolve_token(self, token: str) -> int:
        """
        Resolve token to ID using Vocab (Exact) or RawMap (Loose).
        """
        # 1. Exact Match (e.g. if token is already "C:Cat:01")
        idx = self.ds.vocab.word_to_id.get(token)
        if idx is not None: return idx
        
        # 2. Raw Map (e.g. "Cat" -> "C:Cat:01")
        if hasattr(self.ds, 'raw_map'):
            return self.ds.raw_map.get(token)
            
        return None
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize using Jieba for Chinese support.
        """
        try:
            import jieba
            return jieba.lcut(text)
        except ImportError:
            # Fallback for English logic
            return text.strip().split()
        
    def extract_flow_edges(self, tokens: List[str], window=2) -> List[Tuple[int, int, int]]:
        """
        w_t -> w_{t+1} ... w_{t+k}
        """
        edges = []
        ids = [self.resolve_token(t) for t in tokens]
        # Filter None
        valid_ids = [(i, idx) for i, idx in enumerate(ids) if idx is not None]
        
        for k in range(len(valid_ids) - 1):
            pos_u, u = valid_ids[k]
            # Next few valid words
            for j in range(1, window + 1):
                if k + j < len(valid_ids):
                    pos_v, v = valid_ids[k+j]
                    # Check distance in original tokens
                    if pos_v - pos_u <= window: 
                        edges.append((u, v, self.TYPE_FLOW))
        return edges

    def extract_phrase_edges(self, tokens: List[str]) -> List[Tuple[int, int, int]]:
        """
        Bigram Phrases.
        """
        edges = []
        # Mock: If two words appear together often? 
        # For single document, just add adjacent.
        # Distinguished from Flow by type.
        # Maybe Phrase implies Stronger Bond (Identity-like).
        ids = [self.resolve_token(t) for t in tokens]
        for i in range(len(ids) - 1):
            if ids[i] is not None and ids[i+1] is not None:
                edges.append((ids[i], ids[i+1], self.TYPE_PHRASE))
        return edges

    def extract_dependency_edges(self, text: str) -> List[Tuple[int, int, int]]:
        """
        Delegate to DependencyGeometer.
        """
        raw_edges = self.dep_parser.text_to_edges(text)
        edges = []
        for u_s, v_s, r in raw_edges:
            u = self.resolve_token(u_s)
            v = self.resolve_token(v_s)
            if u is not None and v is not None:
                edges.append((u, v, r))
        return edges

    def ingest_document(self, text: str) -> List[Tuple[int, int, int]]:
        tokens = self.tokenize(text)
        
        flow = self.extract_flow_edges(tokens)
        phrase = self.extract_phrase_edges(tokens)
        dep = self.extract_dependency_edges(text)
        
        return flow + phrase + dep

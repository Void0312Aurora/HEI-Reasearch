"""
Aurora Dialogue State Machine.
==============================

Manages the Physical State of a Dialogue:
1. Context Trajectory (J_ctx).
2. Semantic Anchors (Reference Candidates).
3. Coherence Budget (R_c).
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from ..gauge import GaugeField
from ..ingest.wikipedia import WikipediaIngestor
import re

class DialogueState:
    def __init__(self, gauge_field: GaugeField, x: torch.Tensor, J: torch.Tensor, 
                 ingestor: WikipediaIngestor, device='cuda'):
        self.gauge_field = gauge_field
        self.x = x
        self.J = J
        self.ingestor = ingestor
        self.device = device
        
        # State
        self.J_ctx = torch.zeros(J.shape[1], device=device) # Init as zero or start token
        self.anchors: Dict[str, int] = {} # "word" -> ID
        self.coherence_budget = 5.4 # R_c from Phase 19
        
    def reset(self):
        self.J_ctx.zero_()
        self.anchors.clear()
        
        return f"{token}({best_word})"

    def _infer_type(self, token_id: int) -> str:
        if hasattr(self.ingestor.ds.vocab, 'id_to_word'):
             node_str = self.ingestor.ds.vocab.id_to_word[token_id]
             # Check prefix
             if "Code:A" in node_str or node_str.startswith("A"):
                 return 'Person'
             if "Code:B" in node_str or node_str.startswith("B"):
                 return 'Thing'
             return 'Thing'
        return 'Thing'

    def update(self, text: str):
        """
        Process user input.
        1. Identify Nouns -> Anchors (with Type).
        2. Transport J_ctx along the path of the sentence.
        """
        tokens = self.ingestor.tokenize(text)
        ids = []
        for t in tokens:
            tid = self.ingestor.ds.vocab.word_to_id.get(t)
            if tid is not None:
                atype = self._infer_type(tid)
                self.anchors[t] = {'id': tid, 'type': atype}
                ids.append(tid)
                
        if ids:
            last_id = ids[-1]
            self.J_ctx = self.J[last_id].clone()
            
    def resolve_reference(self, token: str) -> str:
        """
        Resolve 'he', 'it', 'this' to nearest Anchor using Type Bias.
        """
        pronouns_person = {'he', 'she', 'they', 'who'}
        pronouns_thing = {'it', 'that', 'this', 'which'}
        token_lower = token.lower()
        
        target_type = None
        if token_lower in pronouns_person: target_type = 'Person'
        elif token_lower in pronouns_thing: target_type = 'Thing'
        else: return token
            
        if not self.anchors: return token
            
        best_word = token
        best_score = -999.0
        
        for word, info in self.anchors.items():
            if word.lower() in pronouns_person or word.lower() in pronouns_thing: continue
            
            aid = info['id']
            atype = info['type']
            
            J_a = self.J[aid]
            score = torch.sum(self.J_ctx * J_a).item()
            
            # Type Bias
            if target_type and atype == target_type:
                score += 1.0 # Strong Bias
            elif target_type and atype != target_type:
                score -= 1.0 # Penalty
                
            if score > best_score:
                best_score = score
                best_word = word
                
        return f"{token}({best_word})"

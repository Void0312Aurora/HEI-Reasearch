"""
Aurora Geometric Chatbot.
=========================

Implements a dialogue agent based on Geometric Trajectory Inference.
The "Thinking" is Parallel Transport.
The "Speaking" is Decoding local neighborhoods.
"""

import torch
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
from .gauge import GaugeField
from .data import AuroraDataset

class GeometricBot:
    def __init__(self, gauge_field: GaugeField, x: torch.Tensor, J: torch.Tensor, ds: AuroraDataset, G: nx.Graph, device='cuda'):
        self.gauge_field = gauge_field
        self.x = x
        self.J = J # Reference Frame (Physics State)
        self.ds = ds
        self.G = G
        self.device = device
        
    def encode(self, text: str) -> int:
        """Text to Node ID."""
        # Exact match
        if text in self.ds.vocab.word_to_id:
            return self.ds.vocab.word_to_id[text]
            
        # Search label
        for n_id, label in enumerate(self.ds.nodes):
            if text in label:
                return n_id
                
        return -1
        
    def decode(self, idx: int) -> str:
        """Node ID to Text."""
        raw = self.ds.nodes[idx]
        return raw.split(":")[1] if ":" in raw else raw

    def reply(self, text: str, mode='assoc') -> str:
        """
        Generate a single response.
        Modes:
        - 'assoc': Associative (Neutral Transport).
        - 'opp': Opposite? (Not implemented yet).
        """
        u = self.encode(text)
        if u == -1:
            return "I don't know that concept."
            
        # 1. Get Candidates (Neighbors)
        candidates = list(self.G.neighbors(u))
        if not candidates:
            return "I have nothing to say about that."
            
        # 2. Physics: Transport J_u to Neighbors
        cand_t = torch.tensor(candidates, dtype=torch.long, device=self.device)
        u_t = torch.full((len(candidates),), u, dtype=torch.long, device=self.device)
        edges = torch.stack([u_t, cand_t], dim=1)
        
        # U (Neutral for now, or imagine U_QA is learned)
        U = self.gauge_field.get_U(x=self.x, edges=edges)
        
        J_u = self.J[u]
        J_u_exp = J_u.view(1, -1, 1).expand(len(candidates), -1, -1)
        J_trans = torch.matmul(U, J_u_exp).squeeze(-1)
        
        J_cand = self.J[cand_t]
        
        # Alignment
        alignment = torch.sum(J_cand * J_trans, dim=-1)
        
        # Select Best
        best_idx = torch.argmax(alignment).item()
        target_id = candidates[best_idx]
        
        return self.decode(target_id)

    def stream_reply(self, text: str, length=5) -> List[str]:
        """
        Generate a train of thought.
        """
        u = self.encode(text)
        if u == -1: return ["Unknown"]
        
        chain = []
        curr = u
        visited = {u}
        
        for _ in range(length):
            candidates = list(self.G.neighbors(curr))
            candidates = [c for c in candidates if c not in visited]
            if not candidates: break
            
            # Physics Step (Same as reply but iterative)
            cand_t = torch.tensor(candidates, dtype=torch.long, device=self.device)
            u_t = torch.full((len(candidates),), curr, dtype=torch.long, device=self.device)
            edges = torch.stack([u_t, cand_t], dim=1)
            
            U = self.gauge_field.get_U(x=self.x, edges=edges)
            J_curr = self.J[curr]
            J_curr_exp = J_curr.view(1, -1, 1).expand(len(candidates), -1, -1)
            J_trans = torch.matmul(U, J_curr_exp).squeeze(-1)
            
            alignment = torch.sum(self.J[cand_t] * J_trans, dim=-1)
            best_i = torch.argmax(alignment).item()
            next_node = candidates[best_i]
            
            chain.append(self.decode(next_node))
            visited.add(next_node)
            curr = next_node
            
        return chain

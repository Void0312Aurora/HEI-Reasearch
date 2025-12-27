"""
Aurora Language Trainer.
========================

Trains the Typed Gauge Field on Wikipedia Edges.
Objective: Minimize Mixed Energy of Flow, Phrase, and Dependency.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Iterator
from ..gauge import GaugeField
from ..data import AuroraDataset

class LanguageTrainer:
    def __init__(self, gauge_field: GaugeField, x: torch.Tensor, J: torch.Tensor, 
                 lr: float = 0.001, device='cuda'):
        self.gauge_field = gauge_field
        self.x = x
        self.J = J # Fixed Semantic Skeleton
        self.device = device
        
        self.optimizer = optim.Adam(self.gauge_field.parameters(), lr=lr)
        
    def train_epoch(self, edges: List[Tuple[int, int, int]], batch_size=64) -> float:
        self.gauge_field.train()
        total_loss = 0.0
        np.random.shuffle(edges)
        
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size]
            u_b = torch.tensor([e[0] for e in batch], device=self.device)
            v_b = torch.tensor([e[1] for e in batch], device=self.device)
            r_b = torch.tensor([e[2] for e in batch], device=self.device)
            
            self.optimizer.zero_grad()
            
            # Transport
            edges_t = torch.stack([u_b, v_b], dim=1)
            # Note: r_b can be 100, 101. Ideally remap to 0..N for embedding index?
            # NeuralBackend uses nn.Embedding(num_relations).
            # If num_relations is small, index 100 will crash.
            # We assume user initialized GaugeField with num_relations=200 or remapped.
            
            U = self.gauge_field.get_U(x=self.x, edges=edges_t, relation_ids=r_b)
            
            J_u = self.J[u_b].unsqueeze(-1)
            J_trans = torch.matmul(U, J_u).squeeze(-1)
            J_v = self.J[v_b]
            
            # Alignment Loss
            align = torch.sum(J_v * J_trans, dim=-1)
            loss = torch.mean(1.0 - align)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / (len(edges) / batch_size)

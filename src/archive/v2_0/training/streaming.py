"""
Aurora Streaming Trainer.
=========================

Implements continuous training for Corpus-Scale Deployment.
- Accepts infinite stream of Geometric Edges.
- Updates Gauge Field on-the-fly.
- Mocks "Plasticity" handling via rolling buffer.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Iterator, Tuple, List
from ..gauge import GaugeField
from ..data import AuroraDataset

class StreamingTrainer:
    def __init__(self, gauge_field: GaugeField, x: torch.Tensor, J: torch.Tensor, ds: AuroraDataset, 
                 lr: float = 0.001, buffer_size: int = 1000, device='cuda'):
        self.gauge_field = gauge_field
        self.x = x
        self.J = J
        self.ds = ds
        self.device = device
        self.buffer_size = buffer_size
        
        # Optimizer
        self.optimizer = optim.Adam(self.gauge_field.parameters(), lr=lr)
        
        # Rolling Buffer
        self.edge_buffer = [] # List[Tuple[u, v, r]]
        
    def train_stream(self, edge_stream: Iterator[Tuple[str, str, int]], steps: int = 1000):
        """
        Consume the stream and update model.
        """
        self.gauge_field.train()
        
        step = 0
        total_loss = 0.0
        
        for u_str, v_str, r_id in edge_stream:
            # Map strings to IDs
            if u_str not in self.ds.vocab.word_to_id or v_str not in self.ds.vocab.word_to_id:
                # print(f"Skipping {u_str} -> {v_str}")
                continue
                
            # DEBUG: Print first success
            if step == 0:
                print(f"Processing first edge: {u_str} -> {v_str} (Rel: {r_id})")
                
            u = self.ds.vocab.word_to_id[u_str]
            v = self.ds.vocab.word_to_id[v_str]
            
            self.edge_buffer.append((u, v, r_id))
            
            # Maintain Buffer Size
            if len(self.edge_buffer) > self.buffer_size:
                self.edge_buffer.pop(0) # FIFO (Forget old)
                
            # Update Step every 10 items
            if len(self.edge_buffer) >= 10 and step % 10 == 0:
                loss = self._update_step()
                total_loss += loss
                
                if step % 100 == 0:
                    print(f"[Streaming] Step {step}: Loss {loss:.6f}")
                    
            step += 1
            if step >= steps:
                break
                
    def _update_step(self) -> float:
        """
        Perform one update on buffer.
        Loss = Sum ( 1 - Align(u, v) )
        """
        self.optimizer.zero_grad()
        
        # Batch from buffer
        batch_size = min(len(self.edge_buffer), 32)
        indices = np.random.choice(len(self.edge_buffer), batch_size, replace=False)
        batch = [self.edge_buffer[i] for i in indices]
        
        u_batch = torch.tensor([b[0] for b in batch], device=self.device)
        v_batch = torch.tensor([b[1] for b in batch], device=self.device)
        r_batch = torch.tensor([b[2] for b in batch], device=self.device)
        
        edges_t = torch.stack([u_batch, v_batch], dim=1)
        
        # Compute U
        U = self.gauge_field.get_U(x=self.x, edges=edges_t, relation_ids=r_batch)
        
        # Compute Alignment
        J_u = self.J[u_batch]
        J_v = self.J[v_batch]
        
        J_u_exp = J_u.unsqueeze(1).unsqueeze(-1) # (B, 1, k, 1) ? No.
        # U is (B, k, k). J_u is (B, k).
        J_u_exp = J_u.unsqueeze(-1) # (B, k, 1)
        J_trans = torch.matmul(U, J_u_exp).squeeze(-1) # (B, k)
        
        alignment = torch.sum(J_v * J_trans, dim=-1) # (B,)
        
        loss = torch.mean(1.0 - alignment)
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


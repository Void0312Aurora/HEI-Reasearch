"""
Aurora Memory Engine (CCD v3.1)
===============================

Implements Long-Term Memory (LTM) on the Poincare Ball.
Stores crystallized states (Psi, u) and supports efficient
hyperbolic retrieval.

Ref: Axiom 4.3.2 (Topological Crystallization).
"""

import torch
import torch.nn as nn
from ..physics import geometry

class HyperbolicMemory(nn.Module):
    def __init__(self, dim: int, max_capacity: int = 10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_capacity = max_capacity
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Storage Buffers
        # We pre-allocate for speed? Or dynamic list?
        # Pre-allocate is better for PyTorch.
        
        self.register_buffer('q_store', torch.zeros(max_capacity, dim))
        self.register_buffer('m_store', torch.zeros(max_capacity, 1))
        self.register_buffer('J_store', torch.zeros(max_capacity, dim, dim))
        self.register_buffer('p_store', torch.zeros(max_capacity, dim))
        
        # Metadata check? (e.g. char_id, time)
        # For now, just physics.
        
        self.ptr = 0
        self.size = 0
        
    def add(self, q: torch.Tensor, m: torch.Tensor, J: torch.Tensor, p: torch.Tensor):
        """
        Add batch of particles to memory.
        q, p: (B, D)
        m: (B, 1)
        J: (B, D, D)
        """
        B = q.size(0)
        
        # FIFO buffer logic
        indices = torch.arange(self.ptr, self.ptr + B, device=self.device) % self.max_capacity
        
        self.q_store[indices] = q.detach().to(self.device).float()
        self.m_store[indices] = m.detach().to(self.device).float()
        self.J_store[indices] = J.detach().to(self.device).float()
        self.p_store[indices] = p.detach().to(self.device).float()
        
        self.ptr = (self.ptr + B) % self.max_capacity
        self.size = min(self.size + B, self.max_capacity)
        
    def query(self, q_query: torch.Tensor, k: int = 10, mask_indices: torch.Tensor = None):
        """
        Retrieve k nearest neighbors in Hyperbolic Space.
        q_query: (B, D)
        """
        if self.size == 0:
            return None
            
        # Compute pairwise distances
        # dist(x, y)
        # Using geometry.dist broadcasting?
        # q_query: (B, 1, D)
        # self.q_store[:self.size]: (1, N, D)
        
        # NOTE: Full scan is simple but O(N). For N=10k it's fast on GPU.
        # For N=1M we need specific index (HNSW-Hyperbolic).
        # Phase 34 prototype: Linear Scan.
        
        valid_q = self.q_store[:self.size]
        
        # Expand
        # q_query = (B, 1, D)
        # valid_q = (1, N, D)
        
        # Mobius Addition for search?
        # dist = acosh(...) or 2 atanh(...)
        # geometry.dist handles broadcast?
        # Let's check geometry.dist signature. usually expects matching shapes or broadcast.
        
        # Since geometry.dist calls mobius_add, which broadcasts.
        
        dists = geometry.dist(q_query.unsqueeze(1), valid_q.unsqueeze(0)) # Result (B, N)
        
        # Topk (Smallest distance)
        # If N < k, return all.
        k_actual = min(k, self.size)
        
        vals, idxs = torch.topk(dists, k_actual, dim=1, largest=False)
        
        # Gather Results
        # ret_q: (B, k, D)
        ret_q = self.q_store[idxs]
        ret_m = self.m_store[idxs]
        ret_J = self.J_store[idxs]
        ret_p = self.p_store[idxs]
        
        return {
            'q': ret_q,
            'm': ret_m,
            'J': ret_J,
            'p': ret_p,
            'dists': vals,
            'indices': idxs
        }
        
    def get_all(self):
        """Return all memories (for Background Field)."""
        if self.size == 0:
            return None
        return {
            'q': self.q_store[:self.size],
            'm': self.m_store[:self.size],
            'J': self.J_store[:self.size],
            'p': self.p_store[:self.size]
        }

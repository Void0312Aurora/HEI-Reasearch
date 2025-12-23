"""
Local Force Field for Aurora Interaction Engine.
================================================

Efficient KNN-based force computation using Faiss.
Only computes forces from top-K nearest neighbors instead of full O(N²).
"""

import torch
import numpy as np
from typing import Tuple, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Using brute-force KNN.")


class LocalForceField:
    """
    Compute local forces from nearest neighbors in Aurora Base.
    
    Uses Faiss for efficient KNN search in hyperbolic space
    (approximated as L2 distance in embedding space).
    """
    
    def __init__(self, positions: np.ndarray, device: str = 'cuda'):
        """
        Initialize force field with particle positions.
        
        Args:
            positions: (N, dim) array of particle positions (x vectors)
            device: torch device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        self.N, self.dim = positions.shape
        
        # Build Faiss index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(positions.astype(np.float32))
        else:
            self.index = None
            
        # Cache for neighbors
        self._cached_ids = None
        self._cached_pos = None
        self._cache_step = -1
        
    def get_neighbors(self, cursor_pos: torch.Tensor, k: int = 256) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Get k nearest neighbors to cursor.
        
        Args:
            cursor_pos: (dim,) cursor position
            k: number of neighbors
            
        Returns:
            neighbor_ids: (k,) array of particle IDs
            neighbor_pos: (k, dim) tensor of positions
        """
        k = min(k, self.N)
        
        if FAISS_AVAILABLE:
            # Faiss search
            query = cursor_pos.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
            _, I = self.index.search(query, k)
            neighbor_ids = I[0]
        else:
            # Brute force
            dists = torch.norm(self.positions - cursor_pos.unsqueeze(0), dim=1)
            _, indices = torch.topk(dists, k, largest=False)
            neighbor_ids = indices.cpu().numpy()
            
        neighbor_pos = self.positions[neighbor_ids]
        return neighbor_ids, neighbor_pos
    
    def compute_local_force(
        self, 
        cursor_pos: torch.Tensor, 
        neighbor_pos: torch.Tensor,
        attraction_strength: float = 1.0,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute attractive force from neighbors to cursor.
        
        Uses hyperbolic distance approximation:
        F = -k * tanh(d/sigma) * grad(d)
        
        Args:
            cursor_pos: (dim,) cursor position
            neighbor_pos: (k, dim) neighbor positions
            attraction_strength: force strength
            sigma: distance scale
            
        Returns:
            force: (dim,) total force on cursor
        """
        # Compute hyperbolic distances
        # <x, y>_M = -x0*y0 + x_rest * y_rest
        J = torch.ones(self.dim, device=self.device)
        J[0] = -1.0
        
        # Inner product
        inner = torch.sum(cursor_pos * neighbor_pos * J, dim=-1)  # (k,)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        
        # Distance
        dists = torch.acosh(-inner)  # (k,)
        
        # Force magnitude: tanh for bounded force
        force_mag = attraction_strength * torch.tanh(dists / sigma)
        
        # Gradient direction: -grad(d) = direction toward neighbor
        denom = torch.sqrt(inner**2 - 1.0)
        denom = torch.clamp(denom, min=1e-7)
        
        # grad_d points away from neighbor, we want attraction
        grad_d = (-1.0 / denom).unsqueeze(-1) * (neighbor_pos * J)
        
        # Total force (sum of attractions)
        forces = force_mag.unsqueeze(-1) * (-grad_d)  # Negative for attraction
        total_force = torch.sum(forces, dim=0)
        
        return total_force
    
    def get_force_at(
        self, 
        cursor_pos: torch.Tensor, 
        k: int = 256,
        step: int = 0,
        cache_interval: int = 10
    ) -> torch.Tensor:
        """
        Get total background force at cursor position.
        
        Uses caching to avoid recomputing neighbors every step.
        
        Args:
            cursor_pos: (dim,) cursor position
            k: number of neighbors
            step: current simulation step
            cache_interval: refresh neighbors every N steps
            
        Returns:
            force: (dim,) total force
        """
        # Refresh cache periodically
        if self._cached_ids is None or (step - self._cache_step) >= cache_interval:
            self._cached_ids, self._cached_pos = self.get_neighbors(cursor_pos, k)
            self._cache_step = step
            
        return self.compute_local_force(cursor_pos, self._cached_pos)

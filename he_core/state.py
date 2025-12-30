import torch
import numpy as np
from typing import Optional

class ContactState:
    """
    Geometric State Object for Contact Manifold M = T*Q x R.
    Manages (q, p, s) components.
    Supports initialization from flat tensor [q, p, s] for differentiable flow.
    """
    def __init__(self, dim_q: int, batch_size: int = 1, device: str = 'cpu', flat_tensor: Optional[torch.Tensor] = None):
        self.dim_q = dim_q
        self.batch_size = batch_size
        self.device = device
        self.dim_total = 2 * dim_q + 1
        
        if flat_tensor is not None:
            # Reconstruct from flat (q, p, s)
            # Size: n*dim + n*dim + n*1
            expected = 2 * dim_q * batch_size + batch_size
            if flat_tensor.numel() != expected:
                raise ValueError(f"Flat tensor size {flat_tensor.numel()} mismatch expected {expected}")
                
            self._data = flat_tensor.view(batch_size, self.dim_total)
        else:
            # Init zeros
            self._data = torch.zeros(batch_size, self.dim_total, device=device)
            
    @property
    def q(self) -> torch.Tensor:
        """Configuration q: [Batch, dim_q]"""
        return self._data[:, :self.dim_q]
    
    @q.setter
    def q(self, value: torch.Tensor):
        assert value.shape == (self.batch_size, self.dim_q)
        self._data[:, :self.dim_q] = value
        
    @property
    def p(self) -> torch.Tensor:
        """Momentum p: [Batch, dim_q]"""
        return self._data[:, self.dim_q : 2*self.dim_q]
    
    @p.setter
    def p(self, value: torch.Tensor):
        assert value.shape == (self.batch_size, self.dim_q)
        self._data[:, self.dim_q : 2*self.dim_q] = value
        
    @property
    def s(self) -> torch.Tensor:
        """Contact s: [Batch, 1]"""
        return self._data[:, 2*self.dim_q:]
    
    @s.setter
    def s(self, value: torch.Tensor):
        assert value.shape == (self.batch_size, 1)
        self._data[:, 2*self.dim_q:] = value
        
    @property
    def flat(self) -> torch.Tensor:
        """Full state vector [q, p, s]"""
        return self._data
        
    def clone(self):
        return ContactState(self.dim_q, self.batch_size, self.device, self._data.clone())
    
    def detach(self):
        return ContactState(self.dim_q, self.batch_size, self.device, self._data.detach())
        
    def numpy(self):
        return self._data.detach().cpu().numpy()
        
    def requires_grad_(self, requires_grad: bool = True):
        self._data.requires_grad_(requires_grad)
        return self
        
    def __repr__(self):
        return f"ContactState(dim_q={self.dim_q}, q={self.q.shape}, p={self.p.shape}, s={self.s.shape})"

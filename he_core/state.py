import torch
import numpy as np

class ContactState:
    """
    Geometric State Object for Contact Manifold M = T*Q x R.
    Manages (q, p, s) components and their vectorization.
    """
    def __init__(self, dim_q: int, batch_size: int = 1, device: str = 'cpu', data: torch.Tensor = None):
        self.dim_q = dim_q
        self.batch_size = batch_size
        self.device = device
        self.dim_total = 2 * dim_q + 1
        
        if data is not None:
            # Init from existing tensor
            assert data.shape == (batch_size, self.dim_total), f"Shape mismatch: {data.shape} vs {(batch_size, self.dim_total)}"
            self._data = data.to(device)
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

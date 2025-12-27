"""
Aurora Potentials: Vectorized Force Fields.
===========================================

Implements Energy and Gradient computations for:
1. Structural Edges (Tree)
2. Semantic Edges (Graph)
3. Volume Control (Radius Anchor & Repulsion)

Design: Accepts edge indices as Tensors. The indices come from Data module.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from .geometry import dist_hyperbolic, minkowski_inner

class ForceField(nn.Module):
    """Base class for potentials."""
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Energy: (scalar) Total potential energy
            Gradient: (N, dim) Euclidean gradient w.r.t x (before metric scaling)
        """
        raise NotImplementedError

class SpringPotential(ForceField):
    """
    Attracts connected pairs: V = 0.5 * k * (d - l0)^2.
    """
    def __init__(self, edges: torch.Tensor, k: float, l0: float = 0.0):
        """
        Args:
            edges: (E, 2) LongTensor of indices
            k: stiffness
            l0: rest length
        """
        super().__init__()
        self.register_buffer('edges', edges)
        self.k = k
        self.l0 = l0
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, dim)
        u_idx = self.edges[:, 0]
        v_idx = self.edges[:, 1]
        
        xu = x[u_idx]
        xv = x[v_idx]
        
        # d(u, v)
        # inner = -u0v0 + ...
        # d = acosh(-inner)
        
        # Manually inline geometry for grad
        J = torch.ones(x.shape[-1], device=x.device); J[0] = -1.0
        
        inner = (xu * xv * J).sum(dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dist = torch.acosh(-inner)
        
        # Energy
        delta = dist - self.l0
        energy_vec = 0.5 * self.k * delta**2
        total_energy = energy_vec.sum()
        
        # Gradient
        # dV/du = k * delta * grad_u(d)
        # grad_u(d) = (-1/sqrt(inner^2-1)) * J*v
        
        denom = torch.sqrt(inner**2 - 1.0)
        force_mag = self.k * delta #(E,)
        
        factor = -(force_mag / (denom + 1e-9)).unsqueeze(-1) # (E,1)
        
        grad_u = factor * (xv * J)
        grad_v = factor * (xu * J)
        
        # Accumulate
        grad = torch.zeros_like(x)
        grad.index_add_(0, u_idx, grad_u)
        grad.index_add_(0, v_idx, grad_v)
        
        return total_energy, grad

class RadiusAnchorPotential(ForceField):
    """
    Volume Control: V = 0.5 * lambda * (r_i - r_target)^2.
    """
    def __init__(self, targets: torch.Tensor, lamb: float = 1.0):
        super().__init__()
        self.targets = targets
        self.lamb = lamb
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # r = acosh(x0)
        x0 = x[:, 0]
        # x0 should be >= 1.
        x0_safe = torch.clamp(x0, min=1.0 + 1e-7)
        r = torch.acosh(x0_safe)
        
        delta = r - self.targets
        energy = 0.5 * self.lamb * (delta**2).sum()
        
        # Grad
        # dV/dx0 = lambda * delta * (1/sinh(r))
        sinh_r = torch.sqrt(x0_safe**2 - 1.0)
        grad_0 = self.lamb * delta / (sinh_r + 1e-9)
        
        grad = torch.zeros_like(x)
        grad[:, 0] = grad_0
        
        return energy, grad

class RepulsionPotential(ForceField):
    """
    Negative Sampling Repulsion: LogCosh.
    V = A * log(cosh(d / sigma))
    """
    def __init__(self, A: float = 5.0, sigma: float = 1.0, num_neg: int = 5):
        super().__init__()
        self.A = A
        self.sigma = sigma
        self.num_neg = num_neg
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, dim = x.shape
        device = x.device
        
        # Sample negs
        u_idx = torch.randint(0, N, (N * self.num_neg,), device=device)
        v_idx = torch.randint(0, N, (N * self.num_neg,), device=device)
        
        xu = x[u_idx]
        xv = x[v_idx]
        
        J = torch.ones(dim, device=device); J[0] = -1.0
        inner = (xu * xv * J).sum(dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dist = torch.acosh(-inner)
        
        # Energy
        val = dist / self.sigma
        
        # Stable log(cosh(x))
        # For large |x|, log(cosh(x)) approx |x| - log(2)
        # We can use softplus logic or simple masking.
        abs_val = torch.abs(val)
        mask_large = abs_val > 10.0
        
        logcosh = torch.zeros_like(val)
        # Large x: |x| - log(2)
        logcosh[mask_large] = abs_val[mask_large] - 0.69314718
        # Small x: log(cosh(x))
        logcosh[~mask_large] = torch.log(torch.cosh(val[~mask_large]))
        
        energy = (self.A * self.sigma * logcosh).sum() * (1.0 / self.num_neg)
        
        # Grad
        # dV/dd = A * tanh(val)
        force_mag = self.A * torch.tanh(val) # (E_neg,)
        
        denom = torch.sqrt(inner**2 - 1.0)
        # d(d)/du = -1/sqrt * J v
        factor = -(force_mag / (denom + 1e-9)).unsqueeze(-1)
        
        grad_u = factor * (xv * J)
        grad_v = factor * (xu * J)
        
        grad = torch.zeros_like(x)
        # Need to scale by sampling density? Usually handled by learning rate or A.
        # Just pure sum for now.
        grad.index_add_(0, u_idx, grad_u)
        grad.index_add_(0, v_idx, grad_v)
        
        return energy, grad

class CompositePotential(ForceField):
    def __init__(self, components: List[ForceField]):
        super().__init__()
        self.components = nn.ModuleList(components)
        
    def compute_forces(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_e = 0.0
        total_g = torch.zeros_like(x)
        
        for comp in self.components:
            e, g = comp.compute_forces(x)
            total_e = total_e + e
            total_g = total_g + g
            
        return total_e, total_g


"""
PyTorch Potentials and Kernels for HEI-N.
========================================

Vectorized Potential Energy Functions and Gradients.
"""

import torch
import torch.nn as nn
from typing import Protocol, List

class KernelTorch(Protocol):
    def forward(self, dists: torch.Tensor) -> torch.Tensor: ...
    def derivative(self, dists: torch.Tensor) -> torch.Tensor: ...

class SpringAttractionTorch:
    def __init__(self, k: float, l0: float = 0.0):
        self.k = k
        self.l0 = l0
        
    def forward(self, dists: torch.Tensor) -> torch.Tensor:
        delta = dists - self.l0
        return 0.5 * self.k * delta**2
        
    def derivative(self, dists: torch.Tensor) -> torch.Tensor:
        return self.k * (dists - self.l0)

class LogCoshRepulsionTorch:
    def __init__(self, sigma: float, A: float):
        self.sigma = sigma
        self.A = A
        
    def forward(self, dists: torch.Tensor) -> torch.Tensor:
        # A * sigma * log(cosh(d/sigma))
        val = dists / self.sigma
        # Softplus approximation for stability: log(exp(x)+exp(-x)) approx |x| + log(1+exp(-2|x|))
        # But standard logcosh is fine if we avoid overflow.
        # torch.nn.functional.softplus(2*x) - x is log(1+exp(2x)).
        # Let's trust torch's stability or implement safe.
        # log(cosh(x)) = x - log(2) + log(1 + e^-2x) for large x
        
        # Simple for now
        return self.A * self.sigma * torch.log(torch.cosh(val))
        
    def derivative(self, dists: torch.Tensor) -> torch.Tensor:
        # A * tanh(d/sigma)
        return self.A * torch.tanh(dists / self.sigma)

class SparseEdgePotentialTorch:
    def __init__(self, edges: torch.Tensor, kernel: KernelTorch):
        """
        edges: LongTensor (E, 2)
        kernel: Attraction kernel
        """
        self.edges = edges
        self.kernel = kernel
        
    def potential_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (scalar_energy, grad_x).
        """
        # x: (N, dim)
        device = x.device
        dim = x.shape[-1]
        
        u_idx = self.edges[:, 0]
        v_idx = self.edges[:, 1]
        
        xu = x[u_idx] # (E, dim)
        xv = x[v_idx]
        
        # Hyperbolic Distance
        # <u, v> = -u0v0 + uivi
        J_vec = torch.ones(dim, device=device)
        J_vec[0] = -1.0
        
        inner = torch.sum(xu * xv * J_vec, dim=-1) # (E,)
        # Clamp for safety
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dists = torch.acosh(-inner)
        
        # Energy
        energy = torch.sum(self.kernel.forward(dists))
        
        # Gradient
        # dV/dxu = V'(d) * dd/dxu
        # dd/dxu = -1/sqrt(inner^2 - 1) * d(inner)/dxu
        # d(inner)/dxu = J xv
        
        force_mag = self.kernel.derivative(dists) # V'(d)
        # denom = sinh(d) = sqrt(cosh^2 - 1) = sqrt(inner^2 - 1)
        denom_sq = inner**2 - 1.0
        # approx sqrt handling
        denom = torch.sqrt(torch.clamp(denom_sq, min=1e-9))
        
        # Pre-factor for gradients
        # grad_u = factor * xv
        # factor = force_mag * (-1/denom)
        factor = -force_mag / denom
        factor = factor.unsqueeze(-1) # (E, 1)
        
        # Apply J (Minkowski metric) to vectors for inner product derivative
        # d<u,v>/du_i = s_i * v_i where s is sign vector
        J_xv = xv * J_vec
        J_xu = xu * J_vec
        
        grad_u = factor * J_xv
        grad_v = factor * J_xu
        
        # Scatter add
        grad = torch.zeros_like(x)
        grad.index_add_(0, u_idx, grad_u)
        grad.index_add_(0, v_idx, grad_v)
        
        return energy, grad

class NegativeSamplingPotentialTorch:
    def __init__(self, kernel: KernelTorch, num_neg: int = 10, rescale: float = 1.0):
        self.kernel = kernel
        self.num_neg = num_neg
        self.rescale = rescale
        
    def potential_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, dim = x.shape
        device = x.device
        
        # Sample pairs
        # Source nodes: random N? Or just iterate all nodes? 
        # For training, usually iterate all nodes as source is fine if N=100k? 
        # No, iterating all nodes * num_neg = 1M pairs.
        # This fits in GPU memory (1M * 5 floats).
        
        u_idx = torch.randint(0, N, (N * self.num_neg,), device=device)
        v_idx = torch.randint(0, N, (N * self.num_neg,), device=device)
        
        # Ideally exclude self loops u=v. But rare for large N.
        
        xu = x[u_idx]
        xv = x[v_idx]
        
        J_vec = torch.ones(dim, device=device)
        J_vec[0] = -1.0
        
        inner = torch.sum(xu * xv * J_vec, dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dists = torch.acosh(-inner)
        
        energy = torch.sum(self.kernel.forward(dists)) * self.rescale * (N / (N * self.num_neg)) 
        # Rescale energy to match? Usually just sum.
        # But original logic had a rescale factor (0.05 etc).
        
        force_mag = self.kernel.derivative(dists)
        denom = torch.sqrt(torch.clamp(inner**2 - 1.0, min=1e-9))
        factor = -force_mag / denom * self.rescale
        factor = factor.unsqueeze(-1)
        
        J_xv = xv * J_vec
        J_xu = xu * J_vec
        
        grad_u = factor * J_xv
        grad_v = factor * J_xu
        
        grad = torch.zeros_like(x)
        grad.index_add_(0, u_idx, grad_u)
        grad.index_add_(0, v_idx, grad_v)
        
        return energy, grad

class HarmonicPriorTorch:
    def __init__(self, k: float, center: torch.Tensor = None):
        self.k = k
        self.center = center # (dim,)
        
    def potential_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # V = 0.5 * k * d(x, origin)^2
        # d = acosh(x0) if origin=(1,0..)
        
        # If center is origin (1,0,0...)
        # inner = -x0
        # dist = acosh(x0)
        
        # Assuming origin for simplicity, or handle center.
        # Most usage is origin.
        
        if self.center is None:
            # Origin case (optimized)
            x0 = x[:, 0]
            # Safety
            x0 = torch.clamp(x0, min=1.0 + 1e-7)
            dists = torch.acosh(x0)
            
            energy = torch.sum(0.5 * self.k * dists**2)
            
            force = self.k * dists # (N,)
            denom = torch.sqrt(x0**2 - 1.0)
            factor = force / (denom + 1e-9) # Safety denom
            
            grad = torch.zeros_like(x)
            grad[:, 0] = factor
        else:
            # Generic Center
            device = x.device
            dim = x.shape[-1]
            J_vec = torch.ones(dim, device=device)
            J_vec[0] = -1.0
            
            # <x, e0>
            inner = torch.sum(x * self.center * J_vec, dim=-1)
            # Clamp inner <= -1
            inner = torch.clamp(inner, max=-1.0 - 1e-7)
            
            dists = torch.acosh(-inner)
            energy = torch.sum(0.5 * self.k * dists**2)
            
            # Grad
            # grad d = 1/sinh(d) * (-J e0) ? 
            # grad_x <x, e0> = J e0
            # grad d = (-1/sqrt) * J e0
            # grad V = k * d * (-1/sqrt) * J e0
            
            denom = torch.sqrt(inner**2 - 1.0)
            factor = - (self.k * dists) / (denom + 1e-9)
            
            grad = factor.unsqueeze(-1) * (self.center * J_vec)
             
        return energy, grad


class RadiusAnchorPotentialTorch:
    """
    Soft radius regularization: V = λ Σ (r_i - r_target_i)²
    
    Anchors each node's radius to a per-node target (typically from depth mapping).
    Prevents global collapse or expansion while allowing angular optimization.
    """
    def __init__(self, target_radii: torch.Tensor, lamb: float = 1.0):
        """
        target_radii: (N,) Tensor of target radii for each node.
        lamb: Regularization strength.
        """
        self.target_radii = target_radii
        self.lamb = lamb
        
    def potential_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (N, dim), time-first convention.
        # Radius r = acosh(x0).
        
        x0 = x[:, 0]
        x0_safe = torch.clamp(x0, min=1.0 + 1e-7)
        radii = torch.acosh(x0_safe)
        
        delta = radii - self.target_radii
        energy = torch.sum(0.5 * self.lamb * delta**2)
        
        # Gradient: dV/dx0 = λ(r - r_target) * dr/dx0
        # dr/dx0 = 1 / sqrt(x0² - 1) = 1 / sinh(r)
        sinh_r = torch.sqrt(x0_safe**2 - 1.0)
        grad_factor = self.lamb * delta / (sinh_r + 1e-9)
        
        grad = torch.zeros_like(x)
        grad[:, 0] = grad_factor
        
        return energy, grad


class ShortRangeGatedRepulsionTorch:
    """
    Short-range gated repulsion: Only repels when d < epsilon.
    
    Beyond epsilon, repulsion decays rapidly to avoid driving global expansion.
    """
    def __init__(self, A: float, epsilon: float = 0.5, sigma: float = 0.2):
        """
        A: Repulsion amplitude.
        epsilon: Distance threshold for full repulsion.
        sigma: Decay rate beyond epsilon.
        """
        self.A = A
        self.epsilon = epsilon
        self.sigma = sigma
        
    def forward(self, dists: torch.Tensor) -> torch.Tensor:
        # Gate: smooth decay beyond epsilon.
        # V = A * max(0, epsilon - d)² / epsilon² for soft gating
        # Or: V = A * exp(-(d - epsilon)²/sigma²) for d > epsilon
        
        # Use smooth gating factor: gate = sigmoid((epsilon - d) / sigma)
        gate = torch.sigmoid((self.epsilon - dists) / (self.sigma + 1e-9))
        
        # Base repulsion: log(cosh(d/sigma)) style but gated
        base_rep = torch.log(torch.cosh(dists / self.sigma))
        
        return self.A * gate * base_rep
        
    def derivative(self, dists: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid((self.epsilon - dists) / (self.sigma + 1e-9))
        
        # d(gate)/dd = -gate*(1-gate)/sigma
        gate_grad = -gate * (1 - gate) / (self.sigma + 1e-9)
        
        base_rep = torch.log(torch.cosh(dists / self.sigma))
        base_rep_grad = torch.tanh(dists / self.sigma) / self.sigma
        
        # Product rule: d(gate*base)/dd = gate'*base + gate*base'
        return self.A * (gate_grad * base_rep + gate * base_rep_grad)


class CompositePotentialTorch:
    def __init__(self, potentials: list):
        self.potentials = potentials
        
    def potential_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_e = 0.0
        total_grad = torch.zeros_like(x)
        
        for p in self.potentials:
            e, g = p.potential_and_grad(x)
            total_e += e
            total_grad += g
            
        return total_e, total_grad


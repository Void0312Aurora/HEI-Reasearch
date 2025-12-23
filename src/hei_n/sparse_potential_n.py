"""
Sparse Potentials for Scalable HEI-N.
=====================================

Implements O(N) interaction potentials using graph edges and negative sampling.
Computes sparse distances and gradients without forming the full O(N^2) matrix.
"""

import numpy as np
from typing import Callable, List, Tuple
from .geometry_n import minkowski_inner

class SparseEdgePotential:
    """
    Attraction between defined edges (e.g. from semantic graph).
    List of adjacency pairs (u, v).
    Complexity: O(E), where E is number of edges.
    """
    def __init__(self, edges: np.ndarray, 
                 kernel_fn: Callable[[np.ndarray], np.ndarray], 
                 d_kernel_fn: Callable[[np.ndarray], np.ndarray]):
        """
        edges: (E, 2) integer array of indices.
        """
        self.edges = np.asarray(edges, dtype=int)
        self.kernel_fn = kernel_fn
        self.d_kernel_fn = d_kernel_fn
        
    def _compute_dists(self, x: np.ndarray):
        # x: (N, dim+1)
        # Gather node positions
        idx_u = self.edges[:, 0]
        idx_v = self.edges[:, 1]
        
        xu = x[idx_u] # (E, dim+1)
        xv = x[idx_v] # (E, dim+1)
        
        # Inner product
        inner = np.sum(xu * xv * np.array([-1] + [1]*(xu.shape[1]-1)), axis=-1)
        # Or use minkowski_inner helper if adapted for matching shapes
        # minkowski_inner(xu, xv) -> (E,)
        
        # d = arccosh(-inner)
        val = np.maximum(-inner, 1.0 + 1e-7)
        return np.arccosh(val), idx_u, idx_v, val
        
    def potential(self, x: np.ndarray) -> float:
        dists, _, _, _ = self._compute_dists(x)
        return float(np.sum(self.kernel_fn(dists)))
        
    def gradient(self, x: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        dim = x.shape[1]
        grad = np.zeros_like(x)
        
        dists, idx_u, idx_v, val = self._compute_dists(x) # val is -<u,v>
        
        # Overflow safe sqrt using asymptotic
        large_val = (val > 1e100)
        denom = np.zeros_like(val)
        denom[large_val] = val[large_val]
        denom[~large_val] = np.sqrt(val[~large_val]**2 - 1.0)
        
        denom = np.maximum(denom, 1e-7)
        dv = self.d_kernel_fn(dists)
        
        S = dv / denom # Scalar factor for each edge
        
        # Force on u due to v:
        # grad_xu d_uv = (1/sinh) * (-J xv)
        # grad_linear = S * (-J xv)
        # We add this to grad[u]
        
        xu = x[idx_u]
        xv = x[idx_v]
        
        J = np.ones(dim); J[0] = -1.0
        
        Jxv = xv * J[np.newaxis, :]
        Jxu = xu * J[np.newaxis, :]
        
        # Force on u
        Fu = -S[:, np.newaxis] * Jxv
        # Force on v (Symmetric logic, d_uv = d_vu)
        # grad_xv d_uv = (1/sinh) * (-J xu)
        Fv = -S[:, np.newaxis] * Jxu
        
        # Accumulate
        np.add.at(grad, idx_u, Fu)
        np.add.at(grad, idx_v, Fv)
        
        return grad

class NegativeSamplingPotential:
    """
    Stochastic Repulsion using Negative Sampling.
    For each particle i, sample k random particles j.
    Calculates forces for these pairs.
    Complexity: O(N * k).
    """
    def __init__(self, kernel_fn: Callable[[np.ndarray], np.ndarray], 
                 d_kernel_fn: Callable[[np.ndarray], np.ndarray],
                 num_neg: int = 5,
                 rescale: float = 1.0,
                 seed: int = None):
        self.kernel_fn = kernel_fn
        self.d_kernel_fn = d_kernel_fn
        self.num_neg = num_neg
        self.rescale = rescale
        self.rng = np.random.default_rng(seed)
        
    def _sample_pairs(self, N):
        """Generates (N * k, 2) pairs."""
        # For each i in 0..N-1, pick num_neg indices from 0..N-1
        # Simplest: Uniform random.
        # Self-loops will happen, should be masked or dist=0 handled.
        
        sources = np.repeat(np.arange(N), self.num_neg)
        targets = self.rng.integers(0, N, size=N * self.num_neg)
        
        # Mask self-loops
        mask = (sources != targets)
        return sources[mask], targets[mask]
        
    def potential(self, x: np.ndarray) -> float:
        # NOTE: Potential energy is stochastic here.
        # It fluctuates every step. This flucuation is related to "Heat".
        N = x.shape[0]
        u, v = self._sample_pairs(N)
        
        xu = x[u] # (M, dim)
        xv = x[v]
        
        # Compute distances
        inner = np.sum(xu * xv * np.array([-1] + [1]*(xu.shape[1]-1)), axis=-1)
        val = np.maximum(-inner, 1.0 + 1e-7)
        dists = np.arccosh(val)
        
        energy = np.sum(self.kernel_fn(dists))
        return float(energy * self.rescale)
        
    def gradient(self, x: np.ndarray) -> np.ndarray:
        N, dim = x.shape
        grad = np.zeros_like(x)
        
        idx_u, idx_v = self._sample_pairs(N)
        
        xu = x[idx_u]
        xv = x[idx_v]
        
        inner = np.sum(xu * xv * np.array([-1] + [1]*(dim-1)), axis=-1)
        val = np.maximum(-inner, 1.0 + 1e-7)
        dists = np.arccosh(val)
        
        # Overflow safe sqrt using asymptotic
        # if val > 1e150, val**2 overflows float64.
        # sqrt(val**2 - 1) approx val.
        large_val = (val > 1e100)
        denom = np.zeros_like(val)
        
        denom[large_val] = val[large_val]
        denom[~large_val] = np.sqrt(val[~large_val]**2 - 1.0)
        
        denom = np.maximum(denom, 1e-7)
        dv = self.d_kernel_fn(dists)
        
        S = (dv / denom) * self.rescale
        
        J = np.ones(dim); J[0] = -1.0
        Jxv = xv * J[np.newaxis, :]
        Jxu = xu * J[np.newaxis, :]
        
        # Force on u
        Fu = -S[:, np.newaxis] * Jxv
        # Force on v (Action-Reaction! Even though j was sampled as context)
        # If we pushed u away from v, we MUST push v away from u to conserve momentum.
        Fv = -S[:, np.newaxis] * Jxu
        
        np.add.at(grad, idx_u, Fu)
        np.add.at(grad, idx_v, Fv)
        
        return grad

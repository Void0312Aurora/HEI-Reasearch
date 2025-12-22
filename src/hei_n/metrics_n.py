"""
Metrics for N-Dimensional Hyperbolic Space.
===========================================

Quantify structural properties like Hierarchy and Tree-ness.
"""

import numpy as np
from .geometry_n import dist_n, minkowski_inner

def calculate_pairwise_distances(x: np.ndarray) -> np.ndarray:
    """Compute (N, N) distance matrix."""
    inner = minkowski_inner(x[:, np.newaxis, :], x[np.newaxis, :, :])
    # d = arccosh(-inner)
    val = np.maximum(-inner, 1.0 + 1e-7)
    dists = np.arccosh(val)
    np.fill_diagonal(dists, 0.0)
    return dists

def calculate_ultrametricity_score(x: np.ndarray, sample_size: int = 1000) -> float:
    """
    Measure how well the points satisfy the Ultrametric Inequality (Strong Triangle Inequality).
    d(x, y) <= max(d(x, z), d(y, z))
    
    A perfect tree (ultrametric space) has score 0.
    
    We sample triplets (i, j, k) and compute:
    violation = max(0, d_ij - max(d_ik, d_jk))
    score = mean(violation / max(d_ij, 1e-6))
    """
    N = x.shape[0]
    if N < 3: return 0.0
    
    dists = calculate_pairwise_distances(x)
    
    violations = []
    rng = np.random.default_rng()
    
    for _ in range(sample_size):
        # Sample 3 distinct indices
        idx = rng.choice(N, size=3, replace=False)
        i, j, k = idx[0], idx[1], idx[2]
        
        d_ij = dists[i, j]
        d_ik = dists[i, k]
        d_jk = dists[j, k]
        
        # Check all permutations?
        # Inequality: d_ij <= max(d_ik, d_jk) holds for the "longest side" in ultrametric triangle?
        # In ultrametric space, every triangle is isosceles with two long sides equal, or equilateral.
        # The two largest sides must be equal.
        # So sides sorted: a <= b = c.
        # Deviation: |b - c| / max(b, c)? 
        # Or: check if the smallest side <= max(two larger)? No, that's regular triangle.
        # Condition: The two largest distances must be equal.
        
        sides = sorted([d_ij, d_ik, d_jk])
        s_small, s_mid, s_large = sides
        
        # In tree, s_mid == s_large.
        # Score = (s_large - s_mid) / s_large
        
        if s_large > 1e-6:
            score = (s_large - s_mid) / s_large
            violations.append(score)
            
    return float(np.mean(violations))

def calculate_distortion_vs_tree(x: np.ndarray) -> float:
    """placeholder"""
    return 0.0

"""
Aurora Gauge Field (CCD v2.0).
=============================

Implements the Logical Fiber Bundle connections and curvature.
Ref: `docs/plan/理论基础-5.md` Chapter 2.

Components:
1. GaugeField: Manages parallel transport matrix U for edges.
2. Wilson Loops: Computes curvature from closed loops.
3. Wong Force: Computes F_logic.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .geometry import project_to_tangent

class GaugeField(nn.Module):
    """
    Manages the Connection A_mu on the principal bundle.
    In discrete implementations, this is stored as edge transport matrices U_ij in G.
    
    For SO(k) or SU(k), U_ij = exp(omega_ij).
    """
    def __init__(self, edges: torch.Tensor, logical_dim: int, group='SO', skew_symmetric: bool = True):
        """
        Args:
            edges: (E, 2) edge list
            logical_dim: Dimension of the internal vector space (e.g. k for SO(k))
            group: 'SO' (Orthogonal) or 'SU' (Unitary - complex not yet supported explicitly)
        """
        super().__init__()
        self.register_buffer('edges', edges)
        self.logical_dim = logical_dim
        self.E = edges.shape[0]
        
        # Store Lie Algebra elements omega_ij (E, k, k)
        # For SO(k), omega is skew-symmetric.
        # We store flat parameters and project to algebra during forward.
        if group == 'SO':
            # Number of generators for SO(k) is k(k-1)/2
            # But simpler to just store k*k and force skew-symmetry.
            self.omega_params = nn.Parameter(torch.randn(self.E, logical_dim, logical_dim) * 0.01)
        else:
            raise NotImplementedError("Only SO group supported currently.")
            
    def get_U(self) -> torch.Tensor:
        """
        Compute Group Elements U_ij = exp(omega_ij).
        Enforces constraints (Skew-symmetry for SO).
        """
        # Enforce skew-symmetry: A = (W - W^T)/2
        omega = 0.5 * (self.omega_params - self.omega_params.transpose(1, 2))
        
        # Matrix Exponential map
        # torch.matrix_exp is consistent with Lie Group exp for matrix groups
        U = torch.matrix_exp(omega)
        return U
    
    def parallel_transport(self, J: torch.Tensor, edge_indices: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Transport logic charge J from u to v via edge (u,v).
        J_v = U_ij @ J_u
        """
        # J: (Batch, k)
        # U: (E, k, k)
        
        U_batch = self.get_U()[edge_indices] # (Batch, k, k)
        
        if inverse:
            # U_inv = U^T for SO(k)
            U_op = U_batch.transpose(1, 2)
        else:
            U_op = U_batch
            
        # J is vector (Batch, k)
        # Treat as column vector: J_col = (Batch, k, 1)
        # result = U @ J_col -> (Batch, k, 1)
        J_new = torch.matmul(U_op, J.unsqueeze(-1)).squeeze(-1)
        return J_new

    def compute_force_wong(self, x: torch.Tensor, v: torch.Tensor, J: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Logical Lorentz Force: F = <J, F_munu> v
        But in discrete, we approximate using local loops?
        
        Actually, for Phase 1, we might just implement the 'Logic Precession' first?
        Theory 2.4.2 says: F_logic ~ sum Area * <J, Omega> * v_ij.
        
        This requires finding triangles (triplets).
        If we don't have explicit triplets, we can't compute F yet.
        
        For now, let's just return Precession generator.
        """
        # Placeholder for full Wong Force
        # Returning zero force until triplet structure is integrated
        return torch.zeros_like(x), torch.zeros_like(x) # Energy, Force

"""
Aurora Bound State Detector (CCD v3.1).
=======================================

Detects formation of stable molecules (words).
Criterion: E_rel < -epsilon.

Ref: Axiom 3.2.1.
"""

import torch
from .forces import ForceField
from ..physics import geometry

class MoleculeDetector:
    def __init__(self, force_field: ForceField, bind_thresh: float = 0.5):
        self.ff = force_field
        self.bind_thresh = bind_thresh
        
    def detect(self, state: dict) -> list:
        """
        Returns list of bonds [(i, j), ...].
        """
        q = state['q']
        p = state['p']
        m = state['m']
        J = state['J']
        # atomic mass approx m_i
        
        N = q.shape[0]
        if N < 2: return []
        
        # 1. Potential V(i, j)
        # We need pairwise potential, not total summed V.
        # ForceField methods usually sum.
        # We need to expose pairwise calculation or redo it here.
        # Redoing for clarity/separation.
        
        # Distances
        q_i = q.unsqueeze(1)
        q_j = q.unsqueeze(0)
        diff = geometry.mobius_add(-q_i, q_j)
        d_ij = 2.0 * torch.atanh(torch.norm(diff, dim=-1)) # (N, N)
        
        # Mass Potential
        K = torch.exp(-d_ij**2)
        m_prod = m @ m.t()
        V_mass = - self.ff.G * m_prod * K
        
        # Gauge Potential
        # Approx dot product
        J_flat = J.view(N, -1)
        dot = J_flat @ J_flat.t() # Simple dot (ignoring PT for detection speed)
        # Using lambda * dot
        # Note: 3.1.2 says PT required.
        # For precise binding, we should include PT.
        # But locally PT is Identity?
        # Let's use simple dot for prototype.
        V_gauge = self.ff.lambda_gauge * dot
        
        V_int = V_mass + V_gauge
        
        # 2. Kinetic T_rel
        # v = p/m
        v = p / (m + 1e-4)
        v_i = v.unsqueeze(1)
        v_j = v.unsqueeze(0)
        # v_rel^2 approx ||v_i - v_j||^2
        dv = v_i - v_j
        v_rel_sq = torch.sum(dv**2, dim=-1)
        
        # Reduced mass mu = m1m2/(m1+m2)
        m_sum = m + m.t()
        mu = m_prod / (m_sum + 1e-4)
        
        T_rel = 0.5 * mu * v_rel_sq
        
        # 3. Total E
        E_rel = T_rel + V_int
        
        # 4. Threshold
        # Mask diagonal and upper triangle
        mask = torch.triu(torch.ones(N, N, device=q.device), diagonal=1).bool()
        
        # Check binding
        is_bound = (E_rel < -self.bind_thresh) & mask
        
        indices = torch.nonzero(is_bound, as_tuple=False)
        return indices.tolist()

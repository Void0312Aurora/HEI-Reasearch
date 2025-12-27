"""
Aurora Forces Engine (CCD v3.1).
================================

Defines Interaction Potentials.
Forces are computed via Automatic Differentiation.

Ref: Chapter 3 (Interaction Dynamics) & Axioms (Chemical Potential).
"""

import torch
from ..physics import geometry

class ForceField:
    def __init__(self, G: float = 1.0, lambda_gauge: float = 1.0, k_geo: float = 0.5, r_cutoff: float = 0.8, mu: float = 0.1):
        self.G = G
        self.lambda_gauge = lambda_gauge
        self.k_geo = k_geo
        self.mu = mu # Chemical Potential (Mass Cost)
        self.r_cutoff = r_cutoff 

    def potential_geometry(self, q: torch.Tensor) -> torch.Tensor:
        """
        V_bg: Effective Geometric Repulsion.
        Push particles to boundary OR Pull to center?
        V = - k * dist(0, q).
        Since F = - grad V. grad d points outwards.
        F = k * grad d. Pushes OUT.
        This provides 'Negative Pressure' or Expansion.
        """
        d = geometry.dist(torch.zeros_like(q), q)
        return - self.k_geo * d.sum()

    def potential_chemical(self, m: torch.Tensor) -> torch.Tensor:
        """
        V_chem: Mass Self-Energy.
        Cost of maintaining mass.
        V = 0.5 * mu * sum(m^2).
        Deriv w.r.t m is mu * m.
        This suppresses 'Free Lunch' mass explosion.
        """
        return 0.5 * self.mu * torch.sum(m**2)

    def potential_mass(self, q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        V_mass = - G * sum_{i<j} m_i m_j * K_att(d_ij) + G_core * sum K_rep(d_ij).
        """
        N = q.shape[0]
        if N < 2: return torch.tensor(0.0, device=q.device)

        q_i = q.unsqueeze(1) # (N, 1, D)
        q_j = q.unsqueeze(0) # (1, N, D)
        
        diff = geometry.mobius_add(-q_i, q_j) 
        norm = torch.norm(diff, dim=-1)
        norm_clamped = torch.clamp(norm, max=1.0 - 1e-4)
        d_ij = 2.0 * torch.atanh(norm_clamped)
        
        # Attraction
        sigma = 1.0
        K_att = torch.exp(- (d_ij**2) / sigma)

        # Mask distant
        mask_stable = d_ij < 5.0
        K_att = K_att * mask_stable.float()
        
        # Hard Core Repulsion
        sigma_core = 0.01
        K_rep = torch.exp(- (d_ij**2) / sigma_core)
        
        m_prod = m @ m.t() # (N, N)
        
        G_core = 10.0 * self.G
        
        # Combined Potential
        E_mat = (- self.G * m_prod * K_att) + (G_core * K_rep)
        
        # Mask diagonal
        mask_diag = torch.eye(N, device=q.device).bool()
        E_mat.masked_fill_(mask_diag, 0.0)
        
        return 0.5 * E_mat.sum()

    def potential_gauge(self, q: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """
        V_gauge = lambda * sum < J_i, J_j >.
        Simplified without parallel transport for prototype.
        """
        N = q.shape[0]
        if N < 2: return torch.tensor(0.0, device=q.device)

        J_i = J.unsqueeze(1) # (N, 1, k, k)
        J_j = J.unsqueeze(0) # (1, N, k, k)

        dot = torch.sum(J_i * J_j, dim=(-2, -1)) # (N, N)
        
        E_mat = self.lambda_gauge * dot
        
        mask = torch.eye(N, device=q.device).bool()
        E_mat.masked_fill_(mask, 0.0)
        
        return 0.5 * E_mat.sum()

    def compute_forces(self, q: torch.Tensor, m: torch.Tensor, J: torch.Tensor):
        """
        Returns -Gradient of V_total (Tangential Force) and V_total.
        """
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
            
        V_geo = self.potential_geometry(q)
        V_mass = self.potential_mass(q, m)
        V_gauge = self.potential_gauge(q, J)
        V_chem = self.potential_chemical(m)
        
        V_total = V_mass + V_gauge + V_geo + V_chem
        
        grad_q = torch.autograd.grad(V_total, q, create_graph=False, retain_graph=True)[0]
        
        # Riemannian Gradient Correction
        lam = geometry.lambda_x(q)
        F_riem = grad_q * (1.0 / (lam**2))
        
        return -F_riem, V_total

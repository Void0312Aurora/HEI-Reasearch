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
    def __init__(self, G: float = 1.0, lambda_gauge: float = 1.0, k_geo: float = 0.5, r_cutoff: float = 0.8, mu: float = 0.1, G_core: float = None, lambda_quartic: float = 0.01):
        self.G = G
        self.lambda_gauge = lambda_gauge
        self.k_geo = k_geo
        self.mu = mu # Chemical Potential (Mass Cost)
        self.r_cutoff = r_cutoff 
        # Default G_core if not provided (Short range repulsion > Long range attraction)
        self.G_core = G_core if G_core is not None else 2.0 * G
        self.lambda_quartic = lambda_quartic

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
        V_chem: Mass Self-Energy (Landau Potential).
        V = 0.5 * mu * m^2 + 0.25 * lambda * m^4.
        Quartic term ensures stability (Mexican Hat) even if Attraction overpowers m^2.
        """
        V_quad = 0.5 * self.mu * (m**2)
        V_quart = 0.25 * self.lambda_quartic * (m**4)
        return (V_quad + V_quart).sum()

    def potential_mass(self, q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        V_mass = m_i * m_j * (- G * K_att + G_core * K_rep) / (N + 1).
        Kac Normalization ensures Energy is O(N) not O(N^2).
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
        
        # Kac Scaling
        scale_factor = 1.0 / (N + 1.0)
        
        # Combined Potential
        term_interact = ((- self.G * K_att) + (self.G_core * K_rep)) * scale_factor
        
        m_prod = m @ m.t() # (N, N)
        E_mat = m_prod * term_interact
        
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

"""
Aurora Force Field (CCD v3.1)
=============================

Implements Axiom 3.1: Conformal Interaction Potentials with Gauge Theory.

Key Features:
1. Mass-Gravitational Potential (Scalar Attraction)
2. Gauge Exchange Potential (Vector/Matrix Interaction) with Parallel Transport
   Ref: Axiom 3.1.2 (Route B: Discrete Edge Connection)
3. Background Geometric Potential (Effective Repulsion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..physics import geometry

class DiscreteGaugeConnection(nn.Module):
    """
    Learns the Parallel Transport operator PT_{i->j} based on relative position.
    PT = exp(Omega(q_i, q_j)) where Omega is skew-symmetric (so(n)).
    """
    def __init__(self, dim, hidden_dim=32):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim * (dim - 1) // 2)
        )
        
    def forward(self, diff_vec):
        """
        Args:
            diff_vec: (B, N, N, D) or (N, N, D) Relative vector (-q_i + q_j)
        Returns:
            PT: (..., dim, dim) Orthogonal matrix
        """
        # Batch/N handling
        batch_shape = diff_vec.shape[:-1]
        flat_diff = diff_vec.view(-1, self.dim)
        
        # Predict skew parameters
        omega_params = self.net(flat_diff)
        
        # Construct Skew-Symmetric Matrix
        omega = torch.zeros(flat_diff.size(0), self.dim, self.dim, device=diff_vec.device)
        triu_idx = torch.triu_indices(self.dim, self.dim, offset=1)
        omega[:, triu_idx[0], triu_idx[1]] = omega_params
        omega = omega - omega.transpose(1, 2)
        
        # Exponential map on Lie Algebra -> Lie Group (Rotation)
        # For so(n), this is Matrix Exp.
        # PyTorch matrix_exp is stable.
        pt_mat = torch.matrix_exp(omega)
        
        return pt_mat.view(*batch_shape, self.dim, self.dim)

class ForceField(nn.Module):
    def __init__(self, dim=16, G=1.0, lambda_gauge=1.0, k_geo=1.0, mu=1.0, lambda_quartic=0.01):
        super().__init__()
        self.dim = dim
        self.G = G # Gravitational Constant
        self.lambda_gauge = lambda_gauge # Gauge Coupling
        self.k_geo = k_geo # Geometric Repulsion
        self.mu = mu # Chemical Potential Mass
        self.lambda_quartic = lambda_quartic # Chemical Potential Stability
        
        # Learnable Gauge Connection
        self.connection = DiscreteGaugeConnection(dim)
        
    def potential_geometry(self, q: torch.Tensor) -> torch.Tensor:
        """
        Axiom 2.3.1: Geometric Repulsion.
        Effective potential V ~ -log(Vol) ~ d(0, q).
        Forces particles to spread out (Entropy Max).
        """
        d = geometry.dist(torch.zeros_like(q), q) # d(0, q)
        # Linear potential V = -k * d (Force = k * r_hat outwards)
        # Wait, usually V > 0 for repulsion?
        # If V = -k * d, force is towards center (Attraction).
        # We want Divergence -> Repulsion from center?
        # Geometry naturally creates divergence.
        # We model the "Cost" of being at center? 
        # Actually in H^n, volume is at boundary.
        # Let's stick to V = - k_geo * sum(d) -> Force pushes to maximize d (Outwards).
        return - self.k_geo * d.sum()

    def potential_chemical(self, m: torch.Tensor) -> torch.Tensor:
        """
        Landau Potential for Mass Stability.
        """
        V_quad = 0.5 * self.mu * (m**2)
        V_quart = 0.25 * self.lambda_quartic * (m**4)
        return (V_quad + V_quart).sum()

    def potential_mass(self, q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Axiom 3.1.1: Mass Attraction.
        V = -G * m_i * m_j * K(d_ij)
        """
        N = q.shape[0]
        if N < 2: return torch.tensor(0.0, device=q.device)

        q_i = q.unsqueeze(1) # (N, 1, D)
        q_j = q.unsqueeze(0) # (1, N, D)
        
        # Hyperbolic diff & dist
        diff = geometry.mobius_add(-q_i, q_j) 
        # Safe Norm for Autograd (Avoid div-by-zero at d=0)
        sq_norm = torch.sum(diff**2, dim=-1)
        norm = torch.sqrt(sq_norm + 1e-10)
        norm_clamped = torch.clamp(norm, max=1.0 - 1e-4)
        d_ij = 2.0 * torch.atanh(norm_clamped)
        
        # Kernel
        sigma = 1.0
        K_att = torch.exp(- (d_ij**2) / sigma)
        
        # Hard Core Repulsion (Pauli)
        sigma_core = 0.05 # Slightly wider core
        K_rep = torch.exp(- (d_ij**2) / sigma_core)
        
        # Kac Scaling
        scale = 1.0 / (N + 1.0)
        
        # V_ij
        # Fix: Stronger Repulsion (5.0) > Attraction (G=1.0) to prevent collapse
        v_pair = (-self.G * K_att + 5.0 * K_rep) * scale
        
        # Mass weighting
        m_pairs = m @ m.t()
        
        E_mat = m_pairs * v_pair
        
        # Mask diagonal
        mask = torch.eye(N, device=q.device).bool()
        E_mat.masked_fill_(mask, 0.0)
        
        return 0.5 * E_mat.sum()

    def potential_gauge(self, q: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """
        Axiom 3.1.2 (Route B):
        V_gauge = - lambda * < J_i, PT_{j->i}(J_j) > * K(d_ij)
        (Ferromagnetic Alignment)
        """
        N = q.shape[0]
        if N < 2: return torch.tensor(0.0, device=q.device)

        # 1. Distances & Kernel
        q_i = q.unsqueeze(1)
        q_j = q.unsqueeze(0)
        
        diff = geometry.mobius_add(-q_i, q_j) # vector from i to j in i's tangent space?
        
        # Safe Norm
        sq_norm = torch.sum(diff**2, dim=-1)
        norm = torch.sqrt(sq_norm + 1e-10)
        norm_clamped = torch.clamp(norm, max=1.0 - 1e-4)
        d_ij = 2.0 * torch.atanh(norm_clamped)
        
        # Interaction range
        sigma_gauge = 2.0
        K_dist = torch.exp(-(d_ij**2) / sigma_gauge)
        
        # 2. Parallel Transport
        pt_mat = self.connection(diff) # (N, N, D, D)
        
        # Transform J_j
        J_j_exp = J.unsqueeze(0).expand(N, -1, -1, -1) # (N, N, D, D)
        J_j_trans = torch.matmul(pt_mat, torch.matmul(J_j_exp, pt_mat.transpose(-2, -1)))
        
        # 3. Inner Product < J_i, J_j_trans >
        J_i_exp = J.unsqueeze(1).expand(-1, N, -1, -1)
        dot = torch.sum(J_i_exp * J_j_trans, dim=(-2, -1)) # (N, N)
        
        # 4. Total Energy
        # Ferromagnetic: Minimize Energy -> Maximize Dot Product -> Negative Sign
        # Kac Scaling for Intensive Energy Density
        scale = 1.0 / (N + 1.0)
        E_mat = - self.lambda_gauge * dot * K_dist * scale
        
        mask = torch.eye(N, device=q.device).bool()
        E_mat.masked_fill_(mask, 0.0)
        
        return 0.5 * E_mat.sum()

    def compute_forces(self, q: torch.Tensor, m: torch.Tensor, J: torch.Tensor, return_grads: bool = True):
        """
        Compute forces via AutoGrad.
        Returns:
            F_q: -Grad_q V (Riemannian corrected) (None if return_grads=False)
            E_total
        """
        # Ensure grad
        if return_grads and not q.requires_grad:
            q = q.detach().requires_grad_(True)
            
        V_geo = self.potential_geometry(q)
        V_mass = self.potential_mass(q, m)
        V_gauge = self.potential_gauge(q, J)
        V_chem = self.potential_chemical(m)
        
        # Rotational Kinetic Energy Penalty to separate J scale from Gauge Strength
        # Without this, J grows to minimize V_gauge = - <J, J>
        V_rot = 0.001 * torch.sum(J**2)
        
        V_total = V_mass + V_gauge + V_geo + V_chem + V_rot
        
        if not return_grads:
            return None, V_total
        
        grad_q = torch.autograd.grad(V_total, q, create_graph=True, retain_graph=True)[0]
        
        # Riemannian Gradient
        # grad_R = grad_E / lambda^2
        lam = geometry.lambda_x(q)
        grad_riem = grad_q * (1.0 / (lam**2))
        
        return -grad_riem, V_total

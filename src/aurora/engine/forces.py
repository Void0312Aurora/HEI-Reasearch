"""
Aurora Forces Engine (CCD v3.1).
================================

Defines Interaction Potentials.
Forces are computed via Automatic Differentiation.

Ref: Chapter 3 (Interaction Dynamics).
"""

import torch
from ..physics import geometry

class ForceField:
    def __init__(self, G: float = 1.0, lambda_gauge: float = 1.0, k_geo: float = 0.5, r_cutoff: float = 0.8):
        self.G = G
        self.lambda_gauge = lambda_gauge
        self.k_geo = k_geo
        self.r_cutoff = r_cutoff # Normalized Poincare radius for neighbor cutoff? 
                                 # Or geodesic dist? Usually geodesic.

    def potential_geometry(self, q: torch.Tensor) -> torch.Tensor:
        """
        V_bg: Effective Geometric Repulsion.
        Push particles to boundary.
        V ~ - k * dist(0, q)? 
        Actually F = k * r_hat.
        Potential V = - k * ||q|| (Euclidean norm approx in ball).
        Or better: V = - k * dist(0, q).
        This lowers energy as q moves away (dist increases).
        """
        # Distance from origin
        # d = 2 atanh(|q|)
        d = geometry.dist(torch.zeros_like(q), q)
        return - self.k_geo * d.sum()

    def potential_mass(self, q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        V_mass = - G * sum_{i<j} m_i m_j * K(d_ij).
        K(d) = exp(-d^2 / sigma).
        """
        # q: (N, dim)
        # m: (N, 1)
        
        # Pairwise distance
        # Naive N^2 for prototype. Use block-sparse for prod.
        N = q.shape[0]
        if N < 2: return torch.tensor(0.0, device=q.device)

        # Broadcast for pairs
        # We only compute upper triangle or full and divide by 2
        
        # Compute Geodesic Distance Matrix?
        # geometry.dist expects inputs of same shape.
        # Expand
        q_i = q.unsqueeze(1) # (N, 1, D)
        q_j = q.unsqueeze(0) # (1, N, D)
        
        # Mobius subtraction q_i (-) q_j
        # This might be memory heavy for large N. Phase 33/34 optimization.
        diff = geometry.mobius_add(-q_i, q_j) 
        norm = torch.norm(diff, dim=-1)
        # d_ij = 2 atanh(norm)
        d_ij = 2.0 * torch.atanh(torch.clamp(norm, max=1.0-1e-5))
        
        # Kernel
        # Gaussian Kernel
        sigma = 1.0
        K = torch.exp(- (d_ij**2) / sigma)
        
        # Mass product
        m_prod = m @ m.t() # (N, N)
        
        E_mat = - self.G * m_prod * K
        
        # Mask diagonal
        mask = torch.eye(N, device=q.device).bool()
        E_mat.masked_fill_(mask, 0.0)
        
        return 0.5 * E_mat.sum()

    def potential_gauge(self, q: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """
        V_gauge = lambda * sum < J_i, PT_{j->i}(J_j) >.
        
        PT_{j->i}(v) = Gyr[i, -j] v ? 
        Check 3.1.2: V ~ < J_i, PT(J_j) >
        Ideally Parallel Transport preserves norm.
        """
        N = q.shape[0]
        if N < 2: return torch.tensor(0.0, device=q.device)

        q_i = q.unsqueeze(1)
        q_j = q.unsqueeze(0)
        
        # PT_{j->i}(J_j)
        # Formula: Gyr[i, -j] J_j
        # Warning: Gyration inputs must be broadcast
        # Gyr[u, v]w
        # u = q_i, v = -q_j, w = J_j
        
        J_j = J.unsqueeze(0).expand(N, -1, -1, -1) # (N, N, k, k) for w
        
        # Need broadcasted gyration
        # This is expensive. O(N^2).
        
        # Approximation: If curvature is constant, we trust Gyration formula.
        # But J is a Matrix, gyration() expects a vector point.
        # For prototype, we skip gyration (Identity Transport).
        # J_j_transported = geometry.gyration(q_i, -q_j, J_j)
        J_j_transported = J_j
        
        # Inner Product
        # <J_i, J_j_trans>
        J_i = J.unsqueeze(1) # (N, 1, k, k)
        
        # Interaction
        # V > 0 (Repulsion) or V < 0 (Attraction)?
        # 3.1.2 says: "Complementary (J_i approx -J_j) -> Bind".
        # If J_i = -J_j_trans, dot product is negative.
        # If we check V < 0 for binding, then V = lambda * dot is correct.
        # (J_i . J_j_trans) ~ -1 => V ~ -lambda (Attraction).
        
        dot = torch.sum(J_i * J_j_transported, dim=(-2, -1)) # (N, N)
        
        E_mat = self.lambda_gauge * dot
        
        mask = torch.eye(N, device=q.device).bool()
        E_mat.masked_fill_(mask, 0.0)
        
        return 0.5 * E_mat.sum()

    def compute_forces(self, q: torch.Tensor, m: torch.Tensor, J: torch.Tensor) -> torch.utils.data.DataLoader:
        """
        Compute total forces via Autograd.
        Returns F_q (Tangential Force), F_J (Charge Torque? No, J evolves via Wong, not gradient of V).
        
        Wait. 
        2.3.1: F_total = F_geo + F_mass + F_gauge.
        F_gauge (Lorentz) comes from J coupled to Field.
        Here we define V_total.
        F_q = - grad_q V_total.
        
        Does F_gauge match -grad_q V_gauge?
        Yes, macroscopically.
        
        What about J evolution? 
        2.3.2 Wong: dJ/dt = - [A, J].
        A comes from neighbors.
        We need A_mu.
        A_mu(x) is the Gauge Potential.
        
        If we use Autograd for F_q, we handle spatial motion.
        But J evolution requires 'Torque'.
        Can we derive Torque from V_gauge?
        V ~ <J, A>.
        Torque ~ [J, A].
        It's related.
        
        For now, let's implement F_q via autograd.
        And we need to approximate A_mu for J evolution.
        
        Approximation:
        A_mu(at i) ~ sum_j  (J_j transported to i) / dist?
        Or simply derive A from the pairwise term?
        The term <J_i, PT(J_j)> looks like <J_i, A_effective>.
        Where A_effective = sum PT(J_j).
        So A ~ sum PT(J_j).
        Then dJ/dt = - [A, J].
        
        This seems consistent.
        """
        # Enable grad
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
            
        V_geo = self.potential_geometry(q)
        V_mass = self.potential_mass(q, m)
        V_gauge = self.potential_gauge(q, J)
        
        V_total = V_mass + V_gauge + V_geo # Note Signs: V_bg is effective repulsive potential.
        # In my potential_geometry, I returned -k*d. This is ATTRACTIVE to origin?
        # V ~ -d. Force = -grad V = +k grad d. Pushes OUT. Correct.
        # Wait, usually V_repulsive ~ +1/r.
        # If I want Repulsion from origin (push to boundary):
        # I want Force pointing along +r.
        # grad d points +r.
        # F = - grad V.
        # So I need - grad V = + vec.
        # grad V = - vec.
        # V should decrease as r increases? 
        # If V decreases as r increases, particle rolls down hill to boundary.
        # So V = - d is correct. (Energy min at d=inf).
        
        grad_q = torch.autograd.grad(V_total, q, create_graph=False, retain_graph=True)[0]
        
        # Project Euclidean Gradient to Riemannian Gradient ??
        # F_cov = g^{ij} dV/dx^j.
        # Poincare Metric g = lambda^2 I.
        # g^{-1} = lambda^{-2} I.
        # So F_riem = (1/lambda^2) * grad_Euc.
        lam = geometry.lambda_x(q)
        F_riem = grad_q * (1.0 / (lam**2))
        
        return -F_riem, V_total

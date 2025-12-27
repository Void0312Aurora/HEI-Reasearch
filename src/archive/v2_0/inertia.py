"""
Aurora Inertia: Variable Mass Tensor.
=====================================
"""
import torch
from typing import Optional

class RadialInertia:
    """
    Inertia scales with hyperbolic radius.
    Mass m(r) = 1 + alpha * (cosh(r) - 1).
    This creates a 'Centrifugal Force' (Geometric Force) that naturally resists collapse.
    """
    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha
        
    def _get_mass(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar mass for each particle.
        x: (N, dim) Position in Ambient Space (Minkowski).
        x[:, 0] is cosh(r).
        """
        # x0 >= 1.0 theoretically. Clamp for safety.
        x0 = x[:, 0]
        m = 1.0 + self.alpha * torch.clamp(x0 - 1.0, min=0.0)
        return m

    def inverse(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity xi = I^{-1}(M).
        xi = M / m(x)
        """
        m = self._get_mass(x)
        # Broadcast m to (N, 1, 1) for M (N, dim, dim)
        return M / (m.view(-1, 1, 1) + 1e-9)
    
    def kinetic_energy(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Kinetic Energy T = 0.5 * <xi, M> = 0.5 * |M|^2 / m(x).
        Returns: (N,) energy per particle? Or Scalar total?
        Usually we need scalar for Hamiltonian, but gradients are per particle.
        Let's return total T for now (scalar), but logic usually sums internally.
        Wait, for step() we might need components. 
        Let's match the usage: usually just scalar or per-particle for debug.
        Return: (N,) tensor.
        """
        m = self._get_mass(x)
        
        # |M|^2 in Lie Algebra so(1,n).
        # We define |M|^2 = Sum of squares of components?
        # Or <M, M> = Tr(M^T M)?
        # For Boost part (symm): M_{0i} = M_{i0}. 
        # For So(1,n), standard metric is K_i^2 = 1?
        # Let's align with hei_n: T ~ sum(v^2) + 0.5 sum(w^2).
        # M is (N, dim, dim).
        # Boosts: M[:, 0, 1:]
        # Rotations: M[:, 1:, 1:]
        
        v = M[:, 0, 1:]
        w = M[:, 1:, 1:]
        
        # T_num = sum(v^2) + 0.5 * sum(w^2)
        T_num = torch.sum(v**2, dim=1) + 0.5 * torch.sum(w**2, dim=(1,2))
        
        return T_num / (m + 1e-9)

    def geometric_force(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Geometric Force F_geom = - grad_x T.
        T = 0.5 * |M|^2 / m(x).
        grad T = -0.5 * |M|^2 / m(x)^2 * grad m.
        
        m(x) = 1 + alpha * (x0 - 1).
        grad m = (alpha, 0, ...).
        
        F_geom = (0.5 * |M|^2 / m^2) * alpha * e_0.
        
        Vector in R^{n+1}.
        """
        m = self._get_mass(x)
        
        v = M[:, 0, 1:]
        w = M[:, 1:, 1:]
        M_sq = torch.sum(v**2, dim=1) + 0.5 * torch.sum(w**2, dim=(1,2))
        
        # Prefactor: (alpha * M_sq) / (2 * m^2)
        prefactor = (self.alpha * M_sq) / (2.0 * m**2 + 1e-9) # (N,)
        
        F = torch.zeros_like(x)
        F[:, 0] = prefactor
        
        # Note: This force is in ambient space R^{dim}.
        # It pushes in e0 direction (radially outward in projection).
        return F

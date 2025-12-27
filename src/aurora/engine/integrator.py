"""
Aurora Lie Group Integrator (CCD v3.1).
=======================================

Evolves system state on TQ x g.
q: Poincare Ball.
p: Tangent Vector (Levi-Civita PT).
J: Logic Charge (Gauge PT).

Ref: Axiom 2.4.1.
"""

import torch
from ..physics import geometry
from .forces import ForceField

class LieIntegrator:
    def __init__(self, force_field: ForceField):
        self.forces = force_field
        
    def step(self, state: dict, dt: float = 0.01) -> dict:
        """
        Perform one time step.
        State: {
            'q': (N, dim),
            'p': (N, dim),
            'J': (N, k, k),
            'm': (N, 1)
        }
        """
        q = state['q']
        p = state['p']
        J = state['J']
        m = state['m']
        
        # 1. Compute Forces at current step
        # F_q: Tangent force
        F_q, potential = self.forces.compute_forces(q, m, J)
        
        # 2. Update Momentum (Half step for Euler, or Full for Euler)
        # Simple Euler: p' = p + F * dt
        p_new_est = p + F_q * dt
        
        # 3. Update Position
        # q_next = Exp_q( dt * p )
        # Must account for metric? ExpMap handles metric scaling inside.
        # But 'p' is tangent vector. v = p? (Assuming mass=1 for kinematics, or v=p/m).
        # Axiom: "Inertia term... by mass m".
        # So v = p / m.
        # Handling mass:
        v = p_new_est / (m + 1e-4)
        q_next = geometry.exp_map(q, v * dt)
        
        # 4. Parallel Transport
        # q -> q_next
        
        # A. Momentum (Gyration)
        # P_{q->q_next}(p) = Gyr[q_next, -q] p
        p_transported = geometry.parallel_transport(q, q_next, p_new_est)
        
        # B. Charge (Gauge Transport)
        # J_spatial = P(J) (Frame dragging)
        J_spatial = state['J'] # Wait, J is attached to particle.
        # It lives in fiber. Fiber is attached to base.
        # If we move base, we must transport fiber frame.
        # Assuming trivial bundle topology locally -> J values change by Gyration if they are vector-like?
        # But J is in Lie Algebra. Is it a scalar (invariant) or vector in frame?
        # Axiom 0.4: J \in g. Matrix.
        # If it's in the tangent bundle (e.g. vector), it gyrates.
        # If it's internal (isospin), it only changes via Gauge Field A.
        # 3.1.2 says: "Logical charge J precession...".
        # Let's assume J rotates with frame (Gyration) AND precesses (Wong).
        # J acts on Tangent space vectors? Yes "Logic rotation".
        # So J transforms like Tensor. J_new = R J R^T.
        
        # Gyrator R
        # We need explicitly the rotation matrix of Gyration?
        # Or just apply it to the basis?
        # Only feasible if we implement `gyration_matrix`.
        # For now, skip Gyration of J (assume scalar-like for prototype), 
        # Focus on Wong Precession.
        
        # Wong: dJ/dt = -[A_eff, J]
        # A_eff ~ sum_j PT(J_j)
        # We compute A_eff
        A_eff = self._compute_effective_A(q, J)
        
        # Commutator [A, J] = A J - J A
        comm = torch.matmul(A_eff, J) - torch.matmul(J, A_eff)
        
        # Update J
        # J_new = J - dt * comm
        J_next = J - dt * comm
        
        return {
            'q': geometry.check_boundary(q_next),
            'p': p_transported,
            'J': J_next,
            'm': m,
            'E': potential
        }

    def _compute_effective_A(self, q: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """
        Approximate Effective Gauge Potential A seen by each particle.
        A_i = sum_{j != i} PT_{j->i}(J_j) * K(d)
        """
        N = q.shape[0]
        if N < 2: return torch.zeros_like(J)
        
        q_i = q.unsqueeze(1)
        q_j = q.unsqueeze(0)
        J_j = J.unsqueeze(0).expand(N, -1, -1, -1)
        
        # Transport J_j to i
        # Using gyration approximation on each column of J?
        # Too expensive.
        # Prototype: Assume flat transport (A ~ sum J_j).
        # Only valid locally.
        
        # Distance weight
        diff = geometry.mobius_add(-q_i, q_j)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        # Kernel
        weight = torch.exp(-dist**2) # (N, N, 1)
        
        # Mask self
        mask = torch.eye(N, device=q.device).unsqueeze(-1)
        weight = weight * (1 - mask)
        
        # Weighted sum (N, N, 1, 1) * (1, N, k, k) -> (N, N, k, k) -> Sum over dim 1
        weighted_J = weight.unsqueeze(-1) * J_j
        A = weighted_J.sum(dim=1)
        return A

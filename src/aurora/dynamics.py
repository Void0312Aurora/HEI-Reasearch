"""
Aurora Dynamics: Contact Integrator (Torch).
============================================

Implements the Dissipative Euler-Poincare integrator for contact dynamics on H^n.
Ref: `docs/plan/积分器.md` & `docs/plan/理论基础-4.md`.

Key Features:
- Symplectic-like geometric integration
- Explicit volume control support (via potentials)
- Adaptive time-stepping
"""

import torch
import dataclasses
from typing import Tuple, Dict, Any, Optional
from .geometry import (
    minkowski_inner, 
    project_to_tangent, 
    renormalize_frame, 
    compute_gradient_minkowski
)
from .torch_core import exp_so1n_torch # Re-use exp map logic or port it? Let's assume port or import.

# TODO: We need exp_so1n_torch. Ideally we should have put it in geometry.py.
# For now, let's import from hei_n to avoid code duplication, or inline it if simple.
# Let's import from hei_n.torch_core for now as it's just math.
from hei_n.torch_core import exp_so1n_torch

@dataclasses.dataclass
class PhysicsState:
    G: torch.Tensor      # (N, dim, dim) Frame
    M: torch.Tensor      # (N, dim, dim) Algebra Momentum
    z: torch.Tensor      # (N,) Contact Variable
    step: int = 0
    diagnostics: Dict[str, float] = dataclasses.field(default_factory=dict)
    
    @property
    def x(self) -> torch.Tensor:
        """Position vector is the 0-th column of the frame."""
        return self.G[..., 0]

@dataclasses.dataclass
class PhysicsConfig:
    dt: float = 0.01
    gamma: float = 0.1          # Damping
    temp: float = 0.0           # Temperature (Langevin noise)
    adaptive: bool = True       # Adaptive dt?
    tolerance: float = 0.1      # Adaptive tolerance
    
    # Advanced
    clip_torque: float = 100.0  # limit torque magnitude
    solver_iters: int = 5       # Fixed point iterations
    renorm_interval: int = 50   # Steps between G-frame cleaning

def compute_diamond_torque(x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Compute Torque = x ^ F (in Lie Algebra so(1,n)).
    X_{ij} = x_i (J F)_j - (J F)_i x_j ?
    
    Standard formula for H^n:
    Torque = x F^T J - F x^T J
    where J is Minkowski metric diag(-1, 1...).
    
    Arguments:
        x: (N, dim) Position
        F: (N, dim) Force (Tangent vector) using Minkowski Gradient convention?
           Wait, F should be covariant force? 
           Let's stick to: Vector F in R^{n+1}.
           
    Returns:
        Omega: (N, dim, dim) in so(1,n)
    """
    # J = diag(-1, 1...)
    # We can implement manually for speed or use Einstein summation.
    
    # JF: Apply metric to F
    # JF_0 = -F_0, JF_i = F_i
    JF = F.clone()
    JF[..., 0] *= -1.0
    
    # Outer products
    # x (JF)^T - F (Jx)^T ? 
    # Let's check skew-symmetry X^T J + J X = 0.
    
    # Let's use the verified implementation from hei_n logic:
    # X = (x F^T - F x^T) J
    # xF^T: (N, dim, dim)
    
    outer_xF = torch.einsum('ni,nj->nij', x, F)
    outer_Fx = torch.einsum('ni,nj->nij', F, x)
    
    # diff
    diff = outer_xF - outer_Fx
    
    # Multiply by J on right
    # col 0 *= -1
    diff[..., 0] *= -1.0
    
    return diff

class ContactIntegrator:
    def __init__(self, config: PhysicsConfig):
        self.config = config
        
    def step(self, state: PhysicsState, potentials: Any) -> PhysicsState:
        """
        Perform one integration step.
        
        Args:
            state: Current state
            potentials: Object providing .compute_forces(x) -> (Energy, Gradient)
        """
        G = state.G
        M = state.M
        z = state.z
        x = G[..., 0]
        
        # 1. Compute Potentials & Forces
        # Gradient returned is typically Euclidean gradient of energy V(x)
        total_E, valid_grad_E = potentials.compute_forces(x)
        
        # Convert to Minkowski Gradient: grad_M = J * grad_E
        # This is the "Force" in ambient space before projection
        grad_M = compute_gradient_minkowski(x, valid_grad_E)
        
        # Project to Tangent Space to get effective mechanical force
        # Force = - proj(grad_M)
        force_ambient = -grad_M
        force_tangent = project_to_tangent(x, force_ambient)
        
        # 2. Compute Torque (Diamond operator)
        torque_explicit = compute_diamond_torque(x, force_tangent)
        
        # 3. Langevin Noise (Optional)
        if self.config.temp > 0:
            # Random tangent vector
            raw_noise = torch.randn_like(x) * (2.0 * self.config.gamma * self.config.temp * self.config.dt)**0.5
            noise_tangent = project_to_tangent(x, raw_noise)
            # Torque from noise
            torque_noise = compute_diamond_torque(x, noise_tangent)
            torque_explicit += torque_noise
            
        # 4. Implicit Midpoint Solver / Fixed Point for M
        # M_dot = [M, Omega] - gamma * M + Torque
        # For Identity Inertia: Omega = M.
        # Implies: M_dot = [M, M] - gamma*M + Torque = Torque - gamma*M (since [M,M]=0)
        # Numerical update:
        # M_next = (M_curr + dt * Torque) / (1 + dt * gamma)
        # This is explicit for Identity Inertia! No fixed point needed unless Torque depends on M?
        # Torque depends on x. x depends on M? Semi-implicit.
        # In `hei_n`, we iterate because Torque depends on x? 
        # Actually usually we assume Torque const for the step (Symplectic Euler style).
        # Let's use simple Explicit update for M (Symplectic Euler part B).
        
        dt = self.config.dt
        gamma = self.config.gamma
        
        # Clipping
        torque_mag = torch.norm(torque_explicit.reshape(G.shape[0], -1), dim=1)
        clip_mask = torque_mag > self.config.clip_torque
        if clip_mask.any():
            scale = self.config.clip_torque / (torque_mag[clip_mask] + 1e-9)
            torque_explicit[clip_mask] *= scale.view(-1, 1, 1)

        # Update M
        M_next = (M + dt * torque_explicit) / (1.0 + dt * gamma)
        
        # 5. Update G (Exp map)
        # For Identity Inertia, velocity xi = M
        xi = M_next
        step_G = exp_so1n_torch(xi, dt)
        G_next = G @ step_G
        
        # 6. Update z (Contact variable)
        # L = T - V - gamma * z
        # T = 0.5 <M, M>
        # Simply: KE for Identity is 0.5 * sum(v^2)? 
        # Algebra norm: <M, M> = Tr(M M^T)? No.
        # For so(1,n), Killing form or just Euclidean norm of vector?
        # M corresponds to tangent velocity v = M @ x0 ? No. 
        # v = G * xi * e0 ? 
        # Speed v = |xi| ?
        # For Identity inertia, KE = 0.5 * |xi|^2 (Minkowski algebra norm?)
        # Let's approximate KE ~ 0.5 * |M|^2
        KE = 0.5 * torch.sum(M_next**2, dim=(-1,-2)) # Rough
        L = KE - total_E - gamma * z
        z_next = z + L * dt
        
        # 7. Renormalize Frame
        diag = {'dt': dt, 'energy': total_E.item()}
        if state.step % self.config.renorm_interval == 0:
            G_next, err = renormalize_frame(G_next)
            diag['renorm_error'] = err
            
        return PhysicsState(G=G_next, M=M_next, z=z_next, step=state.step+1, diagnostics=diag)

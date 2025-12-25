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
from .torch_core import exp_so1n_torch
# We need inverse of group element G. For SO(1,n), G^-1 = J G^T J.
# Define J locally or import? Let's check imports.
# We can define J since dim is known from G.

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
    dt: float = 0.01            # Base/Max dt
    min_dt: float = 1e-5        # [NEW] Min dt
    gamma: float = 0.1          # Damping
    temp: float = 0.0           # Temperature (Langevin noise)
    adaptive: bool = True       # Adaptive dt?
    tolerance: float = 0.1      # Adaptive tolerance (max disp per step)
    
    # Advanced
    clip_torque: float = 100.0  # limit torque magnitude
    solver_iters: int = 5       # Fixed point iterations
    renorm_interval: int = 50   # Steps between G-frame cleaning

from .diamond import compute_diamond_torque

class ContactIntegrator:
    def __init__(self, config: PhysicsConfig, inertia: Any):
        self.config = config
        self.inertia = inertia
        
    def _compute_adaptive_dt(self, xi_norm: float, torque_norm: float) -> float:
        """
        Calculate safe time step based on displacement tolerance.
        dt <= tol / |v|
        dt <= sqrt(2 * tol / |a|) -> approximated as tol / |a| for conservatism linearity
        """
        cfg = self.config
        
        # Velocity limit: dt * v < tol
        if xi_norm > 1e-9:
            dt_vel = cfg.tolerance / xi_norm
        else:
            dt_vel = cfg.dt
            
        # Acceleration limit: dt * (torque/I) < tol? 
        # Actually dt^2 * a < tol usually. 
        # But let's follow hei_n robust heuristic: dt < tol / a (more conservative than sqrt)
        # Assuming I_min = 1.0 (RadialMass >= 1)
        if torque_norm > 1e-9:
             dt_acc = cfg.tolerance / torque_norm
        else:
             dt_acc = cfg.dt
             
        # Damping limit: dt * gamma < 1.0 (stability)
        if cfg.gamma > 1e-9:
            dt_gamma = 0.8 / cfg.gamma
        else:
            dt_gamma = cfg.dt
            
        dt = min(cfg.dt, dt_vel, dt_acc, dt_gamma)
        dt = max(dt, cfg.min_dt)
        return dt

    def step(self, state: PhysicsState, potentials: Any, freeze_radius: bool = False) -> PhysicsState:
        """
        Perform one integration step with Advection and Geometric Forces.
        """
        G = state.G
        M = state.M
        z = state.z
        x = G[..., 0]
        
        # Capture old radius if freezing
        if freeze_radius:
            r_old = torch.acosh(torch.clamp(x[:, 0], min=1.0 + 1e-7))

        
        # 1. Compute Forces (Potentials + Geometric)
        # Potential Force
        # Gradient returned is Euclidean gradient of V(x).
        total_E, valid_grad_E = potentials.compute_forces(x)
        grad_M = compute_gradient_minkowski(x, valid_grad_E) # J * grad_E
        
        # Geometric Force (Centrifugal)
        # F_geom is already in ambient space R^{dim}
        F_geom = self.inertia.geometric_force(M, x)
        
        # Total Force Magnitude = -grad_V + F_geom
        # Note: grad_M is gradient (covariant?). Force is contravariant?
        # Force = - grad_M.
        # F_total = -grad_M + F_geom
        F_total = -grad_M + F_geom
        
        # Project to Tangent Space
        F_tangent = project_to_tangent(x, F_total)
        
        # 2. Compute Body Torque
        # Pull back F_tangent to body frame to get Omega
        torque_body = compute_diamond_torque(G, F_tangent)
        
        # 3. Langevin Noise
        if self.config.temp > 0:
            raw_noise = torch.randn_like(x) * (2.0 * self.config.gamma * self.config.temp * self.config.dt)**0.5
            noise_tangent = project_to_tangent(x, raw_noise)
            torque_noise = compute_diamond_torque(G, noise_tangent)
            torque_body += torque_noise
        
        # [NEW] Adaptive Time Stepping Logic
        # Calculate norms for adaptation
        # xi = I^-1 M. If I is diagonal/scalar, xi_norm ~ M_norm / m.
        # But we don't have xi handy yet. Compute explicit guess.
        # NOTE: For adaptive check, we use current state.
        xi_guess = self.inertia.inverse(M, x)
        xi_norm = torch.max(torch.norm(xi_guess.reshape(G.shape[0], -1), dim=1)).item()
        torque_norm = torch.max(torch.norm(torque_body.reshape(G.shape[0], -1), dim=1)).item()
        
        if self.config.adaptive:
            dt = self._compute_adaptive_dt(xi_norm, torque_norm)
        else:
            dt = self.config.dt
            
        # 4. Momentum Update Scheme (Implicit Fixed-Point)
        # We need to solve:
        # M_next = (Ad^*_{g^-1} M_prev + dt * Torque(M_next)) / (1 + dt*gamma)
        # where Torque depends on F_geom(M_next) and g depends on xi(M_next).
        
        gamma = self.config.gamma
        dim = G.shape[-1]
        
        # J for inverse Adjoint
        J = torch.eye(dim, device=G.device)
        J[0, 0] = -1.0
        J = J.unsqueeze(0) # (1, dim, dim)
        
        # Initial Guess: Explicit Euler
        xi_curr = xi_guess
        
        # We use the current M as the seed for iteration
        M_next = M.clone() 
        xi_next = xi_curr
        
        # Solver Loop (Using adaptive dt)
        for i in range(self.config.solver_iters):
             # A. Compute Group Step from current guess
             g_step = exp_so1n_torch(xi_next, dt)
             g_inv = J @ g_step.transpose(-2, -1) @ J
             
             # B. Advect Previous Momentum
             # M_adv = g M_prev g^-1
             M_advected = g_step @ M @ g_inv
             
             # C. Compute Forces at Guess State
             # Note: x is technically fixed in this splitting step, 
             # preventing "implicit position" which is very expensive.
             # We rely on "Force evaluated at x, but Momentum evaluated at M_next".
             F_geom = self.inertia.geometric_force(M_next, x)
             
             # Total Tangent Force
             F_total_loop = -grad_M + F_geom
             F_tangent_loop = project_to_tangent(x, F_total_loop)
             
             # Torque
             torque_loop = compute_diamond_torque(G, F_tangent_loop)
             
             # Langevin Noise (Additive, frozen)
             if self.config.temp > 0:
                  torque_loop += torque_noise
             
             # Clip Torque
             t_mag = torch.norm(torque_loop.reshape(G.shape[0], -1), dim=1)
             clip_mask = t_mag > self.config.clip_torque
             if clip_mask.any():
                 scale = self.config.clip_torque / (t_mag[clip_mask] + 1e-9)
                 torque_loop[clip_mask] *= scale.view(-1, 1, 1)
                 
             # D. Update Estimate
             M_target = (M_advected + dt * torque_loop) / (1.0 + dt * gamma)
             
             # Discrepancy
             # diff = torch.max(torch.abs(M_target - M_next))
             
             # Mixing (Under-relaxation)
             beta = 0.5
             M_next = (1.0 - beta) * M_next + beta * M_target
             
             # Update Velocity for next iteration
             xi_next = self.inertia.inverse(M_next, x)
             
        # End of Loop
        
        # 5. Velocity Update
        xi_next = self.inertia.inverse(M_next, x)
        
        # 6. Update G
        step_G = exp_so1n_torch(xi_next, dt)
        G_next = G @ step_G
        
        # 7. Update z (Contact)
        # KE = T(M_next, x). Wait, x has changed? 
        # Standard symplectic approach: use M_next and *old* x or *new* x?
        # Usually semi-implicit. Using old x for KE calc is simpler.
        KE = torch.sum(self.inertia.kinetic_energy(M_next, x)) # Sum or per particle?
        # L = T - V - gamma * z
        # Here we just use per-particle logic? 
        # z is (N,). T is (N,). V is scalar total? 
        # Contact mechanics usually global z.
        # But here z is (N,). So per-particle action.
        T_local = self.inertia.kinetic_energy(M_next, x)
        # V is total E? No, V should be local potential effectively.
        # But potentials return total E.
        # Approximating L per particle is hard if V is interacting.
        # For simple damping we might ignoring V in z update if we only care about global dissipation?
        # Or just assign E/N?
        # hei_n uses `potential_fn(z, action)` -> returns float.
        # This part is fuzzy in high-dim code.
        # Let's simple damping: z_next = z + (T - E/N - gamma z)*dt
        V_per = total_E / G.shape[0]
        L = T_local - V_per - gamma * z
        z_next = z + L * dt
        
        # 8. Renormalize Frame
        # Calculate avg mass for diagnostics
        avg_mass = torch.mean(self.inertia._get_mass(x)).item()
        diag = {'dt': dt, 'energy': total_E.item(), 'avg_mass': avg_mass}
        
        if state.step % self.config.renorm_interval == 0 or freeze_radius:
            # If freezing, we MUST restore radius first
            if freeze_radius:
                # x_next is G_next[..., 0]
                x_curr = G_next[..., 0]
                r_curr = torch.acosh(torch.clamp(x_curr[:, 0], min=1.0+1e-7))
                
                # Target is r_old
                sinh_t = torch.sinh(r_old)
                sinh_c = torch.sinh(r_curr)
                
                # Scale spatial: x_s_new = x_s_curr * (sinh_t / sinh_c)
                scale = (sinh_t / (sinh_c + 1e-9)).unsqueeze(-1)
                
                # Modify G_next column 0 in place
                G_next[..., 0, 1:] *= scale
                G_next[..., 0, 0] = torch.cosh(r_old)
                
                # Renormalize to fix orthogonality
                G_next, err = renormalize_frame(G_next)
                diag['renorm_error'] = err
                diag['frozen'] = 1.0
            
            elif state.step % self.config.renorm_interval == 0:
                G_next, err = renormalize_frame(G_next)
                diag['renorm_error'] = err
            
        return PhysicsState(G=G_next, M=M_next, z=z_next, step=state.step+1, diagnostics=diag)

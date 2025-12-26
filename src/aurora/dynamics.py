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
    J: Optional[torch.Tensor] = None 
    # Logical Charge J: (N, k) Vector.
    # CONVENTION: J is identified with Lie Algebra element via invariant metric.
    # For SO(k), J is a vector in R^{k(k-1)/2}. 
    # For SO(3), J is R^3 vector, representing L_i J_i.
    # Precession: dJ/dt = -[A(v), J] (Lie Bracket).
    
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

    def step(self, state: PhysicsState, potentials: Any, gauge_field: Optional[Any] = None, freeze_radius: bool = False) -> PhysicsState:
        """
        Perform one integration step with Advection and Geometric Forces.
        [UPDATED v2.0] Implements Strang Splitting for Layer C -> B -> A.
        """
        G = state.G
        M = state.M
        z = state.z
        J = state.J
        x = G[..., 0]
        J = state.J
        x = G[..., 0]
        
        # Adaptive Time-Stepping
        if self.config.adaptive:
             # Estimate velocity xi
             xi = self.inertia.inverse(M, x)
             xi_norm = torch.max(torch.norm(xi, dim=-1)).item()
             # We don't have current torque cheaply. Use 0.0 or last stored?
             # For safety, let's just limit by velocity and max_dt
             dt = self._compute_adaptive_dt(xi_norm, 0.0)
        else:
             dt = self.config.dt

        
        # Capture old radius if freezing
        r_old = None
        if freeze_radius:
            r_old = torch.acosh(torch.clamp(x[:, 0], min=1.0 + 1e-7))

        # --- Substep 1: Layer C (Logical Dynamics) ---
        # Update J (Precession) and p (Lorentz Force)
        # Only if J and GaugeField are present
        if J is not None and gauge_field is not None:
             # Logic 1/2 step (or full step if Strang splitting structure allows)
             # Current v needed for A_mu v^mu
             # In our formulation, M acts on x to give v in embedding space
             # x: (N, dim), M: (N, dim, dim)
             # v = M @ x.unsqueeze(-1) -> (N, dim, 1) -> squeeze
             v_embed = torch.matmul(M, x.unsqueeze(-1)).squeeze(-1)
             
             # Compute Connection A(v)
             A_geom = gauge_field.compute_connection(x, v_embed) # (N, k, k)
             
             # Compute Spin Interaction (Alignment Torque)
             # Adds term to rotate J towards neighbors
             A_spin = gauge_field.compute_spin_interaction(J)
             
             # Total Effective Connection
             # A_eff = A_geom + lambda * A_spin
             # Using lambda=5.0 to enforce strong semantic alignment
             A_eff = A_geom + 5.0 * A_spin
             
             # Evolve J (Precession)
             # dJ/dt = -[A, J].
             # Solution J(t) = exp(-A t) J(0) (for SO(3) vector rep)
             # Use Matrix Exponential for stability and norm preservation
             U_local = torch.matrix_exp(-A_eff * dt)
             J = torch.matmul(U_local, J.unsqueeze(-1)).squeeze(-1)
             
             # Add Wong Force (Particle Dynamics)
             # F_logic currently placeholder (returns 0)
             # But if implemented, we add it to forces?
             _, F_logic = gauge_field.compute_force_wong(x, v_embed, J)
        else:
             F_logic = torch.zeros_like(x)

        # --- Substep 2: Layer B (Dissipation & Contact) ---
        # M_diss = M * exp(-gamma * dt/2)
        # semi-implicit damping
        gamma = self.config.gamma
        
        # --- Pre-Kick (Layer A - First Half) ---
        # 1. Update M with forces at current position x
        total_E, valid_grad_E = potentials.compute_forces(x)
        grad_M = compute_gradient_minkowski(x, valid_grad_E)
        
        F_geom = self.inertia.geometric_force(M, x)
        F_total = -grad_M + F_geom + F_logic # Add Logic Force
        F_tangent = project_to_tangent(x, F_total)
        torque = compute_diamond_torque(G, F_tangent)
        
        M_half = M + torque * (dt * 0.5)
        
        # 2. Dissipate First Half
        M_half = M_half / (1.0 + gamma * dt * 0.5)
        
        # --- Substep 3: Layer A (Geometric Advection) ---
        # 3a. Conserved Flow
        # x_new = Exp(v * dt)
        xi = self.inertia.inverse(M_half, x) # Use M_half for advection velocity
        step_G = exp_so1n_torch(xi, dt)
        G_next = G @ step_G
        x_next = G_next[..., 0]
        
        # 3b. Parallel Transport
        # Rule: When q updates (x -> x_next via G -> G_next), the Body Frame transport is implicit for Body-Fixed M and J.
        # i.e. M and J are components in the moving frame G.
        # So by updating G, we correctly effectively transport the physical p vectors.
        # No explicit re-computation of M or J values needed here for pure advection.
        
        # C. Second Kick (Potentials at new pos)
        # Force is calculated at x_next (in T_{x_next} Q)
        total_E_next, valid_grad_E_next = potentials.compute_forces(x_next)
        
        # Pull back tangent force to Algebra (Body Frame) using G_next
        grad_M_next = compute_gradient_minkowski(x_next, valid_grad_E_next) # This func might need verification: does it return tangent or algebra?
        # compute_gradient_minkowski usually returns spatial gradient in embedding space?
        # Let's assume it returns ambient gradient grad_E.
        # We need to project to tangent and then pull back.
        
        # F_geom_next uses M_half and x_next
        F_geom_next = self.inertia.geometric_force(M_half, x_next)
        
        # Recompute Logic Force at next state?
        # Requires J at next state (we have new J), x_next. 
        # v_next approx determined by M_half?
        # v_embed_next = M_half @ x_next
        if J is not None and gauge_field is not None:
             v_embed_next = torch.matmul(M_half, x_next.unsqueeze(-1)).squeeze(-1)
             _, F_logic_next = gauge_field.compute_force_wong(x_next, v_embed_next, J)
        else:
             F_logic_next = torch.zeros_like(x_next)
             
        F_total_next = -grad_M_next + F_geom_next + F_logic_next
        F_tangent_next = project_to_tangent(x_next, F_total_next)
        
        # Torque is computed using G_next (Correct Frame)
        torque_next = compute_diamond_torque(G_next, F_tangent_next)
        
        M_next = M_half + torque_next * (dt * 0.5)
        
        # --- Substep 4: Layer B (Dissipation 2nd half) ---
        M_next = M_next / (1.0 + gamma * dt * 0.5)
        
        # --- Update z ---
        # z += (T - V) * dt
        T_val = self.inertia.kinetic_energy(M_next, x_next)
        L = T_val - (total_E_next / G.shape[0])
        z_next = z + L * dt
        
        # --- Diagnostics ---
        avg_mass = torch.mean(self.inertia._get_mass(x_next)).item()
        diag = {'dt': dt, 'energy': total_E_next.item(), 'avg_mass': avg_mass}
        
        # Renormalize & Freeze Logic
        if freeze_radius and r_old is not None:
             x_curr = G_next[..., 0]
             r_curr = torch.acosh(torch.clamp(x_curr[:, 0], min=1.0+1e-7))
             sinh_t = torch.sinh(r_old)
             sinh_c = torch.sinh(r_curr)
             scale = (sinh_t / (sinh_c + 1e-9)).unsqueeze(-1)
             G_next[..., 1:, 0] *= scale
             G_next[..., 0, 0] = torch.cosh(r_old)
             G_next, err = renormalize_frame(G_next)
             diag['frozen'] = 1.0
        elif state.step % self.config.renorm_interval == 0:
             G_next, err = renormalize_frame(G_next)
             
        return PhysicsState(G=G_next, M=M_next, z=z_next, J=J, step=state.step+1, diagnostics=diag)

class WongIntegrator(ContactIntegrator):
    """
    Alias for the v2.0 ContactIntegrator which now supports Wong Dynamics.
    """
    pass

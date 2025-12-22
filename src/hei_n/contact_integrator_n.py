"""
N-Dimensional Contact Integrator for SO(1, n).
=============================================

Implements the Dissipative Euler-Poincare Equations on the Contact Manifold M = T*Q x R_z.

References:
- HEI/docs/理论基础-4.md (Axiom 1.2, Theorem 3.2)

State:
- G: (N, dim, dim) Configuration in SO(1, n)
- M: (N, dim, dim) Momentum in lie algebra so(1, n)
- z: (1,) Global Contact Action / Free Energy History

Dynamics (Herglotz):
L(xi, a, z) = T(xi) - V(a) - gamma * z

1. d/dt (I xi) = ad*_{xi} (I xi) + Torque - gamma * (I xi)
2. da/dt = xi . a
3. dz/dt = L = T - V - gamma * z

This integrator upgrades the standard GroupIntegratorN by adding the 'z' variable
and requiring the potential oracle to return energy values V, not just gradients.
"""

import numpy as np
import dataclasses
from typing import Protocol, Any, List

from .lie_n import exp_so1n, minkowski_metric
from .geometry_n import project_to_tangent

class PotentialOracleN(Protocol):
    """Protocol for N-dimensional potential."""
    def potential(self, x: np.ndarray) -> float:
        ...
    def gradient(self, x: np.ndarray) -> np.ndarray:
        ...

@dataclasses.dataclass
class ContactStateN:
    G: np.ndarray # (N, dim, dim)
    M: np.ndarray # (N, dim, dim)
    z: float = 0.0 # Global Contact Action
    x: np.ndarray = None # (N, dim) Cached position
    
    def __post_init__(self):
        if self.x is None:
            self.x = self.G[..., 0]

@dataclasses.dataclass
class ContactConfigN:
    dt: float = 0.01
    gamma: float = 0.0 # Damping
    
class ContactIntegratorN:
    def __init__(self, oracle: PotentialOracleN, inertia: Any, config: ContactConfigN):
        self.oracle = oracle
        self.inertia = inertia
        self.config = config

    def step(self, state: ContactStateN) -> ContactStateN:
        G = state.G
        M = state.M
        z = state.z
        dt = self.config.dt
        gamma = self.config.gamma
        
        # 0. Get current position
        x_world = G[..., 0]
        
        # 1. Potential Energy and Gradient
        try:
            V = self.oracle.potential(x_world)
        except AttributeError:
             V = 0.0 
        grad_potential = self.oracle.gradient(x_world)

        # 2. Geometric Force
        # F_geom = - grad_x T(xi)
        grad_geom = self.inertia.geometric_force(M, x_world)
        
        # Total Gradient in Ambient Space
        # Force = - (grad_V + grad_geom) ? 
        # Actually F_geom is usually defined as -grad K.
        # But wait, grad_geom usually points towards lower inertia (higher mass if K ~ 1/m).
        # Yes.
        
        grad_total = grad_potential - grad_geom # -grad_geom is +force
        
        # 3. Project Forces and Compute Torque
        # Force_world = - project_to_tangent(x_world, grad_total)
        # However, geometric_force might strictly require projection too.
        
        Force_world = - project_to_tangent(x_world, grad_total)
        
        # Compute Diamond Torque (Structure Feedback)
        from .inertia_n import compute_diamond_torque
        Torque = compute_diamond_torque(G, Force_world)
        
        # 4. Contact Dynamics (z evolution)
        # L = T - V - gamma * z
        T_val = self.inertia.kinetic_energy(M, x_world)
        
        L = T_val - V - gamma * z
        
        # Update z (Euler)
        z_new = z + L * dt
        
        # 5. Momentum Dynamics (Dissipative Euler-Poincare)
        # dM/dt = ad* (I xi) ... 
        # Wait, if I is scalar, ad* term might cancel or simplify?
        # For M in so(1,n), [M, M] = 0.
        # So dM/dt = Torque - gamma * M.
        # Note: If I was full tensor, we'd need ad*.
        
        M_rate = Torque - gamma * M
        M_new = M + M_rate * dt
        
        # 6. Position Dynamics
        # G_new = G * exp(dt * xi)
        # xi = I^{-1} M
        xi = self.inertia.inverse_inertia(M_new, x_world)

        step_g = exp_so1n(xi, dt=dt)
        G_new = G @ step_g
        
        return ContactStateN(G=G_new, M=M_new, z=z_new)

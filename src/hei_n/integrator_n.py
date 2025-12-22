"""
N-Dimensional Group Integrator for SO(1, n).
=============================================

This integrator generalizes the GroupContactIntegrator to arbitrary dimensions.
It works directly with Matrix Lie Algebra elements so(1, n) instead of vector bases,
avoiding the complexity of constructing structure constants for high dimensions.

State:
- G: (N, dim, dim) Group elements in SO(1, n)
- M: (N, dim, dim) Momentum in algebra so(1, n) (Body Frame)
- x: (N, dim) Position in Embedding Space R^{n+1} (extracted from G)

Hamiltonian:
H = 0.5 * <M, I^{-1} M> + V(x)
For "Riemannian Particle", I is Identity, so H = 0.5 * tr(M^T M) (roughly).
Actually, with Killing form... let's stick to simplest Kinetic Energy.
K(M) = 0.5 * frobenius_norm(M)^2 ? 
Or standard: K = 0.5 * tr(M^T M).

Equations of Motion (Lie-Poisson / Euler-Poincare):
dG/dt = G * (I^{-1} M)
dM/dt = ad*_{I^{-1}M} (M) + Torque
Torque = Lift(Force).

Torque Lifting:
F is force in R^{n+1} (Euclidean gradient projected to Tangent).
Torque in Body Frame:
T_body = G^{-1} * (x_base ^ F) ?
Actually, Torque = x_body ^ F_body.
x_body = (1, 0, ... 0) (Origin in body frame).
F_body = G^T * F_world.
Wedge product in R^{1,n}:
Torque_ij = x_i F_j - x_j F_i (with metric signs).
Let's assert Torque is in so(1,n).
"""

import numpy as np
import dataclasses
from typing import Callable, List, Tuple

from .lie_n import exp_so1n, check_algebra_membership, minkowski_metric, project_to_algebra
from .geometry_n import project_to_tangent, dist_grad_n

@dataclasses.dataclass
class IntegratorStateN:
    G: np.ndarray # (N, dim, dim)
    M: np.ndarray # (N, dim, dim) Momentum Matrix
    x: np.ndarray = None # (N, dim) Cached position
    
    def __post_init__(self):
        if self.x is None:
            # Extract column 0 of G.
            # G maps Body Origin e0=(1,0...) to World x.
            # x = G * e0 = G[:, :, 0]
            self.x = self.G[..., 0] 

@dataclasses.dataclass
class IntegratorConfigN:
    dt: float = 0.01
    gamma: float = 0.0 # Damping
    metric_signature: List[int] = None # [-1, 1, 1, 1] for 3D

class GroupIntegratorN:
    def __init__(self, potential_grad_fn, config: IntegratorConfigN):
        """
        potential_grad_fn(x) -> Euclidean Gradients in R^{dim}
        """
        self.grad_fn = potential_grad_fn
        self.config = config
        
    def _body_torque(self, G, F_world):
        """
        Compute torque in Body Frame.
        Torque = x_body wedge F_body.
        x_body for Origin is e0 = [1, 0, ...].
        F_body = G^{-1} F_world = G^T J G J F_world?
        Actually simpler:
        G maps Body -> World.
        F_body is pull-back of F_world.
        F_body = G^{-1} * F_world (Matrix vector mul).
        Since G is in SO(1,n), G^{-1} = J G^T J.
        
        So F_body = (J G^T J) @ F_world.
        
        Then Torque = e0 ^ F_body.
        Torque_matrix T:
        T = e0 * (J F_body)^T - F_body * (J e0)^T ?
        Constraint: T must be in so(1, n).
        Let check algebra.
        u = F_body projected to "boost" part?
        If x_body = e0, then Torque is purely a Boost generator.
        
        Torque_{0i} = -F_body_i
        Torque_{i0} = -F_body_i
        (Using J=diag(-1,1...))
        
        Construction:
        T = np.zeros(dim, dim)
        T[0, 1:] = F_body[1:]
        T[1:, 0] = F_body[1:]
        
        Wait, F_body[0] (time component force) should be 0 
        if F is tangent to hyperboloid?
        Tangent space at e0 is { (0, v1, ... vn) }.
        So F_body[0] MUST be 0.
        """
        dim = G.shape[-1]
        J = minkowski_metric(dim)
        
        # 1. Pull back Force to Body Frame
        # G_inv = J @ G.swapaxes(-1, -2) @ J
        G_inv = J @ np.swapaxes(G, -1, -2) @ J
        F_body = (G_inv @ F_world[..., np.newaxis]).squeeze(-1)
        
        # 2. Construct Torque
        # Since x_body = e0 = [1, 0...], the torque is the wedge e0 ^ F_body.
        # This corresponds to a pure Boost in the direction of F_body.
        # Matrix T: T @ v = <F, v> e0 - <e0, v> F
        # No, that's regular wedge.
        # Functional derivative of V(g * x0) wrt g at identity.
        # Let g = exp(epsilon * Xi).
        # x_new = (I + eps Xi) x0 = x0 + eps Xi x0.
        # V(x_new) ~ V(x0) + <gradV, eps Xi x0>
        # Torque = grad_Xi <gradV, Xi x0>
        #        = x0 ^ gradV (with metric).
        
        # Result for x0 = e0:
        # Xi x0 = Xi[:, 0] = column 0 of Xi.
        # We need <F_body, column 0 of Xi>_M.
        # Let Xi = [[0, u^T], [u, W]]
        # col 0 = [0, u].
        # <F_body, [0, u]>_M.
        # F_body = [0, f_rest] (since tangent).
        # Inner prod = f_rest dot u.
        # So Torque is the element Xi such that <Torque, H> = f dot u_H.
        # This implies Torque has u = f_rest, W = 0.
        
        T = np.zeros_like(G)
        f_rest = F_body[..., 1:] # (N, dim-1)
        
        # Fill Boost parts
        T[..., 0, 1:] = f_rest
        T[..., 1:, 0] = f_rest
        
        return T

    def step(self, state: IntegratorStateN) -> IntegratorStateN:
        G = state.G
        M = state.M
        dt = self.config.dt
        
        dim = G.shape[-1]
        
        # 1. Get Force in World Frame
        x_world = G[..., 0]
        # Calculate Gradient (Euclidean gradient of potential)
        grad_euc = self.grad_fn(x_world)
        
        # Project Euclidean Gradient to Tangent Space (just in case)
        # Force = - grad V
        # In Minkowski, Force_cov = -grad.
        # Force_con = J * Force_cov ? 
        # Simpler: The return of grad_fn is "Vector that pushes x".
        # Assume it is the correct direction in R^{n+1}.
        
        # Force = - grad_euc projected
        Force_world = - project_to_tangent(x_world, grad_euc)
        
        # 2. Compute Torque
        Torque = self._body_torque(G, Force_world)
        
        # 3. Damping
        if self.config.gamma > 0:
            Torque -= self.config.gamma * M
            
        # 4. Update Momentum (Euler Step for now, Symplectic is harder with exp mapping)
        # dM/dt = [M, Omega] + Torque ?
        # Inertia I=Identity => Omega = M.
        # [M, M] = 0.
        # So dM/dt = Torque.
        # Simple!
        
        M_new = M + Torque * dt
        
        # 5. Update Position (G)
        # G_new = G * exp(dt * I^{-1} M_new)
        # I=Id => exp(dt * M_new)
        
        step_g = exp_so1n(M_new, dt=dt)
        G_new = G @ step_g
        
        return IntegratorStateN(G=G_new, M=M_new)

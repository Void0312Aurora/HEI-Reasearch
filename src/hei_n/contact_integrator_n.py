
"""
N-Dimensional Contact Integrator for SO(1, n) (Robust Group Version).
====================================================================

Implements the Dissipative Euler-Poincare Equations on the Contact Manifold M = T*Q x R_z.
Incorporates robust numerical methods from 'hei.group_integrator':
1. Explicit SO(1,n) Renormalization (Hyperbolic Gram-Schmidt).
2. Implicit Solver for Momentum (Fixed-point iteration).
3. Adaptive Time Stepping.

References:
- HEI/docs/理论基础-4.md (Axiom 1.2, Theorem 3.2)
- Iserles et al. (2000) Lie-group methods.
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
    G: np.ndarray # (N, dim, dim) Configuration in SO(1, n)
    M: np.ndarray # (N, dim, dim) Momentum in lie algebra so(1, n)
    z: float = 0.0 # Global Contact Action
    x: np.ndarray = None # (N, dim) Cached position
    diagnostics: dict = dataclasses.field(default_factory=dict) # NEW: Robustness Metrics
    
    def __post_init__(self):
        if self.x is None:
            self.x = self.G[..., 0]

@dataclasses.dataclass
class ContactConfigN:
    dt: float = 0.01  # Max dt
    min_dt: float = 1e-5
    gamma: float = 0.0 # Base damping
    target_temp: float = None 
    thermostat_tau: float = 10.0 
    
    # Robustness Parameters
    fixed_point_iters: int = 3
    solver_mixing: float = 0.5
    torque_clip: float = 50.0
    renorm_interval: int = 50
    adaptive: bool = True
    tol_disp: float = 0.1 # Max displacement per step

class ContactIntegratorN:
    def __init__(self, oracle: PotentialOracleN, inertia: Any, config: ContactConfigN):
        self.oracle = oracle
        self.inertia = inertia
        self.config = config
        self.gamma = config.gamma 
        self._step_count = 0

    def _renormalize_so1n(self, G: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Hyperbolic Gram-Schmidt Orthogonalization.
        Returns renormalized G and the maximum correction magnitude.
        """
        G_new = G.copy()
        G_old = G.copy()
        
        # G shape: (N, dim, dim)
        dim = G.shape[-1]
        
        def mink_inner(a, b):
            # a, b: (N, dim)
            return np.sum(a * b * np.array([-1] + [1]*(dim-1)), axis=-1)
            
        # Col 0
        v0 = G_new[..., 0]
        norm_sq = mink_inner(v0, v0)
        scale0 = 1.0 / np.sqrt(np.abs(norm_sq) + 1e-15)
        G_new[..., 0] *= scale0[:, np.newaxis]
        
        basis = [G_new[..., 0]] 
        
        # Col 1..dim-1 
        for k in range(1, dim):
            vk = G_new[..., k]
            
            # GS Projection
            for j in range(k):
                ej = basis[j]
                factor = mink_inner(vk, ej)
                if j == 0:
                    factor *= -1.0 
                
                vk -= factor[:, np.newaxis] * ej
                
            # Normalize
            norm_sq = mink_inner(vk, vk)
            scale = 1.0 / np.sqrt(np.abs(norm_sq) + 1e-15)
            vk *= scale[:, np.newaxis]
            
            basis.append(vk)
            G_new[..., k] = vk
            
        # Compute Correction Magnitude (L2 norm of difference)
        diff = np.linalg.norm(G_new - G_old, axis=(1,2))
        max_diff = float(np.max(diff))
        
        return G_new, max_diff

    def _calculate_manifold_error(self, G: np.ndarray) -> float:
        """Calculate max deviation from G^T J G = J."""
        dim = G.shape[-1]
        J = minkowski_metric(dim)
        # G^T J G
        GTG = np.swapaxes(G, -1, -2) @ J @ G
        err = np.abs(GTG - J)
        return float(np.max(err))

    def _adaptive_dt(self, xi_norm: float, torque_norm: float, I_min: float) -> float:
        cfg = self.config
        
        dt_vel = cfg.tol_disp / max(xi_norm, 1e-9)
        dt_acc = cfg.tol_disp / max(torque_norm / I_min, 1e-9) 
        dt_gamma = 0.8 / max(self.gamma, 1e-9)
        
        dt = min(cfg.dt, dt_vel, dt_acc, dt_gamma)
        dt = max(dt, cfg.min_dt)
        return dt

    def step(self, state: ContactStateN) -> ContactStateN:
        self._step_count += 1
        G = state.G
        M = state.M
        z = state.z
        
        diag = {}
        
        # --- Pre-Step Diagnostics ---
        diag['manifold_error'] = self._calculate_manifold_error(G)
        
        x_world = G[..., 0]
        
        # --- Thermostat ---
        if self.config.target_temp is not None:
             N = G.shape[0]
             T_current = self.inertia.kinetic_energy(M, x_world) / N 
             delta_gamma = (self.config.dt / self.config.thermostat_tau) * (T_current - self.config.target_temp)
             self.gamma = max(0.0, self.gamma + delta_gamma)
             
        gamma = self.gamma
        
        # --- 1. Compute Forces ---
        try:
            V = self.oracle.potential(x_world)
        except AttributeError:
             V = 0.0 
        grad_pot = self.oracle.gradient(x_world)
        I_min = 1.0 
        
        grad_geom = self.inertia.geometric_force(M, x_world)
        Force_world = - project_to_tangent(x_world, grad_pot - grad_geom)
        
        from .inertia_n import compute_diamond_torque
        Torque_explicit = compute_diamond_torque(G, Force_world)
        
        # --- 2. Adaptive Time Step ---
        xi_guess = self.inertia.inverse_inertia(M, x_world)
        xi_norm = np.max(np.linalg.norm(xi_guess.reshape(G.shape[0], -1), axis=1))
        torque_norm = np.max(np.linalg.norm(Torque_explicit.reshape(G.shape[0], -1), axis=1))
        
        if self.config.adaptive:
            dt = self._adaptive_dt(xi_norm, torque_norm, I_min)
        else:
            dt = self.config.dt
            
        diag['dt'] = dt
            
        # --- 3. Implicit Momentum Update (Fixed Point) ---
        xi_iter = xi_guess.copy()
        M_current = M.copy()
        damping_factor = 1.0 / (1.0 + dt * gamma)
        
        solver_res = 0.0
        
        for i in range(self.config.fixed_point_iters):
             if hasattr(self.inertia, 'mass'):
                 M_temp = self.inertia.mass(x_world)[:, None, None] * xi_iter 
             else:
                 M_temp = xi_iter 
                 
             grad_geom_iter = self.inertia.geometric_force(M_temp, x_world)
             
             Torque_loop = Torque_explicit 

             t_mag = np.linalg.norm(Torque_loop.reshape(G.shape[0], -1), axis=1)
             mask_clip = t_mag > self.config.torque_clip
             if np.any(mask_clip):
                scale = self.config.torque_clip / (t_mag[mask_clip] + 1e-9)
                Torque_loop[mask_clip] *= scale[:, np.newaxis, np.newaxis]
            
             M_next_est = (M_current + dt * Torque_loop) * damping_factor
             xi_next = self.inertia.inverse_inertia(M_next_est, x_world)
             
             # Residual
             res = np.max(np.abs(xi_next - xi_iter))
             solver_res = res
             
             beta = self.config.solver_mixing
             xi_iter = (1.0 - beta) * xi_iter + beta * xi_next
             
        diag['solver_iterations'] = self.config.fixed_point_iters
        diag['solver_residual'] = float(solver_res)
        
        xi_final = xi_iter
        M_new = M_next_est
        
        # --- 4. Position Update ---
        step_g = exp_so1n(xi_final, dt=dt)
        G_new = G @ step_g
        
        # --- 5. Contact Action Update ---
        T_val = self.inertia.kinetic_energy(M_new, x_world)
        L = T_val - V - gamma * z
        z_new = z + L * dt
        
        # --- 6. Renormalization ---
        renorm_mag = 0.0
        if self._step_count % self.config.renorm_interval == 0:
            G_new, renorm_mag = self._renormalize_so1n(G_new)
            
        diag['renorm_magnitude'] = renorm_mag
        
        return ContactStateN(G=G_new, M=M_new, z=z_new, diagnostics=diag)

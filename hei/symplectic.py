"""
Symplectic Lie Group Integrators (Optimization Mode)
===================================================

Designed for high-efficiency optimization tasks (Structure Learning, Embedding).
Unlike `GroupContactIntegrator` which focuses on exact energy conservation (Simulation),
these integrators focus on symplectic structure preservation with fixed step sizes.

Algorithms:
----------
1. Symplectic Lie Euler (1st order):
   - Update Momentum: m_{k+1} = m_k + h * Torque(z_k)
   - Update Group:    g_{k+1} = g_k * exp(h * I^{-1} m_{k+1})
   - Fast, robust, supports large dt.

2. Symplectic Lie Verlet (2nd order, future work):
   - Störmer-Verlet on Lie Algebra.
"""

import numpy as np
from typing import Callable, Optional
import dataclasses
from numpy.typing import ArrayLike, NDArray

from .lie import exp_sl2, mobius_action_matrix
from .geometry import cayley_uhp_to_disk, disk_to_hyperboloid
from .inertia import riemannian_inertia, apply_inertia, invert_inertia
from .diamond import aggregate_torque

# Re-use state definition or define a lighter one?
# We can re-use GroupIntegratorState for compatibility.
from .group_integrator import GroupIntegratorState

ForceFn = Callable[[ArrayLike, float], NDArray[np.complex128]]

@dataclasses.dataclass
class SymplecticConfig:
    dt: float = 0.01
    gamma: float = 0.5  # Damping (Constant)
    mass: float = 1.0   # Riemannian Mass
    clip_norm: float = 10.0 # Gradient clipping

class SymplecticLieEuler:
    def __init__(self, force_fn: ForceFn, config: SymplecticConfig):
        self.force_fn = force_fn
        self.config = config
        self._step_count = 0
        
    def step(self, state: GroupIntegratorState) -> GroupIntegratorState:
        """
        Symplectic Euler Step on SL(2,R) x UHP.
        
        Updates:
        1. Momentum (Map): m_{k+1} = (1 - gamma*dt) * m_k + dt * Torque(z_k)
        2. Group (Action): g_{k+1} = g_k * exp(dt * I^{-1} m_{k+1})
        """
        cfg = self.config
        dt = cfg.dt
        
        # 1. Compute Torque at current position
        z_curr = state.z_uhp
        forces = self.force_fn(z_curr, state.action)
        use_batch = (state.xi.ndim == 2)
        torque = aggregate_torque(z_curr, forces, sum_torque=not use_batch)
        
        # Gradient Clipping (Optional but recommended for stiff potentials)
        t_norm = np.linalg.norm(torque)
        if t_norm > cfg.clip_norm:
            torque = torque * (cfg.clip_norm / t_norm)
        
        # 2. Update Momentum (Semi-implicit Euler / Symplectic Euler)
        # Apply damping: m = m - gamma * m * dt
        # Apply torque: m = m + torque * dt
        # Combined: m_new = m_old * (1 - gamma*dt) + torque * dt
        damping_factor = max(0.0, 1.0 - cfg.gamma * dt)
        
        # Initial momentum?
        if state.m is None:
            # Need h to compute I? Or assume Riemannian constant?
            # For optimization, we assume Riemannian Inertia usually.
            # Let's compute I just in case.
            h = state.h # lazy eval
            I = riemannian_inertia(h, cfg.mass)
            m_curr = apply_inertia(I, state.xi)
        else:
            m_curr = state.m
            
        m_new = m_curr * damping_factor + torque * dt
        
        # 3. Update Position (via Group)
        # xi = I^{-1} m_new
        # We need I at... which point?
        # Symplectic Euler: Use new momentum to update position.
        # But I depends on position (in Rigid Body).
        # In Riemannian Particle model, I is constant in body frame!
        # So we can use current I (or even constant I).
        
        # Optimization: Assume Riemannian Inertia (Constant in body frame)
        # I = diag(m, m, m)
        # xi = m_new / mass
        xi_new = m_new / cfg.mass
        
        # Update Group
        g_step = exp_sl2(xi_new, dt)
        G_new = state.G @ g_step
        
        # Renormalize occasionally
        if self._step_count % 100 == 0:
            G_new = self._renormalize(G_new)
            
        self._step_count += 1
        
        return GroupIntegratorState(
            G=G_new,
            z0_uhp=state.z0_uhp,
            xi=xi_new,
            m=m_new,
            action=state.action, # Not updated for now
            dt_last=dt,
            gamma_last=cfg.gamma
        )

    def _renormalize(self, G):
        # reuse renormalization from group_integrator logic
        det = G[..., 0, 0] * G[..., 1, 1] - G[..., 0, 1] * G[..., 1, 0]
        mask_neg = det < 0
        if np.any(mask_neg):
            G[mask_neg, 0, 0] *= -1.0
            G[mask_neg, 1, 0] *= -1.0
            det[mask_neg] *= -1.0
        
        mask_scale = np.abs(det - 1.0) > 1e-10
        if np.any(mask_scale):
            scale = 1.0 / np.sqrt(det[mask_scale] + 1e-15)
            G_sub = G[mask_scale]
            G_sub *= scale[:, np.newaxis, np.newaxis]
            G[mask_scale] = G_sub
        return G

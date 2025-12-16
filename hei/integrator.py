"""Contact splitting integrator for CCD on the upper half-plane."""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .diamond import aggregate_torque
from .inertia import apply_inertia, invert_inertia
from .lie import exp_sl2, mobius_action_matrix, matrix_to_vec, vec_to_matrix
from .geometry import cayley_uhp_to_disk, cayley_disk_to_uhp, clamp_disk_radius, clamp_uhp
from .metrics import poincare_distance_disk


ForceFn = Callable[[ArrayLike, float], NDArray[np.complex128]]
GammaFn = Callable[[ArrayLike], float]
PotentialFn = Callable[[ArrayLike, float], float]


@dataclasses.dataclass
class IntegratorState:
    z_uhp: NDArray[np.complex128]
    xi: NDArray[np.float64]  # (u, v, w)
    I: NDArray[np.float64]  # 3x3 inertia
    m: NDArray[np.float64] | None = None  # momentum m = I xi
    action: float = 0.0  # discrete contact variable Z_k
    dt_last: float = 0.0
    gamma_last: float = 0.0


@dataclasses.dataclass
class IntegratorConfig:
    eps_disp: float = 1e-2  # max geometric displacement per step
    max_dt: float = 1e-3
    min_dt: float = 1e-6
    fixed_point_iters: int = 2
    v_floor: float = 1e-6  # floor for |xi| to avoid exploding dt
    torque_dt_scale: float = 1e-4  # weight on torque norm in adaptive dt denominator
    xi_clip: float = 200.0  # cap on |xi| to avoid blow-up
    step_clip: float = 5.0  # cap on |xi|*dt to avoid huge exp
    relax_eta: float = 0.0  # structural relaxation on -∇V; 0 disables to avoid drift by default
    dt_damping_safety: float = 1.5  # allow h*gamma up to ~dt_damping_safety for semi-implicit stability


class ContactSplittingIntegrator:
    """
    Discrete contact variational integrator (DCVI) with implicit fixed-point solve.

    Implementation follows a semi-implicit update:
        I(a_k) xi_k = alpha_{k-1} Ad^*_{exp(-h xi_{k-1})}(I(a_{k-1}) xi_{k-1}) + h * torque(a_k)
        Z_{k+1} = alpha_k Z_k + h/(1 + h gamma_k / 2) * (T_k - V_k)
        z_{k+1} = exp(h xi_k) ⋅ z_k  + h * (-eta ∇V)   # structural relaxation (non-rigid)

    torques are geometric (diamond) forces; scaling is dt-linear and consistent
    with the residual diagnostics.
    """

    def __init__(
        self,
        force_fn: ForceFn,
        potential_fn: PotentialFn,
        gamma_fn: GammaFn,
        config: IntegratorConfig | None = None,
    ) -> None:
        self.force_fn = force_fn
        self.potential_fn = potential_fn
        self.gamma_fn = gamma_fn
        self.config = config or IntegratorConfig()

    def _coadjoint_transport(self, xi: ArrayLike, m: ArrayLike, dt: float) -> NDArray[np.float64]:
        """
        Exact finite coadjoint transport Ad^*_{exp(-dt xi)} m via matrix conjugation.
        """
        g = exp_sl2(xi, dt)  # exp(dt xi)
        g_inv = np.linalg.inv(g)
        M = vec_to_matrix(m)
        M_new = g_inv @ M @ g  # Ad^*_{g} under trace pairing
        return matrix_to_vec(M_new)

    def coadjoint_transport(self, xi: ArrayLike, m: ArrayLike, dt: float) -> NDArray[np.float64]:
        """
        Public wrapper for coadjoint transport used by diagnostics to match integrator logic.
        """
        return self._coadjoint_transport(xi, m, dt)

    def _alpha(self, gamma_val: ArrayLike, dt: float) -> ArrayLike:
        # Exponential decay for damping to improve stability under strong gamma
        return np.exp(-dt * np.asarray(gamma_val, dtype=float))

    def _max_hyperbolic_disp(self, z_old: NDArray[np.complex128], z_new: NDArray[np.complex128]) -> float:
        """Maximum hyperbolic distance between two position sets (elementwise paired)."""
        disk_old = cayley_uhp_to_disk(z_old)
        disk_new = cayley_uhp_to_disk(z_new)
        max_d = 0.0
        for a, b in zip(disk_old.flat, disk_new.flat):
            d = poincare_distance_disk(a, b)
            if d > max_d:
                max_d = d
        return max_d

    def _stabilize_uhp(self, z: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Project points to a safe region: disk clamp -> back to UHP -> clamp imag/real.
        """
        disk = clamp_disk_radius(cayley_uhp_to_disk(z))
        z_safe = cayley_disk_to_uhp(disk)
        z_safe = clamp_uhp(z_safe)
        return np.nan_to_num(z_safe, nan=0.0)

    def _backtrack_dt(self, state: IntegratorState, torque: ArrayLike, xi: ArrayLike) -> tuple[float, NDArray[np.complex128]]:
        """
        Choose dt using strict First-Principles constraints:
        1. Displacement limit (Kinematic constraint)
        2. Damping stability (Thermodynamic constraint): h * gamma < safety
        3. Spectral stability: account for inertia spectrum when strong torques push acceleration
        """
        xi_norm = float(np.linalg.norm(xi))
        torque_norm = float(np.linalg.norm(torque))
        # spectral leverage from inertia: ||I^{-1}|| ~ 1 / eig_min(I)
        vals = np.linalg.eigvalsh(state.I)
        eig_min = float(max(vals.min(), 1e-6))
        inv_spec = 1.0 / eig_min
        # blend velocity and acceleration scales
        denom = max(1e-9, xi_norm + self.config.torque_dt_scale * inv_spec * torque_norm)
        dt_geo = self.config.eps_disp / denom
        gamma_curr = state.gamma_last if state.gamma_last > 0 else 1.0
        # allow semi-implicit scheme to tolerate larger h*gamma before cutting dt
        SAFETY_FACTOR = self.config.dt_damping_safety
        dt_thermo = SAFETY_FACTOR / (gamma_curr + 1e-9)
        dt = min(self.config.max_dt, dt_geo, dt_thermo)
        dt = max(dt, self.config.min_dt)
        z_trial = state.z_uhp
        for _ in range(5):
            g_step = exp_sl2(xi, dt)
            z_trial = mobius_action_matrix(g_step, state.z_uhp)
            z_trial = self._stabilize_uhp(z_trial)
            disp = self._max_hyperbolic_disp(state.z_uhp, z_trial)
            if np.isnan(disp) or np.isinf(disp) or disp > self.config.eps_disp * 1.5:
                dt *= 0.5
                if dt < self.config.min_dt:
                    dt = self.config.min_dt
                    break
            else:
                break
        return dt, z_trial

    def step(self, state: IntegratorState) -> IntegratorState:
        # emergency brake to avoid runaway velocities causing numerical blow-up
        xi_mag = float(np.linalg.norm(state.xi))
        if xi_mag > 1e4:
            state.xi = state.xi * 0.01
            state.m = apply_inertia(state.I, state.xi)

        gamma_field_prev = self.gamma_fn(state.z_uhp)
        gamma_prev = float(np.mean(gamma_field_prev)) if np.asarray(gamma_field_prev).size else 0.0
        xi_prev = state.xi
        m_prev = state.m if state.m is not None else apply_inertia(state.I, xi_prev)

        z_guess = state.z_uhp
        action_guess = state.action
        xi_new = xi_prev
        m_new = m_prev

        # determine step size once per step using initial torque guess
        forces0 = self.force_fn(z_guess, action_guess)
        torque0 = aggregate_torque(z_guess, forces0)
        dt, z_guess = self._backtrack_dt(state, torque0, xi_prev)
        alpha_prev = self._alpha(gamma_prev, dt)

        for _ in range(max(1, self.config.fixed_point_iters)):
            gamma_field = self.gamma_fn(z_guess)
            gamma_eff = float(np.mean(gamma_field)) if np.asarray(gamma_field).size else 0.0
            alpha_prev = self._alpha(gamma_eff, dt)

            forces = self.force_fn(z_guess, action_guess)
            if np.ndim(gamma_field) > 0:
                forces = forces / (1.0 + 0.5 * dt * gamma_field)
            torque = aggregate_torque(z_guess, forces)
            m_advected = self._coadjoint_transport(xi_prev, m_prev, dt)
            m_new = alpha_prev * m_advected + dt * torque
            # refresh inertia and gamma with updated geometry
            from .inertia import locked_inertia_uhp
            I_new = locked_inertia_uhp(z_guess)
            state.I = I_new
            gamma_curr = gamma_eff
            xi_new = invert_inertia(I_new, m_new)
            # limit per-step speed based on current dt to avoid huge jumps
            max_xi_norm = 10.0 / max(dt, 1e-8)
            xi_norm_val = float(np.linalg.norm(xi_new))
            if xi_norm_val > max_xi_norm:
                xi_new = xi_new * (max_xi_norm / xi_norm_val)
                m_new = apply_inertia(I_new, xi_new)
            # hard velocity cap to keep exp_sl2 numerically stable
            xi_norm_val = float(np.linalg.norm(xi_new))
            if xi_norm_val > self.config.xi_clip:
                xi_new = xi_new * (self.config.xi_clip / xi_norm_val)
                m_new = apply_inertia(I_new, xi_new)
            # clip step magnitude
            step_norm = float(np.linalg.norm(xi_new)) * dt
            if step_norm > self.config.step_clip:
                scale = self.config.step_clip / max(step_norm, 1e-8)
                xi_new = xi_new * scale
                m_new = apply_inertia(I_new, xi_new)

            # Update positions via finite Möbius action exp(dt xi_new)
            g_step = exp_sl2(xi_new, dt)
            z_guess = mobius_action_matrix(g_step, state.z_uhp)
            z_guess = self._stabilize_uhp(z_guess)
            # Structural relaxation: allow non-rigid drift along -∇V
            if self.config.relax_eta > 0.0:
                relax_force = self.force_fn(z_guess, action_guess)
                # hyperbolic metric factor ~ 1 / y^2 to avoid overshoot near boundary
                y = np.maximum(np.imag(z_guess), 1e-6)
                relax_step = (self.config.relax_eta * dt) * (relax_force * (y * y))
                z_relaxed = z_guess + relax_step
                z_guess = self._stabilize_uhp(z_relaxed)

            # Contact action update (using current guess)
            T = 0.5 * float(xi_new @ (state.I @ xi_new))
            V = float(self.potential_fn(z_guess, action_guess))
            denom = 1.0 + 0.5 * dt * gamma_curr
            action_guess = alpha_prev * state.action + dt / denom * (T - V)

        m_out = apply_inertia(state.I, xi_new)
        return IntegratorState(
            z_uhp=z_guess,
            xi=xi_new,
            I=state.I,
            m=m_out,
            action=action_guess,
            dt_last=dt,
            gamma_last=gamma_curr,
        )

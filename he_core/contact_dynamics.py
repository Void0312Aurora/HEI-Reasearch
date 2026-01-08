import torch
import torch.nn as nn
import math
import os
from typing import Callable
from he_core.state import ContactState

from he_core.integrators.group_integrator import GroupContactIntegrator

class ContactIntegrator(nn.Module):
    """
    Implements Contact Hamiltonian Dynamics.
    Equations:
    dot_q = dH/dp
    dot_p = -(dH/dq + p * dH/ds)
    dot_s = p * dH/dp - H
    """
    def __init__(
        self,
        method: str = 'euler',
        dim_q: int | None = None,
        damping: float = 0.1,
        gamma_clip: float = 0.0,
        substeps: int = 1,
    ):
        super().__init__()
        self.method = str(method or "euler")
        self.group_integrator = None
        self.gamma_clip = float(gamma_clip)
        self.substeps = int(substeps or 1)
        if self.substeps < 1:
            raise ValueError("substeps must be >= 1")

        if self.method == 'group':
            if dim_q is None:
                raise ValueError("dim_q must be provided for group integrator")
            self.group_integrator = GroupContactIntegrator(dim_q, damping)
        
    def step(self, state: ContactState, generator: Callable[[ContactState], torch.Tensor], dt: float = 0.1) -> ContactState:
        """
        Perform one step of integration.
        Returns NEW state (out-of-place).
        """
        dt = float(dt)
        if self.method == 'group':
            return self._step_group(state, generator, dt)

        # Note: group mode already sub-steps internally; for vector-space methods we can
        # optionally apply uniform sub-stepping for stiffness control.
        if self.substeps > 1:
            dt_sub = dt / float(self.substeps)
            cur = state
            for _ in range(self.substeps):
                cur = self._step_once(cur, generator, dt_sub)
            return cur

        return self._step_once(state, generator, dt)

    def _step_once(self, state: ContactState, generator: Callable[[ContactState], torch.Tensor], dt: float) -> ContactState:
        if self.method in ("semi", "semi_implicit", "contact_si", "contact_semi_implicit"):
            return self._step_semi_implicit(state, generator, dt)
        return self._step_euler(state, generator, dt)

    def _ensure_grad_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            return x
        if x.is_leaf:
            return x.requires_grad_(True)
        return x.detach().requires_grad_(True)

    def _compute_h_and_grads(
        self,
        state: ContactState,
        generator: Callable[[ContactState], torch.Tensor],
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, ContactState]:
        x0 = state.flat if create_graph else state.flat.detach()
        x0 = self._ensure_grad_tensor(x0)
        # Important: build any view ops (ContactState uses `.view`) under grad-enabled mode.
        # If a view is created under `torch.no_grad()`, autograd will later treat it as a leaf
        # without history, and `autograd.grad` can return None (zero dynamics) even if we
        # re-enable grad inside this function.
        with torch.enable_grad():
            s0 = ContactState(state.dim_q, state.batch_size, state.device, x0)
            H = generator(s0)
            grads = torch.autograd.grad(
                H.sum(),
                x0,
                create_graph=create_graph,
                allow_unused=True,
            )[0]
        if grads is None:
            grads = torch.zeros_like(x0)
        if not create_graph:
            H = H.detach()
            grads = grads.detach()
        return H, grads, s0

    def _step_euler(self, state: ContactState, generator: Callable[[ContactState], torch.Tensor], dt: float) -> ContactState:
        need_grad = torch.is_grad_enabled()
        H, grads, s0 = self._compute_h_and_grads(state, generator, create_graph=need_grad)
        d = state.dim_q
        dH_dq = grads[:, :d]
        dH_dp = grads[:, d : 2 * d]
        dH_ds = grads[:, 2 * d :]

        q, p = s0.q, s0.p
        dot_q = dH_dp
        dot_p = -(dH_dq + p * dH_ds)
        dot_s = (p * dH_dp).sum(dim=1, keepdim=True) - H

        new_flat = s0.flat + torch.cat([dot_q, dot_p, dot_s], dim=1) * dt
        if not need_grad:
            new_flat = new_flat.detach()
        return ContactState(state.dim_q, state.batch_size, state.device, new_flat)

    def _step_semi_implicit(self, state: ContactState, generator: Callable[[ContactState], torch.Tensor], dt: float) -> ContactState:
        """
        Semi-implicit contact step (vector-space, O(B·D)).

        Uses an integrating-factor update for `p` to improve stability for stiff
        dissipation: dp/dt = -dH/dq - (dH/ds)·p.

        For q update, we approximate dot_q = dH/dp by estimating a per-sample
        scalar `a(q)` such that dH/dp ≈ a(q)·p (exact for K(p,q)=0.5·a(q)||p||^2).
        """
        need_grad = torch.is_grad_enabled()
        H, grads, s0 = self._compute_h_and_grads(state, generator, create_graph=need_grad)
        d = state.dim_q
        dH_dq = grads[:, :d]
        dH_dp = grads[:, d : 2 * d]
        dH_ds = grads[:, 2 * d :]

        q, p, s = s0.q, s0.p, s0.s

        gamma = dH_ds
        if self.gamma_clip > 0:
            # Optional last-resort clamp (changes dynamics); keep disabled for theory runs.
            gamma = torch.clamp(gamma, min=-self.gamma_clip, max=self.gamma_clip)

        # Guard against exp overflow when gamma is very negative (strong energy pumping).
        # This is a *numerics* check: if it trips, the dynamics/params are already unstable.
        neg_a_dt = -gamma * dt
        torch._assert_async((neg_a_dt <= 80.0).all(), "semi integrator: exp overflow risk (gamma too negative or dt too large)")

        exp_term = torch.exp(neg_a_dt)  # exp(-gamma dt)
        one_minus_exp = -torch.expm1(neg_a_dt)  # 1 - exp(-gamma dt), stable for small dt
        phi = torch.where(gamma.abs() < 1e-6, torch.full_like(gamma, dt), one_minus_exp / gamma)

        # Exact for dp/dt = F - gamma p with frozen F, gamma.
        force = -dH_dq
        p_next = exp_term * p + phi * force

        # Estimate a(q) in dH/dp ≈ a(q) p.
        p_norm_sq = (p * p).sum(dim=1, keepdim=True)
        pdH_dp = (p * dH_dp).sum(dim=1, keepdim=True)
        a_est = torch.where(p_norm_sq > 1e-8, pdH_dp / (p_norm_sq + 1e-8), torch.ones_like(p_norm_sq))

        # Integrate q using the exact integral of the linearized system:
        # q_dot = a_est * p,   p_dot = force - gamma * p  (with frozen a_est, force, gamma)
        psi = torch.where(
            gamma.abs() < 1e-6,
            torch.full_like(gamma, 0.5 * dt * dt),
            (dt - phi) / gamma,
        )
        q_next = q + a_est * (phi * p + psi * force)

        # dot_s = p·dH/dp - H, evaluated at updated (q,p) but current s for stability.
        pdH_dp_next = a_est * (p_next * p_next).sum(dim=1, keepdim=True)
        temp_flat = torch.cat([q_next, p_next, s], dim=1)
        temp_state = ContactState(state.dim_q, state.batch_size, state.device, temp_flat)
        with torch.enable_grad():
            H_tilde = generator(temp_state)
        if not need_grad:
            H_tilde = H_tilde.detach()
        s_next = s + dt * (pdH_dp_next - H_tilde)

        new_flat = torch.cat([q_next, p_next, s_next], dim=1)
        if not need_grad:
            new_flat = new_flat.detach()
        return ContactState(state.dim_q, state.batch_size, state.device, new_flat)

    def _step_group(self, state: ContactState, generator: Callable[[ContactState], torch.Tensor], dt: float = 0.1) -> ContactState:
        """
        Step using GroupContactIntegrator.
        Maps flat state -> Group state -> Step -> Flat state.

        Performance note: We compute grad(H) ONCE per dt (frozen-force), then
        subdivide the integration into safe micro-steps purely for numerical
        stability of the exp-map. This dramatically reduces autograd overhead
        while keeping the dynamics stable.
        """
        d = state.dim_q
        device = state.flat.device
        batch_size = state.batch_size

        # If the caller disabled grad (e.g., speculative rollout), we must NOT build
        # a giant autograd graph through micro-steps. Still, we need local dH/dx to
        # advance the dynamics. We compute grads with create_graph=False in that case.
        need_grad = torch.is_grad_enabled()

        # Initialize group state from current flat state.
        # As in `_compute_h_and_grads`, ensure ContactState view ops are created under
        # grad-enabled mode so `autograd.grad` can see the dependency on `x0` even when the
        # outer caller is in `torch.no_grad()`.
        x0 = state.flat if need_grad else state.flat.detach()
        if not x0.requires_grad:
            x0.requires_grad_(True)
        with torch.enable_grad():
            s0 = ContactState(d, batch_size, state.device, x0)
            state_group = self.group_integrator.flat_to_group(s0.q, s0.p)
            state_group.z = s0.s

        debug_nan = os.getenv("HEI_DEBUG_NAN", "0") == "1"
        max_step_rapidity = float(getattr(self.group_integrator, 'max_step_rapidity', 50.0))

        # ====== Compute H and gradients ONCE at the start of dt ======
        with torch.enable_grad():
            H = generator(s0)
            if debug_nan and (not torch.isfinite(H).all()):
                self._raise_nonfinite("contact_dynamics.H", H)
            grads = torch.autograd.grad(
                H.sum(),
                x0,
                create_graph=need_grad,
                allow_unused=True,
            )[0]
            if grads is None:
                grads = torch.zeros_like(x0)

        if not need_grad:
            # Break any accidental graph retention (we only wanted numeric grads here).
            H = H.detach()
            grads = grads.detach()

        if debug_nan and (not torch.isfinite(grads).all()):
            self._raise_nonfinite("contact_dynamics.grads", grads)

        dH_dq = grads[:, :d]
        dH_dp = grads[:, d:2 * d]
        dH_ds = grads[:, 2 * d:]

        force = -dH_dq
        pdH_dp = (s0.p * dH_dp).sum(dim=1, keepdim=True)

        # Determine number of micro-steps based on velocity norm
        v_norm = torch.norm(dH_dp, dim=1)
        v_max = float(v_norm.max().item()) if v_norm.numel() > 0 else 0.0
        if not math.isfinite(v_max):
            raise RuntimeError("ContactIntegrator(group): non-finite ||dH/dp|| detected.")
        if v_max <= 0.0:
            n_micro = 1
        else:
            n_micro = max(1, int(math.ceil(dt * v_max / max_step_rapidity)))
        dt_sub = dt / n_micro

        # Build force algebra once
        force_algebra = torch.zeros(batch_size, d + 1, d + 1, device=device, dtype=x0.dtype)
        force_algebra[:, 0, 1:] = force
        force_algebra[:, 1:, 0] = force

        # ====== Micro-step loop ======
        # If need_grad=False, explicitly disable grad to prevent building graphs through exp-map.
        if need_grad:
            for _ in range(n_micro):
                state_group = self.group_integrator.step(
                    state_group,
                    force_algebra,
                    dt_sub,
                    H_val=H,
                    dH_ds=dH_ds,
                    pdH_dp=pdH_dp,
                    v_override=None,
                    assume_safe_dt=True,
                )
        else:
            with torch.no_grad():
                for _ in range(n_micro):
                    state_group = self.group_integrator.step(
                        state_group,
                        force_algebra,
                        dt_sub,
                        H_val=H,
                        dH_ds=dH_ds,
                        pdH_dp=pdH_dp,
                        v_override=None,
                        assume_safe_dt=True,
                    )

        if debug_nan:
            q_chk, p_chk = self.group_integrator.group_to_flat(state_group)
            if not torch.isfinite(q_chk).all():
                self._raise_nonfinite("contact_dynamics.q_next", q_chk)
            if not torch.isfinite(p_chk).all():
                self._raise_nonfinite("contact_dynamics.p_next", p_chk)
            if not torch.isfinite(state_group.z).all():
                self._raise_nonfinite("contact_dynamics.s_next", state_group.z)

        q_new, p_new = self.group_integrator.group_to_flat(state_group)
        s_new = state_group.z
        new_flat = torch.cat([q_new, p_new, s_new], dim=1)
        if not need_grad:
            new_flat = new_flat.detach()
        return ContactState(state.dim_q, batch_size, state.device, new_flat)

    @staticmethod
    def _raise_nonfinite(tag: str, tensor: torch.Tensor):
        t = tensor.detach()
        stats = {"shape": tuple(t.shape), "dtype": str(t.dtype), "device": str(t.device)}
        try:
            stats["max_abs"] = float(t.abs().max().item())
            stats["min"] = float(t.min().item())
            stats["max"] = float(t.max().item())
        except Exception:
            pass
        raise RuntimeError(f"NaN/Inf detected at {tag}: {stats}")

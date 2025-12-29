# SPEC v0 (Frozen)

**Status**: Frozen for Iteration 0
**Date**: 2025-12-29

## 1. Object Dictionary (Log Contract v0)

All implementations must log these fields with the exact names below.

| Field | Symbol | Definition |
| :--- | :--- | :--- |
| `x_int` | $x_{int}$ | **Internal State**. The "kernel state" (e.g., $q, p, s$ or latent $z$). Must be accessible for diagnostics. |
| `x_blanket` | $x_{blanket}$ | **Blanket State**. State used for read/write, perception, action, and memory write-back. Can be explicit or derived. |
| `x_ext_proxy` | $x_{ext}$ | **External Proxy**. Environment state or its proxy. Used for A1/A2 control comparison. |
| `u_t` | $u(t)$ | **Port Input**. The total input to the kernel. Must support decomposition into `u_env` (from environment) and `u_self` (from self-loop/replay). |
| `step_meta` | - | **Step Metadata**. Must include `phase` (online/offline), `u_source` (env/replay/imagine), and global `step` count. |

## 2. Step Semantics (One-step Map v0)

A single "step" consists of the following operator sequence, strictly ordered:

1.  **Kernel Propagation**: $x_{int}^{t+1} = \text{Kernel}(x_{int}^t, u^t)$
2.  **Readout**: $y^t = \text{Readout}(x_{int}^{t+1})$
3.  **Sampling/Deciding**: $a^t, \text{flags} = \text{Policy}(y^t)$
4.  **Write-back**: update short-term memory or generate $u_{self}^{t+1}$
5.  **Routing/Gating**: Combine $u_{env}^{t+1}$ and $u_{self}^{t+1}$ to form $u^{t+1}$

*Constraint*: Every step must be deterministic given the state and random seed.

## 3. Evidence Ladder (Evidence v0)

All diagnostic conclusions must be tagged with an evidence level.

*   **E2 (Strong)**: Control group + Statistically Significant + Reproducible (consistent across seeds).
*   **E1 (Medium)**: Control group + Clear Effect, but statistics limited or n=1.
*   **E0 (Weak)**: Visual inspection or single case. Only for generating hypotheses.

**Target**: core conclusions (D1, D3) must reach **E1** in Iteration 0.

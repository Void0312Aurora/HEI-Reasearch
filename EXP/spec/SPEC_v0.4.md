# SPEC v0.4: Situated Entity & Evidence Contracts

## 1. Object Model: `UnifiedGeometricEntity`
The Entity is a container for the "Soul" dynamics and its interactions.

### 1.1 Components
*   **State**: $x \in M = T^*Q \times \mathbb{R}$ (Contact Manifold). Managed by `ContactState`.
*   **Dynamics (L1)**: `PortCoupledGenerator`.
    *   $H(x, u) = H_{int}(x) + \langle u, B(q) \rangle$.
    *   $B(q) = Wq$ (Learnable Linear Map).
*   **Structure (L2)**: `Atlas` & `Router`.
    *   $K$ Charts.
    *   Router $\mathcal{R}(q) \to w \in \Delta^{K-1}$ (Learnable Gating).
*   **Integrator**: `ContactIntegrator` (Symplectic/Contact Geometric Integrator).
*   **Interface**: `ActionInterface` / `AuxInterface` (Projections).

### 1.2 Trainable Parameters ($\theta$)
*   `generator.coupling.W`: Port Shape.
*   `atlas.router.net`: Skill Gating.
*   (Future) `atlas.transitions`: L2 Transport.

---

## 2. Logic Specifications (The Gates)
All Experiments and Training Loops must respect these Gates.

### 2.1 Protocol 1: Spectral Gap (Structure)
*   **Definition**: Gap between Real parts of eigenvalues of Jacobian $J$.
*   **Spec**: `max_gap > 5.0` (for Stiff/Fast-Slow systems).
*   **Status**: `PASS` if gap > Threshold, else `FAIL`.
*   **Invalid**: If spectrum calculation fails or unstable.

### 2.2 Protocol 3: Persistent Slow Modes (LTM)
*   **Definition**: Count of DMD eigenvalues with $|\lambda| \in [0.99, 1.01]$.
*   **Spec**: `slow_mode_count >= 1`.
*   **Status**: `PASS` if distinct slow modes exist.
*   **Invalid**: If trajectory diverges or DMD ill-conditioned.

### 2.3 Protocol 5: Port Gain (ISS/Stability)
*   **Definition**: Ratio $\frac{||y - y'||}{||u - u'||}$ under perturbation.
*   **Spec**: `gain < 2.0` (Small Gain Theorem proxy).
*   **Status**: `PASS` if gain bounded.
*   **Critilal FAIL**: If gain > 10.0 (Explosion).

### 2.4 Protocol A3: Consistency (Self-Supervision)
*   **Definition**: Drift of Potential $V$ (or $H$) in Offline Mode ($u=0$).
*   **Spec**:
    *   Offline/Dissipative: $\dot V \le 0$ (Negative Drift).
    *   Driven/Replay: Bounded $|V| < B$.
*   **Status**: `PASS` if `DISSIPATIVE` or `STABLE`.
*   **FAIL**: `RANDOM_WALK` (Zero drift, High Variance) or `RUNAWAY` (Positive Drift).

## 3. Reporting Schema (`report.json`)
Every experiment output must conform to:
```json
{
  "entity_config": {...},
  "metrics": {
    "spectral_gap": float,
    "port_gain": float,
    "dmd_slow_count": int,
    "consistency_drift": float
  },
  "gates": {
    "protocol_1": "PASS/FAIL/INVALID",
    "protocol_3": "PASS/FAIL/INVALID",
    "protocol_5": "PASS/FAIL/INVALID",
    "protocol_a3": "PASS/FAIL/INVALID"
  },
  "overall_status": "READY" | "BLOCKED"
}
```

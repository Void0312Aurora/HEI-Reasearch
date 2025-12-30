# Research Object Specification v0.2 (Entity)
Date: 2025-12-30
Status: Frozen

This document defines the interface, state space, and validation contract for the "HEI Entity v0.2". Any deviation from this spec invalidates experimental results.

## 1. Object Dictionary
The Entity consists of the tuple $(x_{int}, x_{blanket}, \Theta, \Phi)$.

### 1.1 State Tensors
| Tensor | Name | Shape (Batch, Dim) | Description | Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `x_int` | Internal | $(1, d_{int})$ | The private, causal dynamical state. | Hidden from Env. Must have finite energy. |
| `x_blanket` | Blanket | $(1, d_b)$ | The interface boundary. Concatenation of Sensation and Action. | Split into [Sensory, Active]. |
| `sensory` ($u_{env}$) | Sensory Port | $(1, d_q)$ | Input from Environment (World State or Sensation). | Read-Only for Entity (mostly). |
| `active` ($u_{self}$) | Active Port | $(1, d_q)$ | Output to Environment (Action/Intervention). | Write-Only for Entity. |

**Relation**: $x_{blanket} = [sensory, active]$.

### 1.2 One-Step Map
The canonical update cycle $T: S \times U \to S \times A$ is:
1.  **Perception**: Read `sensory` ($u_{env}$) from `Env` or `Buffer`.
2.  **Readout**: Compute `active` ($u_{self}$) and Prediction ($pred\_x$) from `x_int`.
    *   $u_{self} = \pi(x_{int}, u_{env})$
    *   $pred\_x = g(x_{int})$
3.  **Dynamics**: Evolve `x_int`.
    *   $u_{total} = \sigma(u_{env}, u_{self}, phase)$ (Scheduler/Gating)
    *   $x_{int}^{t+1} = K(x_{int}^t, u_{total})$
4.  **Write-back**: Update `x_blanket` with new [Sensory, Active].

## 2. Evidence Ladder (Validation Contract)

### E0: Existence (Smoke Test)
-   Can initialize.
-   Can step without NaN.
-   Satisfies Energy Bounds ($|x| < C$).

### E1: Minimal Structure (A1 Weak + A3 Weak)
-   **A1 (Blanket)**: $x_{int}$ dynamics are distinct from purely external dynamics.
-   **A3 (Potential)**: Learning ($\dot{\theta}$) correlates with Error reduction ($\dot{F} < 0$).

### E2: Causal Structure (A1 Strong + A4 Identity)
-   **A1 (Independence)**: $I(x_{int}; x_{ext} | x_{blanket})$ is minimized.
-   **A4 (Identity)**: Perturbation recovery to Attractor Fingerprint.

## 3. Log Contract (Minimal Audit Fields)
Every experiment MUST produce a log (CSV/JSON/Parquet) containing at least:

```json
{
  "step": "int (monotonic)",
  "phase": "string ('online'|'offline')",
  "x_int_sample": "float[] (subset or projection, e.g. first 2 dims)",
  "sensory": "float[] (u_env)",
  "active": "float[] (u_self)",
  "pred_error": "float[] (u_env - pred_x)",
  "pred_x": "float[] (internal prediction)",
  "energy_int": "float (scalar summary of internal energy)",
  "meta_seed": "int"
}
```

**Validation Rule**: If any field is missing or constant zero (unexpectedly), the run is **INVALID**.

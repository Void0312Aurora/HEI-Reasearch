# Aurora Roadmap: Gap Analysis & Implementation Plan

**Based on**: `理论基础-4.md`, `Aim.md`, `hei_n/integrator_n.py`
**Date**: 2025-12-22

---

## 1. Gap Analysis: Theory vs Code

We compared the Strict Axiomatic Theory (Theory 4) with the current `hei_n` implementation.

| Component | Theory 4 Specification | Current `hei_n` Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Manifold** | **Contact Manifold** ($M = T^*Q \times \mathbb{R}_z$) | **Symplectic Manifold** ($M = T^*Q$) | **CRITICAL GAP** |
| **State** | $(\xi, a, z)$ (Mind, Structure, Action) | $(M, G, \text{None})$ | **Missing z** |
| **Inertia** | **Variable** $\mathbb{I}(a)$ (Definition 2.3) | **Constant** $I = Id$ (Riemannian) | **CRITICAL GAP** |
| **Damping** | **Constant** $\gamma$ (Schur Lemma, Axiom 4.2) | Constant $\gamma$ | **Compliant** |
| **Dynamics** | **Dissipative Euler-Poincaré** (Theorem 3.2) | Lie-Poisson (Simple Damping) | **Partial** |

**Conclusion**:
The current `hei_n` is a high-quality **Riemannian Optimizer** but lacks the **Thermodynamic Drive** ($z$-variable) and **Structural Feedback** (Variable Inertia) required for a "Living Entity". It cannot "feel" surprise or "resist" change based on structure.

---

## 2. The Aurora Plan (Implementation Roadmap)

To realize the vision of "Aurora" (The Hyperbolic Embodied Entity), we must upgrade the engine to a full **Contact Cognitive Dynamics (CCD)** system.

### Phase 1: The Primordial Prototype (Kernel Upgrade)
**Goal**: Build a minimal "Living" Agent (HEI-Braitenberg).
**Tasks**:
1.  **Upgrade Integrator** (`src/hei_n/contact_integrator_n.py`):
    *   Add state variable `z` (Action/Surprise).
    *   Implement evolution $\dot{z} = L(\xi, a) - \langle p, \xi \rangle$? No, $\dot{z} = \langle p, \dot{q} \rangle - H$?
    *   Re-check Theory 3/4 for exact $\dot{z}$ equation. Usually $\dot{z} = \xi \cdot p - H$ in Contact Hamiltonian.
2.  **Implement Variable Inertia** (`src/hei_n/inertia_n.py`):
    *   Define $\mathbb{I}(a)$ such that "Cognitive Mass" increases with "Entrenchment" (e.g., node depth or connection density).
    *   Implement **Diamond Operator** for $SO(1, n)$.
3.  **Validation**:
    *   **Intent Persistence Test**: Can the agent maintain a goal-directed trajectory (Momentum $p$) after the sensory signal $S$ vanishes?

### Phase 2: Emergence of Language (Particle Big Bang)
**Goal**: Self-Organized Hierarchy on massive scale ($N=10^5$).
**Tasks**:
1.  **Scalability**: Implement **Hyperbolic Cutoff / Negative Sampling** ($O(N)$).
2.  **Environment**: Create a "Text Stream" potential where force comes from word co-occurrences.
3.  **Observation**: Verify "Spontaneous Hierarchy Emergence" (Axiom 4.3).

### Phase 3: Embodied Interaction
**Goal**: Sensorimotor Integration.
**Tasks**:
1.  **Coupling**: Connect the "Semantic Manifold" ($H^n$) with "Motor Manifold" ($\mathbb{R}^k$).
2.  **Active Inference**: Use Diamond Torque to drive motor actions that minimize prediction error.

---

## 3. Immediate Action Items

1.  **Create `src/hei_n/contact_integrator_n.py`**.
2.  **Create `src/hei_n/inertia_n.py`**.
3.  **Update `task.md`** to reflect the new "Aurora" phases.

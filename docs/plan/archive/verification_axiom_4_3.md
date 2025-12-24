# Supplementary Verification Plan: Axiom 4.3

**Objective**: Rigorously verify Axiom 4.3 (Energy Evolution & Hierarchy Emergence) and address gaps identified in `temp-01.md`.

## 1. Energy Evolution Verification (Quantitative)
**Goal**: Verify the dissipative Euler-Poincaré energy balance:
$$ \frac{dE}{dt} = -2 \mathcal{R}(\xi) $$
(Assuming $\mathcal{P}_{struct} = 0$ for now or explicitly calculating it).

**Experiment**: `experiments/verification_energy.py`
- **Setup**: Single particle or small N system with `RadialInertia`.
- **Metrics**:
  - $E(t) = T(\xi) + V(x)$
  - $\mathcal{R}(t) = \frac{1}{2} \gamma \langle \xi, \mathbb{I}(a) \xi \rangle$
  - $\Delta E_{actual} = E(t+\Delta t) - E(t)$
  - $\Delta E_{predicted} = -2 \mathcal{R}(t) \Delta t$
- **Success Criteria**: $\Delta E_{actual} \approx \Delta E_{predicted}$ (within numerical error).

## 2. Hierarchy Emergence Metrics
**Goal**: Quantify "tree-ness" of the emerging structure.
**Experiment**: Update `experiments/aurora_emergence.py` or new `experiments/verification_hierarchy.py`.
- **Metrics**:
  - **Ultrametricity Score**: For all triplets $(i, j, k)$, check $\delta$-hyperbolicity condition $d(x, y) \ge \min(d(x, z), d(y, z)) - \delta$.
  - **Tree Distortion**: Fit a tree (e.g., NJ or MST) to pairwise distances and compute distortion.
- **Success Criteria**: Metrics show significant improvement over random initialization.

## 3. Ablation Studies
**Goal**: Prove necessity of Contact Dynamics and Variable Inertia.
**Experiment**: `experiments/ablation_emergence.py`
- **Conditions**:
  1. **Baseline**: Contact + Variable Inertia + Potentials.
  2. **No Contact**: $\gamma = 0$ (Conservative). Expect: No convergence/Explosion.
  3. **No Variable Inertia**: $I = Identity$. Expect: Different structure? Slower convergence?
  4. **Potentials Only**: Is the structure just minimizing $V$?
     - Compare Final Structure of (1) vs (3).

## 4. Theoretical Alignment (Radial Inertia)
**Issue**: Radial Inertia introduces origin bias, contradicting Isometry Invariance.
**Action**:
- Acknowledge this in the report.
- Justify as "Embedded Observer" bias or check if "Relative Inertia" (distance to neighbors) is better?
- For now, stick to Radial but document the deviation.

## Implementation Steps
1. Create `tests/integration/test_energy_conservation.py` for Step 1.
2. Create `src/hei_n/metrics_n.py` for Step 2.
3. Update `experiments/aurora_emergence.py` to include Metrics and Ablation flags.

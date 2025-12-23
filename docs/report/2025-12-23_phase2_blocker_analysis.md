# Phase II Status Report: Volume Recovery vs. Alignment Stagnation
**Date**: 2025-12-23
**Focus**: Semantic Neighborhood Refinement & Numerical Stability

## Executive Summary
We have successfully resolved the critical "Density Disaster" (Volume Collapse) that plagued early Phase II tests. The system now maintains a healthy geometric volume ($R \approx 2.0$, $d_{nn} \approx 0.38$).

However, the **Semantic Alignment** phase faces a new blocker: **"Frozen Time"**.
To maintain numerical stability (Renorm $< 10^{-3}$) under stringent volume constraints, we were forced to reduce the time step $dt$ to $5 \times 10^{-5}$. This resulted in near-zero physical evolution ($T_{sim} \approx 0.25$ over 5000 steps), causing the semantic rank to stagnate ($P50 \approx 100k$) despite healthy geometry.

The path forward requires **trading compute for time**: significantly increasing solver precision to allow larger time steps ($dt \approx 2 \times 10^{-4}$), effectively buying "physical time" for angular sorting to occur.

---

## 1. Achievements: Resolving the "Density Disaster"

### The Problem
Initial runs showed the embedding space collapsing into a "black hole" ($R \approx 0.5$, $d_{nn} \approx 0.05$). In this "Crowded Phase", semantic forces ($k \approx 0.5$) were mathematically overwhelmed by the density, making angular sorting impossible.

### The Solution: Controlled Volume Recovery
We implemented a robust volume control stack:
1.  **De-collapse with G-Frame Repair**: One-shot kinematic scaling ($R \to 2.5x$) with Gram-Schmidt orthonormalization to fix the coordinate frame tangent space.
2.  **Soft Radius Anchor**: A potential $\frac{1}{2}\lambda(r - r_{tgt})^2$ with $\lambda=10.0$ to hold the volume against collapse without hard projection artifacts.
3.  **Short-Range Gated Repulsion**: High-force ($A=5.0$) repulsion active only at $d < 0.2$, preserving local separability without driving global drift.

### Result
| Metric | Collapse State (Fail) | Recovered State (Success) | Target |
| :--- | :--- | :--- | :--- |
| **KNN Top-1 Dist (P50)** | 0.056 | **0.377** | $> 0.15$ |
| **Mean Radius (R)** | 0.49 | **2.00** | $\sim 1.9$ |
| **Renorm Max** | $> 10^{-2}$ (Exploded) | **$1.5 \times 10^{-3}$** | $< 10^{-3}$ |

The system is now geometrically healthy and ready for alignment.

---

## 2. Current Blocker: The "Frozen Time" Trap

Despite good geometry, **PMI Rank** (the core quality metric) has not improved.
*   **Baseline Rank (Random):** $\sim 140k$
*   **Segment 1 (PMI 0.35):** $\sim 86k$
*   **Segment 2 (PMI 0.50):** $\sim 104k$ (Degraded)

### Root Cause Analysis
Diagnosed as **Insufficient Physical Time Evolution**.

1.  **Stability Constraint**: To keep Renorm stable with the new strong Volume potentials, we lowered $dt$ to `5e-5`.
2.  **Simulation Time**: 5000 steps $\times$ `5e-5` = **0.25s** physical time.
3.  **Dynamics**: Point masses in hyperbolic space moving under weak semantic forces ($k=0.5$) require physical time $T \sim 5.0 - 10.0s$ to migrate across the manifold and cluster angularly.

**We are effectively running the simulation in "Bullet Time" (superslow motion).** The particles are twitching in the right direction, but haven't had time to move.

---

## 3. Roadblocks & Trade-offs

| Strategy | Pros | Cons | Outcome |
| :--- | :--- | :--- | :--- |
| **Small dt (Current)** | High Stability (Renorm low) | Time Frozen; No Semantic Progress | **Stagnation** |
| **Large dt (Aggressive)** | Fast Alignment | Renorm Explosion; Manifold Departure | **Numerical Failure** |
| **High Solver Precision** | Large dt + Stability | Low Throughput (slower wall-clock) | **Proposed Path** |

---

## 4. Remediation Plan: "Compute for Time"

We must shift from "more steps" to "better steps".

### Action Items
1.  **Boost $dt$**: Increase physical time step to **$2 \times 10^{-4}$** (4x speedup).
2.  **Boost Precision**: Increase `fixed_point_iters` from 8 $\to$ **20**.
    *   This suppresses the Renorm instability caused by larger steps.
    *   Cost: Throughput drops (e.g., 25ms $\to$ 60ms per step), but "Physics per Wall-Clock Second" increases by $\sim 2x$.
3.  **Drive Harder**: Increase Semantic Force Limit (Tangent LR) or PMI weight if stability holds.

### Expected Outcome
*   Renorm maintained at $\approx 10^{-4}$ via brute-force solver precision.
*   Physical time accumulates 4x faster per step.
*   PMI Rank begins to drop visibly within 2000 steps.

### Next Step
Execute **Segment 3**: High-dt / High-Precision run.

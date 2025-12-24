# Aurora Phase 2: Emergence of Concept - Analysis & Plan

**Goal**: Scale the Aurora Engine from a 50-particle prototype to a 100,000-particle "Digital Mind" (WordNet Scale) driven by semantic streams.

## 1. The Challenge: Scalability Bottleneck
In Phase 1, we successfully verified Axiom 4.3 using `PairwisePotentialN` with $N=50$.
- **Current Complexity**: $O(N^2)$ (All-to-All interactions).
- **Target Scale**: $N \approx 10^5$ (WordNet nouns).
- **Impact**: $N^2 = 10^{10}$ pairs. Storing the distance matrix requires ~40 GB (float32), but computing gradients for it is prohibitively slow for real-time simulation.

**Conclusion**: We CANNOT use `PairwisePotentialN` for Phase 2. We must implement **Sparse Interactions**.

## 2. Theoretical Solution: Hyperbolic Cutoff & Sampling
Hyperbolic space expands exponentially. Most points are astronomically far apart.
According to `Aim.md`, we should apply:
1.  **Positive Interactions (Attraction)**: Driven by the "Text Stream" (or WordNet graph edges). Sparsity $\approx O(N)$ (average degree ~10).
2.  **Negative Interactions (Repulsion)**: Needed to prevent collapse. Instead of repelling *everyone*, we use **Negative Sampling**:
    - Each particle repels $k$ random neighbors per step.
    - Due to hyperbolic volume growth, repelling random samples effectively pushes particles into "free space" at the boundary.

**Target Complexity**: $O(N \cdot (K_{pos} + K_{neg}))$, which is linear.

## 3. Implementation Plan

### 3.1 New Modules
*   **`src/hei_n/sparse_potential_n.py`**:
    *   `SparseEdgePotential`: Takes a list of pairs `(u, v)` (Graph edges) and applies attraction.
    *   `NegativeSamplingPotential`: Generates random pairs dynamically for repulsion.
*   **`src/hei_n/dataloader_n.py`**:
    *   Loaders for WordNet (using `nltk` or raw files).
    *   Stream simulators (converting text window to positive pairs).

### 3.2 Experiment Design: `experiments/aurora_wordnet.py`
*   **Data**: WordNet Mammal Subtree (~1000 nodes) first, then full noun hierarchy (~80k).
*   **Dynamics**:
    *   Run `ContactIntegratorN` with Sparse Potentials.
    *   Use **Variable Inertia** (proven necessary in Phase 1).
    *   **Annealing**: Start with high temperature (high random force), cool down to let structure freeze.
*   **Metric**:
    *   compare `dist_hyperbolic(dog, cat)` vs `dist_tree(dog, cat)`.
    *   Metric: **Tree Distortion** (Simulated vs Ground Truth).

## 4. Work Packages (Step-by-Step)

### Step 2.1: Infrastructure Upgrade (Sparse Engine)
1.  Implement `SparseEdgePotential` (Vectorized for list of pairs).
2.  Implement `NegativeSamplingPotential`.
3.  Unit Test: Compare Sparse vs Dense gradient for small $N$.

### Step 2.2: Data Pipeline
1.  Create script to download/parse WordNet.
2.  Convert to edge list `(id1, id2)`.

### Step 2.3: "The Garden of Words" (Small Scale)
1.  Run simulation on $N \approx 1000$ (e.g., Animals).
2.  Visualize in Poincaré Disk.
3.  Verify semantic clustering (e.g., "Canine" cluster, "Feline" cluster).

### Step 2.4: Scale Up (Full Mind)
1.  Run on $N=80,000$.
2.  Analyze computation time and stability.

## 5. Request for Decision
Do you agree with this **Sparse Interaction** strategy?
If yes, I will begin by implementing `SparseEdgePotential` (Step 2.1).

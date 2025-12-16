"""Potential and gradient oracles for CCD experiments."""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import cayley_disk_to_uhp, sample_disk_hyperbolic, uhp_distance_and_grad
from .metrics import pairwise_tree_distance


class PotentialOracle(Protocol):
    """
    Interface for potential differentials on the upper half-plane.

    dV_dz / gradient must return the covector components (∂V/∂x + i ∂V/∂y)
    in the native UHP coordinates (Euclidean partials), NOT the Riemannian
    gradient with metric raising. Diamond expects this 1-form input.
    """

    def potential(self, z_uhp: ArrayLike, z_action: float | None = None) -> float:
        ...

    def dV_dz(self, z_uhp: ArrayLike, z_action: float | None = None) -> NDArray[np.complex128]:
        ...

    def gradient(self, z_uhp: ArrayLike, z_action: float | None = None) -> NDArray[np.complex128]:
        """
        Backward-compatibility alias for dV_dz; returns covector components.
        """
        ...


@dataclasses.dataclass
class GaussianWellsPotential:
    """
    Gaussian wells + global hyperbolic confinement to keep points from escaping.
    """

    centers: NDArray[np.complex128]
    weight: float = 1.0
    width: float = 0.3
    anneal_beta: float = 0.0  # depth scaling with contact action
    prior_stiffness: float = 0.1

    def _annealed_weight(self, z_action: float | None) -> float:
        if z_action is None or self.anneal_beta == 0.0:
            return self.weight
        scale = 1.0 + self.anneal_beta * max(0.0, float(z_action))
        return self.weight / scale

    def _global_confinement(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Harmonic confinement to UHP origin: V = 0.5 * k * d(z, i)^2.
        Returns (potential_sum, gradient array).
        """
        origin = 1.0j
        z_flat = z.ravel()
        pot_sum = 0.0
        grad_flat = np.zeros_like(z_flat, dtype=np.complex128)
        k = self.prior_stiffness
        if k <= 0:
            return pot_sum, grad_flat.reshape(z.shape)
        for i, val in enumerate(z_flat):
            d, g_d = uhp_distance_and_grad(val, origin)
            pot_sum += 0.5 * k * (d ** 2)
            grad_flat[i] = k * d * g_d
        return pot_sum, grad_flat.reshape(z.shape)

    def potential(self, z_uhp: ArrayLike, z_action: float | None = None) -> float:
        z = np.asarray(z_uhp, dtype=np.complex128)
        z_flat = z.ravel()
        c_flat = self.centers.ravel()
        if c_flat.size == 0 or z_flat.size == 0:
            V_prior, _ = self._global_confinement(z)
            return float(V_prior)

        weight = self._annealed_weight(z_action)
        V_wells_sum = 0.0
        for c in c_flat:
            dists = np.array([uhp_distance_and_grad(zi, c)[0] for zi in z_flat], dtype=float)
            wells = np.exp(-(dists**2) / (2 * self.width**2))
            V_wells_sum += wells.sum()

        V_wells = -weight * V_wells_sum
        V_prior, _ = self._global_confinement(z)
        return float(V_wells + V_prior)

    def dV_dz(self, z_uhp: ArrayLike, z_action: float | None = None) -> NDArray[np.complex128]:
        z = np.asarray(z_uhp, dtype=np.complex128)
        z_flat = z.ravel()
        c_flat = self.centers.ravel()
        weight = self._annealed_weight(z_action)
        grad_flat = np.zeros_like(z_flat, dtype=np.complex128)

        for i, zi in enumerate(z_flat):
            accum = 0.0 + 0.0j
            for c in c_flat:
                d, grad_d = uhp_distance_and_grad(zi, c)
                coeff = (d / (self.width**2)) * np.exp(-(d**2) / (2 * self.width**2))
                accum += weight * coeff * grad_d
            grad_flat[i] = accum

        grad = grad_flat.reshape(z.shape)
        _, grad_prior = self._global_confinement(z)
        return grad + grad_prior

    def gradient(self, z_uhp: ArrayLike, z_action: float | None = None) -> NDArray[np.complex128]:
        # Alias for compatibility
        return self.dV_dz(z_uhp, z_action)


def build_baseline_potential(
    n_anchors: int = 3,
    max_rho: float = 2.0,
    rng: np.random.Generator | None = None,
) -> GaussianWellsPotential:
    """Convenience helper to create a few Gaussian wells on the disk and map to UHP."""
    rng = np.random.default_rng() if rng is None else rng
    anchors_disk = sample_disk_hyperbolic(n=n_anchors, max_rho=max_rho, rng=rng)
    anchors_uhp = cayley_disk_to_uhp(anchors_disk)
    return GaussianWellsPotential(centers=anchors_uhp, width=0.4, weight=1.0, prior_stiffness=0.2)


@dataclasses.dataclass
class HierarchicalSoftminPotential:
    """
    Hierarchical potential with hyperbolic soft-min wells + weak repulsion.

    - Uses hyperbolic distance on UHP.
    - Soft-min via negative log-sum-exp to avoid one-well domination.
    - Optional pairwise repulsion to prevent full collapse.
    """

    centers: NDArray[np.complex128]
    depths: NDArray[np.int_]
    centers_disk: NDArray[np.complex128] | None = None
    parents: NDArray[np.int_] | None = None
    branching: int = 0
    max_depth: int = 0
    base_width: float = 0.5
    base_weight: float = 1.0
    softmin_scale: float = 0.2
    repulsion_weight: float = 0.0
    repulsion_p: float = 2.0
    repulsion_eps: float = 1e-2  # smoothing for repulsion kernel to avoid singularity
    ancestor_bias: float = 0.3
    anneal_beta: float = 0.0
    entropy_weight_base: float = 0.1
    entropy_gap_target: float = 0.2
    entropy_gap_scale: float = 0.1
    gap_eps: float = 1e-3
    beta: float = 1.0
    beta_eta: float = 0.5
    beta_target_bridge: float = 0.25
    beta_min: float = 0.5
    beta_max: float = 3.0
    bridge_ema: float = 0.0
    bridge_alpha: float = 0.1
    current_lambda: float = 0.0
    point_logit_bias: NDArray[np.float64] | None = None  # (K, N) additive logit bias from signal S (optional)
    point_leaf_ids: NDArray[np.int_] | None = None  # length N, latent leaf IDs used to generate S (optional)
    signal_scale: float = 1.0
    use_signal: bool = False  # default to pure self-organization (no point-level bias)
    prior_weight: float = 0.1  # global confining harmonic prior base weight (milder confinement)

    def _path_overlap_bias(self, idx: int, ref: int = 0) -> float:
        """Bias favoring anchors sharing ancestor with ref anchor."""
        if self.parents is None:
            return 0.0
        if idx >= len(self.parents) or ref >= len(self.parents):
            return 0.0
        pi = self.parents[idx]
        pr = self.parents[ref]
        if pi == pr and pi != -1:
            return -self.ancestor_bias
        if pi != -1 and pr != -1:
            gpi = self.parents[pi] if pi < len(self.parents) else -1
            gpr = self.parents[pr] if pr < len(self.parents) else -1
            if gpi == gpr and gpi != -1:
                return -0.5 * self.ancestor_bias
        return 0.0

    def _sibling_groups(self) -> list[list[int]]:
        """Return sibling index groups from parents array."""
        if self.parents is None:
            return []
        groups = {}
        for idx, p in enumerate(self.parents):
            if p == -1:
                continue
            groups.setdefault(p, []).append(idx)
        return [g for g in groups.values() if len(g) > 1]

    def _anneal_scale(self, z_action: float | None) -> float:
        if z_action is None or self.anneal_beta == 0.0:
            return 1.0
        return 1.0 / (1.0 + self.anneal_beta * max(0.0, float(z_action)))

    def _soft_assign(self, logits: np.ndarray) -> np.ndarray:
        """Stable softmax (beta already applied in logits construction)."""
        shifted = logits - np.max(logits, axis=0, keepdims=True)
        weights = np.exp(shifted)
        return weights / (weights.sum(axis=0, keepdims=True) + 1e-12)

    def _well_scores(self, z: np.ndarray, z_action: float | None) -> tuple[np.ndarray, np.ndarray]:
        """Compute soft-min logits and their gradients for each point."""
        logits = []
        grads = []
        weight_scale = self._anneal_scale(z_action)
        for idx, (c, d) in enumerate(zip(self.centers, self.depths)):
            width = self.base_width / (1 + d)
            weight = self.base_weight * (1 + d) * weight_scale
            d_hyp = np.zeros_like(z, dtype=float)
            grad_hyp = np.zeros_like(z, dtype=np.complex128)
            it = np.nditer(z, flags=["multi_index"])
            while not it.finished:
                val = complex(it[0])
                dist, grad = uhp_distance_and_grad(val, c)
                d_hyp[it.multi_index] = dist
                grad_hyp[it.multi_index] = grad
                it.iternext()
            # distance-adaptive width to reduce far-field pulling: width -> width*(1 + d_hyp)
            width_eff = width * (1.0 + d_hyp)
            energy = -self.softmin_scale * d_hyp / np.maximum(width_eff, 1e-8)
            bias = self._path_overlap_bias(idx, 0)
            logit = np.log(weight + 1e-12) + self.beta * (energy + bias)
            if self.use_signal and self.point_logit_bias is not None:
                logit = logit + self.signal_scale * self.point_logit_bias[idx]
            logits.append(logit)
            # dE/dd = -softmin_scale * (1 + d_hyp)^{-2} / width
            grad_coeff = -self.softmin_scale / (width * np.maximum((1.0 + d_hyp) ** 2, 1e-8))
            grads.append(self.beta * (grad_coeff * grad_hyp))
        logits_arr = np.stack(logits)  # (K, ...)
        logits_arr = np.clip(logits_arr, -50, 50)  # avoid overflow
        grads_arr = np.stack(grads)  # (K, ...)
        return logits_arr, grads_arr

    def _prior_strength(self) -> float:
        """Scale prior by tree depth to confine more strongly for deeper hierarchies."""
        depth_scale = max(1.0, float(self.max_depth)) if self.max_depth else 1.0
        return self.prior_weight * depth_scale

    def _compute_gaps(
        self, z_uhp: np.ndarray, soft: np.ndarray
    ) -> dict:
        """Compute gap diagnostics: leaf gap, path min gap, and geometric margin."""
        gaps_leaf = []
        gaps_path_min = []
        margins = []
        depths_arr = np.asarray(self.depths, dtype=int)
        max_depth = depths_arr.max() if depths_arr.size else 0
        leaf_idx = np.where(depths_arr == max_depth)[0]
        # precompute sibling groups
        sib_groups = self._sibling_groups()
        parent_map = np.asarray(self.parents, dtype=int) if self.parents is not None else None
        centers = self.centers
        for j in range(soft.shape[1]):
            # leaf gap
            if leaf_idx.size >= 2:
                vals = soft[leaf_idx, j]
                svals = np.sort(vals)
                gaps_leaf.append(float(svals[-1] - svals[-2]))
            else:
                gaps_leaf.append(0.0)
            # path min gap along MAP path
            if parent_map is not None and parent_map.size > 0:
                k_max = int(np.argmax(soft[:, j]))
                gaps_levels = []
                node = k_max
                while node != -1 and node < parent_map.size:
                    p = parent_map[node]
                    if p == -1:
                        break
                    # siblings of this parent
                    sib = [idx for idx, pp in enumerate(parent_map) if pp == p]
                    vals = soft[sib, j]
                    norm = vals.sum()
                    if vals.size >= 2 and norm > 0:
                        vs = np.sort(vals / norm)
                        gaps_levels.append(float(vs[-1] - vs[-2]))
                    node = p
                gaps_path_min.append(float(min(gaps_levels)) if gaps_levels else 0.0)
            else:
                gaps_path_min.append(0.0)
            # geometric margin using anchor distances
            zc = z_uhp.flat[j]
            dists = []
            for c in centers:
                d, _ = uhp_distance_and_grad(zc, c)
                dists.append(d)
            if len(dists) >= 2:
                sd = sorted(dists)
                margins.append(float(sd[1] - sd[0]))
            else:
                margins.append(0.0)
        gap_leaf_arr = np.array(gaps_leaf)
        gap_path_arr = np.array(gaps_path_min)
        margin_arr = np.array(margins)
        return {
            "gap_leaf": gap_leaf_arr,
            "gap_leaf_mean": float(np.mean(gap_leaf_arr)) if gap_leaf_arr.size else 0.0,
            "gap_leaf_median": float(np.median(gap_leaf_arr)) if gap_leaf_arr.size else 0.0,
            "gap_path_min": gap_path_arr,
            "gap_path_min_mean": float(np.mean(gap_path_arr)) if gap_path_arr.size else 0.0,
            "gap_path_min_median": float(np.median(gap_path_arr)) if gap_path_arr.size else 0.0,
            "gap_path_min_frac_small": float(np.mean(gap_path_arr < self.gap_eps)) if gap_path_arr.size else 0.0,
            "margin": margin_arr,
            "margin_mean": float(np.mean(margin_arr)) if margin_arr.size else 0.0,
            "margin_median": float(np.median(margin_arr)) if margin_arr.size else 0.0,
        }

    def update_lambda(self, z_uhp: ArrayLike, z_action: float | None = None) -> float:
        """
        Lagged entropy weight: compute once per step and keep fixed within the step.

        This avoids mixing d(lambda)/dz into the force, which would otherwise break
        the variational interpretation because gaps involve non-smooth ops (sort/min).
        """
        z = np.asarray(z_uhp, dtype=np.complex128)
        logits, _ = self._well_scores(z, z_action)
        soft = self._soft_assign(logits)
        gaps = self._compute_gaps(z, soft)
        gap_drive = gaps.get("gap_path_min_mean", 0.0)
        scale = max(self.entropy_gap_scale, 1e-6)
        lam = self.entropy_weight_base * max(0.0, (self.entropy_gap_target - gap_drive) / scale)
        self.current_lambda = float(lam)
        return self.current_lambda

    def potential(self, z_uhp: ArrayLike, z_action: float | None = None) -> float:
        z = np.asarray(z_uhp, dtype=np.complex128)
        logits, _ = self._well_scores(z, z_action)
        m = np.max(logits, axis=0)
        lse = m + np.log(np.exp(logits - m).sum(axis=0) + 1e-12)
        V_wells = -lse.sum()

        # repulsion
        V_rep = 0.0
        if self.repulsion_weight > 0:
            z_flat = z.ravel()
            n = z_flat.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    d, _ = uhp_distance_and_grad(z_flat[i], z_flat[j])
                    eps2 = self.repulsion_eps * self.repulsion_eps
                    V_rep += 1.0 / np.power(d * d + eps2, 0.5 * self.repulsion_p)
            V_rep *= self.repulsion_weight

        # entropy hardening over sibling groups (using path min gap)
        soft = self._soft_assign(logits)
        lam = self.current_lambda
        H_sib = 0.0
        groups = self._sibling_groups()
        eps = 1e-12
        for g in groups:
            p_g = soft[g, :]
            norm = p_g.sum(axis=0, keepdims=True)
            p_norm = p_g / (norm + eps)
            H = -(p_norm * np.log(p_norm + eps)).sum(axis=0)
            H_sib += H.sum()

        # global hyperbolic harmonic prior around origin (UHP origin at 1j)
        V_prior = 0.0
        if self.prior_weight > 0:
            center_uhp = 1.0j
            dists_origin = []
            it = np.nditer(z, flags=["multi_index"])
            while not it.finished:
                val = complex(it[0])
                d, _ = uhp_distance_and_grad(val, center_uhp)
                dists_origin.append(d * d)
                it.iternext()
            V_prior = 0.5 * self._prior_strength() * sum(dists_origin)

        return float(V_wells + V_rep + lam * H_sib + V_prior)

    def dV_dz(self, z_uhp: ArrayLike, z_action: float | None = None) -> NDArray[np.complex128]:
        z = np.asarray(z_uhp, dtype=np.complex128)
        logits, grads = self._well_scores(z, z_action)
        soft = self._soft_assign(logits)
        grad_wells = -np.sum(soft * grads, axis=0)

        if self.repulsion_weight > 0:
            z_flat = z.ravel()
            n = z_flat.shape[0]
            rep_grad = np.zeros_like(z_flat, dtype=np.complex128)
            for i in range(n):
                for j in range(i + 1, n):
                    d, grad_ij = uhp_distance_and_grad(z_flat[i], z_flat[j])
                    eps2 = self.repulsion_eps * self.repulsion_eps
                    base = np.power(d * d + eps2, 0.5 * self.repulsion_p + 1)
                    coeff = -self.repulsion_p * d / base
                    rep_grad[i] += coeff * grad_ij
                    rep_grad[j] -= coeff * grad_ij
            rep_grad = rep_grad.reshape(z.shape)
            grad_wells += self.repulsion_weight * rep_grad

        # entropy hardening gradient
        lam = self.current_lambda
        if lam > 0:
            groups = self._sibling_groups()
            eps = 1e-12
            dV_dlogits = np.zeros_like(logits)
            for g in groups:
                p_g = soft[g, :]
                norm = p_g.sum(axis=0, keepdims=True)
                p_norm = p_g / (norm + eps)
                v = np.log(p_norm + eps) + 1.0
                v_dot = (p_norm * v).sum(axis=0, keepdims=True)
                dH = -(p_norm * (v - v_dot))  # shape (len(g), ... )
                for idx_local, idx_global in enumerate(g):
                    dV_dlogits[idx_global, :] += lam * dH[idx_local]
            # chain to z via grads (d logit / dz)
            grad_entropy = np.sum(dV_dlogits * grads, axis=0)
            grad_wells += grad_entropy

        grad_prior = np.zeros_like(z, dtype=np.complex128)
        if self.prior_weight > 0:
            center_uhp = 1.0j
            it = np.nditer(z, flags=["multi_index"])
            while not it.finished:
                val = complex(it[0])
                d, g_d = uhp_distance_and_grad(val, center_uhp)
                grad_prior[it.multi_index] = self._prior_strength() * d * g_d
                it.iternext()

        return grad_wells + grad_prior

    def gradient(self, z_uhp: ArrayLike, z_action: float | None = None) -> NDArray[np.complex128]:
        # Alias for compatibility; returns covector components.
        return self.dV_dz(z_uhp, z_action)

    def forces_decomposed(self, z_uhp: ArrayLike, z_action: float | None = None) -> dict:
        """
        Return decomposed gradients: total, align(+rep), entropy, prior.
        """
        z = np.asarray(z_uhp, dtype=np.complex128)
        logits, grads = self._well_scores(z, z_action)
        soft = self._soft_assign(logits)
        grad_align = -np.sum(soft * grads, axis=0)
        grad_entropy = np.zeros_like(grad_align)
        grad_prior = np.zeros_like(grad_align)

        rep_grad = 0.0
        if self.repulsion_weight > 0:
            z_flat = z.ravel()
            n = z_flat.shape[0]
            rep_grad = np.zeros_like(z_flat, dtype=np.complex128)
            for i in range(n):
                for j in range(i + 1, n):
                    d, grad_ij = uhp_distance_and_grad(z_flat[i], z_flat[j])
                    eps2 = self.repulsion_eps * self.repulsion_eps
                    base = np.power(d * d + eps2, 0.5 * self.repulsion_p + 1)
                    coeff = -self.repulsion_p * d / base
                    rep_grad[i] += coeff * grad_ij
                    rep_grad[j] -= coeff * grad_ij
            rep_grad = rep_grad.reshape(z.shape)
            grad_align = grad_align + self.repulsion_weight * rep_grad

        gaps = self._compute_gaps(z, soft)
        lam = self.current_lambda
        if lam > 0:
            groups = self._sibling_groups()
            eps = 1e-12
            dV_dlogits = np.zeros_like(logits)
            for g in groups:
                p_g = soft[g, :]
                norm = p_g.sum(axis=0, keepdims=True)
                p_norm = p_g / (norm + eps)
                v = np.log(p_norm + eps) + 1.0
                v_dot = (p_norm * v).sum(axis=0, keepdims=True)
                dH = -(p_norm * (v - v_dot))
                for idx_local, idx_global in enumerate(g):
                    dV_dlogits[idx_global, :] += lam * dH[idx_local]
                grad_entropy = np.sum(dV_dlogits * grads, axis=0)
        if self.prior_weight > 0:
            center_uhp = 1.0j
            it = np.nditer(z, flags=["multi_index"])
            while not it.finished:
                val = complex(it[0])
                d, g_d = uhp_distance_and_grad(val, center_uhp)
                grad_prior[it.multi_index] = self._prior_strength() * d * g_d
                it.iternext()

        grad_total = grad_align + grad_entropy + grad_prior
        return {
            "grad_total": grad_total,
            "grad_align": grad_align,
            "grad_entropy": grad_entropy,
            "grad_prior": grad_prior,
            "lambda_ent": lam,
            "gap_stats": gaps,
            "beta": self.beta,
        }

    def gap_stats(self, z_uhp: ArrayLike, z_action: float | None = None) -> dict:
        """Return statistics of gap measures and current lambda."""
        z = np.asarray(z_uhp, dtype=np.complex128)
        logits, _ = self._well_scores(z, z_action)
        soft = self._soft_assign(logits)
        gaps = self._compute_gaps(z, soft)
        lam = self.current_lambda
        out = {
            "gap_leaf_mean": gaps.get("gap_leaf_mean", 0.0),
            "gap_leaf_median": gaps.get("gap_leaf_median", 0.0),
            "gap_leaf_min": float(np.min(gaps.get("gap_leaf", np.array([0.0])))) if gaps.get("gap_leaf", None) is not None else 0.0,
            "gap_leaf_max": float(np.max(gaps.get("gap_leaf", np.array([0.0])))) if gaps.get("gap_leaf", None) is not None else 0.0,
            "gap_path_min_mean": gaps.get("gap_path_min_mean", 0.0),
            "gap_path_min_median": gaps.get("gap_path_min_median", 0.0),
            "gap_path_min": float(np.min(gaps.get("gap_path_min", np.array([0.0])))) if gaps.get("gap_path_min", None) is not None else 0.0,
            "gap_path_min_frac_small": gaps.get("gap_path_min_frac_small", 0.0),
            "gap_leaf_q25": float(np.percentile(gaps.get("gap_leaf", np.array([0.0])), 25)) if gaps.get("gap_leaf", None) is not None else 0.0,
            "gap_leaf_q75": float(np.percentile(gaps.get("gap_leaf", np.array([0.0])), 75)) if gaps.get("gap_leaf", None) is not None else 0.0,
            "gap_path_q25": float(np.percentile(gaps.get("gap_path_min", np.array([0.0])), 25)) if gaps.get("gap_path_min", None) is not None else 0.0,
            "gap_path_q75": float(np.percentile(gaps.get("gap_path_min", np.array([0.0])), 75)) if gaps.get("gap_path_min", None) is not None else 0.0,
            "gap_path_max": float(np.max(gaps.get("gap_path_min", np.array([0.0])))) if gaps.get("gap_path_min", None) is not None else 0.0,
            "gap_path_frac_1e3": float(np.mean(gaps.get("gap_path_min", np.array([0.0])) < 1e-3)) if gaps.get("gap_path_min", None) is not None else 0.0,
            "gap_path_frac_1e2": float(np.mean(gaps.get("gap_path_min", np.array([0.0])) < 1e-2)) if gaps.get("gap_path_min", None) is not None else 0.0,
            "margin_mean": gaps.get("margin_mean", 0.0),
            "margin_median": gaps.get("margin_median", 0.0),
            "lambda_ent": lam,
        }
        return out

    def entropy_energy(self, z_uhp: ArrayLike, z_action: float | None = None) -> dict:
        """Return entropy term value for diagnostics."""
        z = np.asarray(z_uhp, dtype=np.complex128)
        logits, _ = self._well_scores(z, z_action)
        soft = self._soft_assign(logits)
        lam = self.current_lambda
        H_sib = 0.0
        groups = self._sibling_groups()
        eps = 1e-12
        for g in groups:
            p_g = soft[g, :]
            norm = p_g.sum(axis=0, keepdims=True)
            p_norm = p_g / (norm + eps)
            H = -(p_norm * np.log(p_norm + eps)).sum(axis=0)
            H_sib += H.sum()
        return {"V_ent": float(lam * H_sib), "lambda_ent": lam}

    def layer_soft_stats(self, z_uhp: ArrayLike, z_action: float | None = None) -> dict:
        """
        Return soft responsibilities aggregated per depth to diagnose dominance.
        """
        z = np.asarray(z_uhp, dtype=np.complex128)
        logits, _ = self._well_scores(z, z_action)
        weights = np.exp(logits)
        soft = weights / (weights.sum(axis=0, keepdims=True) + 1e-12)  # (K, ...)
        depths_arr = np.asarray(self.depths, dtype=int)
        depth_vals = np.unique(depths_arr)
        depth_responsibility = {}
        for d in depth_vals:
            mask = depths_arr == d
            if not np.any(mask):
                continue
            depth_responsibility[int(d)] = float(soft[mask].sum())
        total_resp = sum(depth_responsibility.values()) or 1.0
        depth_share = {k: v / total_resp for k, v in depth_responsibility.items()}
        max_depth = int(depth_vals.max()) if depth_vals.size else 0
        return {"depth_responsibility": depth_responsibility, "depth_share": depth_share, "max_depth": max_depth}


def _tree_r_disk(depth_idx: int, depth: int, max_rho: float) -> float:
    """Map tree depth index into a disk radius using geodesic radius max_rho."""
    if depth <= 1:
        return 0.0
    frac = float(depth_idx) / float(depth - 1)
    rho = frac * float(max_rho)
    return float(np.tanh(0.5 * rho))


def _build_tree_anchor_layout_disk(
    depth: int,
    branching: int,
    max_rho: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.complex128], NDArray[np.int_], NDArray[np.int_]]:
    centers_disk: list[NDArray[np.complex128]] = []
    depths: list[NDArray[np.int_]] = []
    parents: list[NDArray[np.int_]] = []
    offset = 0
    sectors = [(0.0, 2 * np.pi)]  # root sector
    for d in range(depth):
        r = _tree_r_disk(d, depth=depth, max_rho=max_rho)
        layer_pts: list[complex] = []
        layer_sectors: list[tuple[float, float]] = []
        if d == 0:
            theta = (sectors[0][0] + sectors[0][1]) / 2
            layer_pts.append(r * np.exp(1j * theta))
            depths.append(np.array([d], dtype=int))
            parents.append(np.array([-1], dtype=int))
            layer_sectors.append(sectors[0])
        else:
            prev_count = branching ** (d - 1)
            parent_start = offset - prev_count
            parent_indices: list[int] = []
            for p_idx in range(prev_count):
                p_sector = sectors[p_idx]
                width = (p_sector[1] - p_sector[0]) / branching
                for b in range(branching):
                    th0 = p_sector[0] + b * width
                    th1 = th0 + width
                    theta = rng.uniform(th0, th1)
                    layer_pts.append(r * np.exp(1j * theta))
                    parent_indices.append(parent_start + p_idx)
                    layer_sectors.append((th0, th1))
            depths.append(np.full(len(layer_pts), d, dtype=int))
            parents.append(np.asarray(parent_indices, dtype=int))
        centers_disk.append(np.asarray(layer_pts, dtype=np.complex128))
        sectors = layer_sectors
        offset += len(layer_pts)
    centers_disk_arr = np.concatenate(centers_disk)
    depths_arr = np.concatenate(depths)
    parents_arr = np.concatenate(parents)
    return centers_disk_arr, depths_arr, parents_arr


def _sample_hierarchical_features(
    parents: NDArray[np.int_],
    feature_dim: int,
    node_noise: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample a feature vector per tree node with parent→child Gaussian increments."""
    n = int(parents.shape[0])
    feats = np.zeros((n, feature_dim), dtype=float)
    if n == 0:
        return feats
    feats[0] = rng.normal(size=feature_dim)
    for i in range(1, n):
        p = int(parents[i])
        base = feats[p] if p >= 0 else 0.0
        feats[i] = base + float(node_noise) * rng.normal(size=feature_dim)
    return feats


def build_hierarchical_potential(
    n_points: int = 50,
    depth: int = 3,
    branching: int = 3,
    max_rho: float = 3.0,
    feature_dim: int = 4,
    node_noise: float = 1.0,
    obs_noise: float = 0.3,
    rng: np.random.Generator | None = None,
    use_signal: bool = False,
) -> HierarchicalSoftminPotential:
    """
    Build a hierarchical V_FEP(a,S) oracle.

    By default (use_signal=False) this returns a pure self-organization potential
    with only geometric tree wells (no point-level bias). Set use_signal=True to
    inject observation-conditioned logits via point_logit_bias/point_leaf_ids.

    Signal S is generated by:
    - sampling a tree-structured feature vector per node;
    - sampling each point's observation from a random leaf feature + noise.

    The resulting log-likelihood over tree nodes becomes a per-point additive bias
    on the geometric soft-min logits. This breaks global SL(2) symmetry and yields
    non-trivial Diamond torque while keeping the oracle cheap and differentiable.
    """
    rng = np.random.default_rng() if rng is None else rng
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if depth < 2:
        raise ValueError("depth must be >= 2")
    if branching < 2:
        raise ValueError("branching must be >= 2")
    if use_signal:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if obs_noise <= 0:
            raise ValueError("obs_noise must be > 0")

    centers_disk_arr, depths_arr, parents_arr = _build_tree_anchor_layout_disk(
        depth=depth,
        branching=branching,
        max_rho=max_rho,
        rng=rng,
    )
    centers_uhp = cayley_disk_to_uhp(centers_disk_arr)

    point_logit_bias = None
    point_leaf_ids = None
    if use_signal:
        node_features = _sample_hierarchical_features(
            parents=parents_arr,
            feature_dim=feature_dim,
            node_noise=node_noise,
            rng=rng,
        )
        leaf_nodes = np.where(depths_arr == (depth - 1))[0]
        if leaf_nodes.size == 0:
            raise ValueError("tree has no leaves")
        point_leaf_ids = rng.choice(leaf_nodes, size=n_points, replace=True).astype(int)
        obs = node_features[point_leaf_ids] + float(obs_noise) * rng.normal(size=(n_points, feature_dim))

        diff = obs[None, :, :] - node_features[:, None, :]  # (K, N, F)
        dist2 = np.sum(diff * diff, axis=-1)  # (K, N)
        point_logit_bias = -0.5 * dist2 / float(obs_noise * obs_noise)

    return HierarchicalSoftminPotential(
        centers=centers_uhp,
        depths=depths_arr,
        centers_disk=centers_disk_arr,
        parents=parents_arr,
        branching=branching,
        max_depth=depth,
        base_width=0.25,
        base_weight=2.0,
        softmin_scale=1.0,
        repulsion_weight=1e-3,
        point_logit_bias=point_logit_bias,
        point_leaf_ids=point_leaf_ids,
        use_signal=use_signal,
    )


def build_hierarchical_anchor_potential(
    depth: int = 3,
    branching: int = 3,
    max_rho: float = 3.0,
    rng: np.random.Generator | None = None,
) -> HierarchicalSoftminPotential:
    """Anchor-only (no signal S) version of the hierarchical soft-min potential."""
    rng = np.random.default_rng() if rng is None else rng
    centers_disk_arr, depths_arr, parents_arr = _build_tree_anchor_layout_disk(
        depth=depth,
        branching=branching,
        max_rho=max_rho,
        rng=rng,
    )
    centers_uhp = cayley_disk_to_uhp(centers_disk_arr)
    return HierarchicalSoftminPotential(
        centers=centers_uhp,
        depths=depths_arr,
        centers_disk=centers_disk_arr,
        parents=parents_arr,
        branching=branching,
        max_depth=depth,
        base_width=0.25,
        base_weight=2.0,
        softmin_scale=1.0,
        repulsion_weight=1e-3,
    )

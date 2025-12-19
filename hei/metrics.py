"""Clustering metrics on the Poincaré disk."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import hyperbolic_karcher_mean_disk


def poincare_distance_disk(z1: ArrayLike, z2: ArrayLike) -> float:
    """Geodesic distance on the Poincaré disk."""
    z1c = complex(z1)
    z2c = complex(z2)
    r1 = min(abs(z1c), 1 - 1e-9)
    r2 = min(abs(z2c), 1 - 1e-9)
    z1c = r1 * np.exp(1j * np.angle(z1c)) if r1 > 0 else 0.0
    z2c = r2 * np.exp(1j * np.angle(z2c)) if r2 > 0 else 0.0
    num = abs(z1c - z2c) ** 2
    den = max((1 - abs(z1c) ** 2) * (1 - abs(z2c) ** 2), 1e-12)
    val = 1 + 2 * num / den
    val = max(val, 1.0)
    return float(np.arccosh(val))


def pairwise_poincare(z: ArrayLike) -> NDArray[np.float64]:
    """Pairwise Poincaré distances for a 1D array of disk points (Vectorized)."""
    z_arr = np.asarray(z, dtype=np.complex128).ravel()
    
    # Vectorized computation using broadcasting
    # z_col: (N, 1), z_row: (1, N)
    z_col = z_arr[:, np.newaxis]
    z_row = z_arr[np.newaxis, :]
    
    # Helper for distance
    # d = arccosh(1 + 2 * |u-v|^2 / ((1-|u|^2)(1-|v|^2)))
    
    # |u - v|^2
    num = np.abs(z_col - z_row)**2
    
    # (1 - |u|^2)
    # This term depends only on the point itself.
    # one_minus_sq: (N,)
    one_minus_sq = 1.0 - np.abs(z_arr)**2
    # Denom: (N, N) via broadcast
    # den = (1-|u|^2) * (1-|v|^2)
    den = np.maximum(one_minus_sq[:, np.newaxis] * one_minus_sq[np.newaxis, :], 1e-12)
    
    arg = 1.0 + 2.0 * num / den
    
    # Handle floating point inaccuracies
    arg = np.maximum(arg, 1.0)
    
    return np.arccosh(arg)


def assign_nearest(z_disk: ArrayLike, anchors_disk: ArrayLike) -> NDArray[np.int_]:
    """Assign each point to the nearest anchor in hyperbolic distance."""
    z_arr = np.asarray(z_disk, dtype=np.complex128).ravel()
    a_arr = np.asarray(anchors_disk, dtype=np.complex128).ravel()
    labels = np.empty(z_arr.shape[0], dtype=int)
    for i, zc in enumerate(z_arr):
        dists = np.array([poincare_distance_disk(zc, ac) for ac in a_arr])
        labels[i] = int(np.argmin(dists))
    return labels


def silhouette_score(dist_matrix: NDArray[np.float64], labels: ArrayLike) -> float:
    """
    Compute silhouette score from a precomputed distance matrix.
    """
    labels_arr = np.asarray(labels, dtype=int)
    n = dist_matrix.shape[0]
    scores = []
    for i in range(n):
        same = labels_arr == labels_arr[i]
        other = labels_arr != labels_arr[i]
        # if singleton cluster, silhouette defined as 0
        if np.sum(same) <= 1:
            scores.append(0.0)
            continue
        # a: mean intra-cluster distance (exclude self)
        a = dist_matrix[i, same].sum() / (np.sum(same) - 1)
        # b: min mean distance to other clusters
        b = np.inf
        for lbl in np.unique(labels_arr[other]):
            mask = labels_arr == lbl
            b = min(b, dist_matrix[i, mask].mean())
        if not np.isfinite(b) or max(a, b) <= 0:
            s = 0.0
        else:
            s = (b - a) / max(a, b)
        scores.append(s)
    scores = [s for s in scores if np.isfinite(s)]
    return float(np.mean(scores)) if scores else 0.0


def build_parents_array(depth: int, branching: int) -> NDArray[np.int_]:
    """Return parent indices for a full branching^depth tree (global BFS indexing)."""
    total = sum(branching**d for d in range(depth))
    parents = np.full(total, -1, dtype=int)
    offset_prev = 0
    offset = 0
    for d in range(depth):
        n = branching**d
        offset_prev = offset
        offset += n
        if d == 0:
            continue
        for j in range(n):
            parent = offset_prev - branching ** (d - 1) + j // branching
            parents[offset_prev + j] = parent
    return parents


def lca_depth(i: int, j: int, parents: NDArray[np.int_], depths: NDArray[np.int_]) -> int:
    """Depth of lowest common ancestor using parent chains."""
    ancestors_i = set()
    ci = i
    while ci != -1:
        ancestors_i.add(ci)
        ci = parents[ci]
    cj = j
    while cj not in ancestors_i and cj != -1:
        cj = parents[cj]
    if cj == -1:
        return 0
    return int(depths[cj])


def pairwise_tree_distance(parents: NDArray[np.int_], depths: NDArray[np.int_]) -> NDArray[np.float64]:
    """Pairwise tree path lengths (edge count) between anchors."""
    n = depths.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            lca_d = lca_depth(i, j, parents, depths)
            d = depths[i] + depths[j] - 2 * lca_d
            dist[i, j] = dist[j, i] = d
    return dist


def pearson_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    a_flat = a.ravel()
    b_flat = b.ravel()
    if a_flat.size != b_flat.size:
        raise ValueError("Vectors must have same size for correlation")
    a_norm = (a_flat - a_flat.mean())
    b_norm = (b_flat - b_flat.mean())
    denom = np.linalg.norm(a_norm) * np.linalg.norm(b_norm)
    return float(a_norm.dot(b_norm) / denom) if denom > 0 else 0.0


def compute_hierarchical_metrics(
    z_disk: ArrayLike,
    anchors_disk: ArrayLike,
    depths: ArrayLike,
    parents: ArrayLike,
) -> dict:
    """
    Compute clustering metrics:
    - silhouette (hyperbolic) w.r.t. anchor assignments
    - correlation between hyperbolic distances and tree path lengths
    """
    labels_anchor = assign_nearest(z_disk, anchors_disk)
    dist_points = pairwise_poincare(z_disk)
    sil_anchor = silhouette_score(dist_points, labels_anchor)

    tree_dist = pairwise_tree_distance(np.asarray(parents, dtype=int), np.asarray(depths, dtype=int))
    # Map point-pair tree distance via assigned anchors
    n = len(labels_anchor)
    tree_pairs = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            td = tree_dist[labels_anchor[i], labels_anchor[j]]
            tree_pairs[i, j] = tree_pairs[j, i] = td
    # use upper triangle only
    idx = np.triu_indices(n, k=1)
    corr = pearson_corr(dist_points[idx], tree_pairs[idx])
    mean_tree_pair = float(tree_pairs[idx].mean()) if idx[0].size > 0 else 0.0

    # anchor spacing diagnostics
    depths_arr = np.asarray(depths, dtype=int)
    anchors_arr = np.asarray(anchors_disk, dtype=np.complex128)
    deepest_mask = depths_arr == depths_arr.max()
    deep_anchors = anchors_arr[deepest_mask] if deepest_mask.any() else anchors_arr
    deep_dist = pairwise_poincare(deep_anchors)
    deep_dist_vals = deep_dist[np.triu_indices(deep_dist.shape[0], k=1)]
    min_deep_dist = float(deep_dist_vals.min()) if deep_dist_vals.size else 0.0
    mean_deep_dist = float(deep_dist_vals.mean()) if deep_dist_vals.size else 0.0

    # intra / inter distances based on anchor labels
    labels_arr = np.asarray(labels_anchor, dtype=int)
    intra_dists = []
    inter_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels_arr[i] == labels_arr[j]:
                intra_dists.append(dist_points[i, j])
            else:
                inter_dists.append(dist_points[i, j])
    mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0

    return {
        "sil_anchor": sil_anchor,
        "tree_dist_corr": corr,
        "mean_tree_pair": mean_tree_pair,
        "labels_anchor": labels_anchor.tolist(),
        "mean_intra_anchor": mean_intra,
        "mean_inter_anchor": mean_inter,
        "deep_anchor_min_dist": min_deep_dist,
        "deep_anchor_mean_dist": mean_deep_dist,
    }


def _ancestor_at_depth(node: int, target_depth: int, parents: NDArray[np.int_], depths: NDArray[np.int_]) -> int:
    """Return the ancestor of node whose depth is target_depth (or closest above root)."""
    if target_depth < 0:
        return int(node)
    n = int(node)
    while n != -1 and int(depths[n]) > target_depth:
        n = int(parents[n])
    return int(n if n != -1 else node)


def labels_from_leaf_ids_at_depth(
    leaf_ids: ArrayLike,
    parents: ArrayLike,
    depths: ArrayLike,
    label_depth: int,
) -> NDArray[np.int_]:
    """
    Map each leaf node id to its ancestor node id at a given depth, then relabel to 0..K-1.
    """
    leaf_arr = np.asarray(leaf_ids, dtype=int).ravel()
    parents_arr = np.asarray(parents, dtype=int)
    depths_arr = np.asarray(depths, dtype=int)
    anc = np.array([_ancestor_at_depth(int(n), label_depth, parents_arr, depths_arr) for n in leaf_arr], dtype=int)
    uniq = {node: idx for idx, node in enumerate(np.unique(anc))}
    return np.array([uniq[int(a)] for a in anc], dtype=int)


def compute_signal_metrics(
    z_disk: ArrayLike,
    leaf_ids: ArrayLike,
    parents: ArrayLike,
    depths: ArrayLike,
    label_depth: int = 1,
) -> dict:
    """
    Metrics for signal-derived tree supervision (no anchor wells).

    - tree_dist_corr: correlation between point hyperbolic distances and tree distances
      between the (fixed) leaf IDs assigned to each point.
    - sil_tree: silhouette using labels given by ancestor group at label_depth.
    """
    z_arr = np.asarray(z_disk, dtype=np.complex128).ravel()
    leaf_arr = np.asarray(leaf_ids, dtype=int).ravel()
    if z_arr.size != leaf_arr.size:
        raise ValueError("z_disk and leaf_ids must have the same length")
    n = z_arr.size
    dist_points = pairwise_poincare(z_arr)
    tree_dist = pairwise_tree_distance(np.asarray(parents, dtype=int), np.asarray(depths, dtype=int))
    tree_pairs = tree_dist[np.ix_(leaf_arr, leaf_arr)].astype(float)

    idx = np.triu_indices(n, k=1)
    corr = pearson_corr(dist_points[idx], tree_pairs[idx]) if idx[0].size else 0.0
    mean_tree_pair = float(tree_pairs[idx].mean()) if idx[0].size else 0.0

    labels = labels_from_leaf_ids_at_depth(leaf_arr, parents, depths, label_depth=label_depth)
    sil_tree = silhouette_score(dist_points, labels) if np.unique(labels).size >= 2 else 0.0

    return {
        "tree_dist_corr": float(corr),
        "mean_tree_pair": float(mean_tree_pair),
        "sil_tree": float(sil_tree),
        "label_depth": int(label_depth),
        "labels_tree": labels.tolist(),
        "leaf_ids": leaf_arr.tolist(),
    }


def silhouette_timeseries_labels(
    positions_disk: list[ArrayLike],
    labels: ArrayLike,
) -> np.ndarray:
    """Silhouette over time using fixed labels and hyperbolic distance."""
    labels_arr = np.asarray(labels, dtype=int)
    sil = []
    for pos in positions_disk:
        dist = pairwise_poincare(pos)
        sil.append(silhouette_score(dist, labels_arr))
    return np.array(sil)


def compute_label_timeseries(
    positions_disk: list[ArrayLike],
    anchors_disk: ArrayLike,
    depths: ArrayLike,
    max_depth: int,
) -> dict:
    """
    For each timestep, assign nearest anchor and compute:
    - mean depth E[d(t)]
    - fraction of points at deepest leaves
    - label entropy (natural log; not normalized)
    """
    anchors = np.asarray(anchors_disk, dtype=np.complex128)
    depths_arr = np.asarray(depths, dtype=int)
    deepest_mask = depths_arr == (max_depth - 1)

    mean_depth = []
    deep_frac = []
    entropy = []

    for pos in positions_disk:
        labels = assign_nearest(pos, anchors)
        dvals = depths_arr[labels]
        mean_depth.append(float(dvals.mean()))
        if deepest_mask.any():
            deep_frac.append(float(np.mean(deepest_mask[labels])))
        else:
            deep_frac.append(0.0)
        # entropy
        counts = np.bincount(labels, minlength=anchors.shape[0]).astype(float)
        probs = counts[counts > 0] / counts.sum() if counts.sum() > 0 else np.array([])
        ent = float(-(probs * np.log(probs)).sum()) if probs.size > 0 else 0.0
        entropy.append(ent)

    return {
        "mean_depth": np.array(mean_depth),
        "deep_frac": np.array(deep_frac),
        "entropy": np.array(entropy),
    }


def silhouette_timeseries(
    positions_disk: list[ArrayLike],
    anchors_disk: ArrayLike,
) -> np.ndarray:
    """Silhouette over time using nearest-anchor labels and hyperbolic distance."""
    anchors = np.asarray(anchors_disk, dtype=np.complex128)
    sil = []
    for pos in positions_disk:
        labels = assign_nearest(pos, anchors)
        dist = pairwise_poincare(pos)
        sil.append(silhouette_score(dist, labels))
    return np.array(sil)


def _init_centers_farthest(z: NDArray[np.complex128], k: int, rng: np.random.Generator) -> NDArray[np.complex128]:
    """Farthest-point init under hyperbolic distance."""
    n = z.shape[0]
    centers = [z[rng.integers(0, n)]]
    for _ in range(1, k):
        dists = np.array([min(poincare_distance_disk(zi, c) for c in centers) for zi in z])
        total = dists.sum()
        if total <= 0 or not np.isfinite(total):
            probs = np.full(n, 1.0 / n)
        else:
            probs = dists / total
        probs = np.where(np.isfinite(probs), probs, 0.0)
        if probs.sum() <= 0:
            probs = np.full(n, 1.0 / n)
        else:
            probs = probs / probs.sum()
        centers.append(z[rng.choice(n, p=probs)])
    return np.array(centers)


def _hyperbolic_kmeans_once(
    points_complex: NDArray[np.complex128],
    k: int,
    rng: np.random.Generator,
    max_iters: int = 30,
) -> np.ndarray:
    """
    Hyperbolic k-means on disk coords:
    - initialization via farthest-point
    - assignment via Poincaré distance
    - update via hyperbolic Karcher mean
    - empty cluster handled by reinit to farthest point
    """
    z = np.asarray(points_complex, dtype=np.complex128).ravel()
    n = z.shape[0]
    if k >= n:
        return np.arange(n)
    centers = _init_centers_farthest(z, k, rng)
    for _ in range(max_iters):
        dists = np.zeros((n, k), dtype=float)
        for j in range(k):
            cj = centers[j]
            for i, zi in enumerate(z):
                dists[i, j] = poincare_distance_disk(zi, cj)
        labels = np.argmin(dists, axis=1)
        # handle empty clusters by reassigning center to farthest point
        for j in range(k):
            if not np.any(labels == j):
                far_idx = np.argmax(np.min(dists, axis=1))
                centers[j] = z[far_idx]
                labels[far_idx] = j
        new_centers = []
        fused = False
        for j in range(k):
            pts_j = z[labels == j]
            if pts_j.size == 0:
                new_centers.append(centers[j])
                continue
            c_mean = hyperbolic_karcher_mean_disk(pts_j, max_iter=20)
            new_centers.append(c_mean)
        new_centers = np.array(new_centers, dtype=np.complex128)
        # detect center fusion
        for a in range(k):
            for b in range(a + 1, k):
                if poincare_distance_disk(new_centers[a], new_centers[b]) < 1e-3:
                    far_idx = np.argmax(np.min(dists, axis=1))
                    new_centers[b] = z[far_idx]
                    fused = True
        if not fused and np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def cluster_summary_kmeans(
    z_disk: ArrayLike,
    k_list: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """
    Scan k-means over k_list (default 2..5). Return best silhouette and cluster stats.
    """
    z_arr = np.asarray(z_disk, dtype=np.complex128).ravel()
    if k_list is None:
        k_list = [2, 3, 4, 5]
    best = {"k": None, "silhouette": -1e9, "labels": None}
    rng = np.random.default_rng(seed)
    dist = pairwise_poincare(z_arr)
    for k in k_list:
        labels = _hyperbolic_kmeans_once(z_arr, k, rng)
        if np.unique(labels).size < 2:
            continue  # degenerate: collapsed to single cluster
        sil = silhouette_score(dist, labels)
        if sil > best["silhouette"]:
            best = {"k": k, "silhouette": sil, "labels": labels}
    labels = best.get("labels")
    if labels is None:
        # fallback: single cluster
        labels = np.zeros(len(z_arr), dtype=int)
        k_best = 1
        best_sil = 0.0
    else:
        k_best = best["k"]
        best_sil = best["silhouette"]
    uniq, counts = np.unique(labels, return_counts=True)
    counts_sorted = np.sort(counts)[::-1]
    # full counts including empty clusters
    counts_full = [int(np.sum(labels == j)) for j in range(k_best)]
    total = len(labels)
    max_frac = float(counts_sorted[0] / total)
    top3_frac = float(counts_sorted[:3].sum() / total) if counts_sorted.size >= 3 else float(counts_sorted.sum() / total)
    return {
        "best_k": int(k_best) if k_best is not None else None,
        "best_silhouette": float(best_sil),
        "cluster_counts": counts_sorted.tolist(),
        "cluster_counts_full": counts_full,
        "max_frac": max_frac,
        "top3_frac": top3_frac,
        "labels": labels.tolist(),
    }


def cluster_anchor_profiles(
    labels: ArrayLike,
    points_disk: ArrayLike,
    anchors_disk: ArrayLike,
    depths: ArrayLike,
    parents: ArrayLike,
) -> dict:
    """
    For each cluster label, compute:
    - depth stats of nearest anchors (mean, frac deepest)
    - entropy over anchor IDs (within-cluster semantic consistency)
    - pairwise mean tree distance between clusters
    """
    labels_arr = np.asarray(labels, dtype=int)
    pts_arr = np.asarray(points_disk, dtype=np.complex128)
    anchors = np.asarray(anchors_disk, dtype=np.complex128)
    depths_arr = np.asarray(depths, dtype=int)
    parents_arr = np.asarray(parents, dtype=int)
    tree_dist = pairwise_tree_distance(parents_arr, depths_arr)
    clusters = {}
    uniq = np.unique(labels_arr)
    deepest = depths_arr.max() if depths_arr.size else 0

    # map each point to nearest anchor id
    point_anchor_ids = assign_nearest(pts_arr, anchors)

    # Build anchor ids per label
    for lbl in uniq:
        mask = labels_arr == lbl
        if not np.any(mask):
            continue
        anchor_ids = point_anchor_ids[mask]
        dvals = depths_arr[anchor_ids]
        mean_depth = float(dvals.mean()) if dvals.size else 0.0
        frac_deep = float(np.mean(dvals == deepest)) if dvals.size else 0.0
        counts = np.bincount(anchor_ids, minlength=anchors.shape[0]).astype(float)
        probs = counts[counts > 0] / counts.sum() if counts.sum() > 0 else np.array([])
        ent = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
        clusters[int(lbl)] = {
            "mean_depth": mean_depth,
            "frac_deepest": frac_deep,
            "anchor_entropy": ent,
        }

    # inter-cluster tree distance
    inter = {}
    for i in uniq:
        for j in uniq:
            if j <= i:
                continue
            ids_i = point_anchor_ids[labels_arr == i]
            ids_j = point_anchor_ids[labels_arr == j]
            if ids_i.size == 0 or ids_j.size == 0:
                continue
            td = tree_dist[np.ix_(ids_i, ids_j)].mean()
            inter[f"{int(i)}-{int(j)}"] = float(td)

    return {"clusters": clusters, "inter_tree_distance": inter}

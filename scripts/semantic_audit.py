"""
Semantic Audit Panel for Aurora Base Checkpoints.
==================================================

Analyzes trained HEI embeddings for:
1. Hierarchy Preservation (Correlation).
2. Cluster Quality (Contrast Ratio).
3. Radius Drift.
4. NaN / Renorm stats.
"""

import numpy as np
import pickle
import argparse
import os
import sys
from scipy.stats import spearmanr, pearsonr

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

def load_checkpoint(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_radii(G):
    """Compute hyperbolic radii from group elements."""
    # x = G[:, :, 0] (First column is position vector)
    x = G[:, :, 0]
    x0 = x[:, 0]
    # r = arccosh(x0)
    x0 = np.maximum(x0, 1.0)
    return np.arccosh(x0)

def compute_hierarchy_correlation(radii, depths):
    """
    Compute correlation between hyperbolic radius and graph depth.
    Reports both Spearman and Pearson.
    """
    mask = depths > 0  # Exclude root
    r = radii[mask]
    d = depths[mask]
    
    spearman, _ = spearmanr(r, d)
    pearson, _ = pearsonr(r, d)
    
    return spearman, pearson

def compute_contrast_ratio(radii, depths, num_bins=5):
    """
    Compute Contrast Ratio.
    Idea: Nodes at similar depths should have similar radii.
    Contrast = Variance(radii) across layer / Variance within layer.
    Simplified: Just ratio of Inter-layer spread to Intra-layer spread.
    """
    bins = np.linspace(0, np.max(depths), num_bins + 1)
    layer_means = []
    intra_vars = []
    
    for i in range(num_bins):
        mask = (depths >= bins[i]) & (depths < bins[i+1])
        if np.sum(mask) > 1:
            layer_means.append(np.mean(radii[mask]))
            intra_vars.append(np.var(radii[mask]))
            
    if len(layer_means) < 2:
        return 1.0  # Not enough layers
        
    inter_var = np.var(layer_means)
    mean_intra = np.mean(intra_vars)
    
    contrast = inter_var / (mean_intra + 1e-9)
    return contrast

def audit(checkpoint_path, depths_path=None):
    print(f"Loading checkpoint: {checkpoint_path}")
    data = load_checkpoint(checkpoint_path)
    
    G = data['G']
    nodes = data.get('nodes', None)
    N = G.shape[0]
    
    print(f"Loaded {N} particles.")
    
    # Radii
    radii = compute_radii(G)
    mean_R = np.mean(radii)
    R_95 = np.percentile(radii, 95)
    
    # NaN Check
    nan_count = np.sum(np.isnan(G))
    
    print("\n=== SEMANTIC AUDIT PANEL ===")
    print(f"Particles:       {N}")
    print(f"Mean Radius:     {mean_R:.4f}")
    print(f"R_95:            {R_95:.4f}")
    print(f"NaN Count:       {nan_count}  {'PASS' if nan_count == 0 else 'FAIL'}")
    
    # Hierarchy Correlation (needs depths)
    # Try to load depths from external file or compute
    if depths_path and os.path.exists(depths_path):
        depths = np.load(depths_path)
    else:
        # Attempt to reconstruct from graph (requires node list + taxonomy)
        # For now, use a placeholder
        print("\n[WARN] Depths not provided. Skipping Hierarchy Correlation.")
        depths = None
        
    if depths is not None:
        spearman, pearson = compute_hierarchy_correlation(radii, depths)
        contrast = compute_contrast_ratio(radii, depths)
        
        print(f"\nSpearman Corr:   {spearman:.4f}  {'PASS' if spearman > 0.4 else 'WARN'}")
        print(f"Pearson Corr:    {pearson:.4f}")
        print(f"Contrast Ratio:  {contrast:.4f}  {'PASS' if contrast > 1.5 else 'WARN'}")
    
    print("============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .pkl checkpoint")
    parser.add_argument("--depths", type=str, default=None, help="Path to depths .npy (optional)")
    
    args = parser.parse_args()
    audit(args.checkpoint, args.depths)

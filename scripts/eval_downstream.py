"""
Phase 11 Final Validation: Downstream Tasks.
============================================

Compares two representations for Link Prediction:
1. Hyperbolic Baseline: s(u, v) = -distance(u, v)
2. Gauge Enhanced:      s(u, v) = Alignment(J_u, J_v)

Also performs "Wormhole Detection": identifying pairs that are far apart in embedding space
but strongly connected via the Gauge Field (High Alignment).
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
import subprocess

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
# Add scripts to path to import eval_semantic_truth
sys.path.append(os.path.dirname(__file__))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField
from aurora.geometry import dist_hyperbolic
from aurora.dynamics import PhysicsState
from eval_semantic_truth import build_parent_map, get_synonym_group

def compute_scores(u, v, x, J, gauge_field, batch_size=1000, device='cuda'):
    """
    Compute Hyperbolic and Gauge scores for pairs (u, v).
    """
    num_edges = len(u)
    hyp_scores = []
    gauge_scores = []
    
    # Process in batches
    for i in range(0, num_edges, batch_size):
        end = min(i + batch_size, num_edges)
        u_b = u[i:end]
        v_b = v[i:end]
        
        # 1. Hyperbolic Score (Negative Distance)
        xu = x[u_b]
        xv = x[v_b]
        
        # Manually compute dist to avoid overhead if needed, or use geometry function
        # Using simple approximation or rigorous formula
        # dist_hyperbolic expects (N, dim)
        dists = dist_hyperbolic(xu, xv)
        hyp_scores.append(-dists.detach().cpu().numpy())
        
        # 2. Gauge Score (Alignment)
        # Construct edge tensor for batch
        # We need (u, v) pairs
        edges_b = torch.stack([u_b, v_b], dim=1)
        
        # Get Parallel Transport
        U_b = gauge_field.get_U(x=x, edges=edges_b) # (B, k, k)
        
        Ju = J[u_b]
        Jv = J[v_b]
        
        # Transport Ju -> Ju_trans
        Ju_trans = torch.matmul(U_b, Ju.unsqueeze(-1)).squeeze(-1)
        
        # Alignment
        align = torch.sum(Jv * Ju_trans, dim=-1)
        gauge_scores.append(align.detach().cpu().numpy())
        
    return np.concatenate(hyp_scores), np.concatenate(gauge_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--semantic_path", type=str, required=True, help="Ground Truth Semantic Edges")
    parser.add_argument("--dataset", type=str, default="cilin")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--neg_ratio", type=int, default=1, help="Negatives per positive")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    print(f"Loading checkpoint {args.checkpoint}...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
        
    x = torch.tensor(ckpt['x'], dtype=torch.float32, device=device)
    J = torch.tensor(ckpt['J'], dtype=torch.float32, device=device)
    
    # Reconstruct Gauge Field
    print("Reconstructing Gauge Field...")
    # Need edge topology used in training? 
    # Actually for inference get_U(x, edges) usually allows arbitrary edges IF backend is neural.
    # If backend is table, we can ONLY predict on training edges.
    # Check backend.
    
    gf_state = ckpt['gauge_field']
    is_neural = any('backend.net' in k for k in gf_state.keys())
    
    if not is_neural:
        print("Error: Downstream validation requires Neural Backend for generalization.")
        return
        
    # Reconstruct Generic GaugeField (edges don't matter for Neural inference, just input_dim/logical_dim)
    # But constructor checks edges.
    # Let's pass dummy edges.
    dummy_edges = torch.zeros((1, 2), dtype=torch.long, device=device)
    gauge_field = GaugeField(dummy_edges, logical_dim=3, input_dim=5, backend_type='neural').to(device)
    # Filter out mismatching buffers (edges, tri_*)
    # Neural backend logic only depends on self.backend.net
    filtered_state = {k: v for k, v in gf_state.items() if not (k.startswith('edges') or k.startswith('tri_'))}
    
    gauge_field.load_state_dict(filtered_state, strict=False)
    
    # 2. Load Evaluation Data
    print("Loading Ground Truth Semantics...")
    ds = AuroraDataset(args.dataset, limit=args.limit)
    
    # Load ALL edges (Train + Holdout) to test global capability?
    # Or just holdout? The user wants "Downstream Validation".
    # Typically we evaluate on Link Prediction (Positive vs Negative).
    # Since we mining added *new* edges, we should evaluate on the *Input* semantic set (Ground Truth).
    
    gt_edges = ds.load_semantic_edges(args.semantic_path, split="all")
    
    # Convert to indices
    pos_u = []
    pos_v = []
    
    # NOTE: semantic_path might contain weighted edges.
    # We take all.
    for u, v, w in gt_edges:
        pos_u.append(u)
        pos_v.append(v)
        
    pos_u = torch.tensor(pos_u, dtype=torch.long, device=device)
    pos_v = torch.tensor(pos_v, dtype=torch.long, device=device)
    
    num_pos = len(pos_u)
    print(f"Evaluate on {num_pos} Positive Semantic Edges.")
    
    # 3. Generate Negatives
    print(f"Generating {args.neg_ratio}x Negatives...")
    num_neg = num_pos * args.neg_ratio
    neg_u = torch.randint(0, ds.num_nodes, (num_neg,), device=device)
    neg_v = torch.randint(0, ds.num_nodes, (num_neg,), device=device)
    # Simple random negatives (unlikely to be true positives)
    
    # 4. Compute Scores
    print("Computing Scores for Positives...")
    pos_hyp, pos_gauge = compute_scores(pos_u, pos_v, x, J, gauge_field, device=device)
    
    print("Computing Scores for Negatives...")
    neg_hyp, neg_gauge = compute_scores(neg_u, neg_v, x, J, gauge_field, device=device)
    
    # 5. Metrics
    y_true = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
    y_hyp = np.concatenate([pos_hyp, neg_hyp])
    y_gauge = np.concatenate([pos_gauge, neg_gauge])
    
    auc_hyp = roc_auc_score(y_true, y_hyp)
    auc_gauge = roc_auc_score(y_true, y_gauge)
    
    print("\n=== Downstream Validation Results ===")
    print(f"Task: Link Prediction (N_pos={num_pos})")
    print(f"Hyperbolic Baseline AUC: {auc_hyp:.4f}")
    print(f"Gauge-Enhanced AUC:      {auc_gauge:.4f}")
    print(f"Improvement:             {auc_gauge - auc_hyp:+.4f}")
    
    # 5b. Hard Subset AUC (Discordant Regime)
    # Define "Hard": Hyperbolic Distance > 3.0 (or P80) on Hyberbolic Baseline
    # Filter y_true and score based on y_hyp (scores are negative distance)
    # So "Hard" is y_hyp < -3.0
    
    threshold_hard = -3.0
    mask_hard = y_hyp < threshold_hard 
    
    if mask_hard.sum() > 100:
        y_true_hard = y_true[mask_hard]
        y_hyp_hard = y_hyp[mask_hard]
        y_gauge_hard = y_gauge[mask_hard]
        
        # Only compute if we have both classes
        if len(np.unique(y_true_hard)) > 1:
            auc_hyp_h = roc_auc_score(y_true_hard, y_hyp_hard)
            auc_gauge_h = roc_auc_score(y_true_hard, y_gauge_hard)
            print(f"\n--- Hard Subset (Dist > {-threshold_hard}) N={mask_hard.sum()} ---")
            print(f"Hyperbolic AUC: {auc_hyp_h:.4f}")
            print(f"Gauge AUC:      {auc_gauge_h:.4f}")
            print(f"Improvement:    {auc_gauge_h - auc_hyp_h:+.4f}")
        else:
            print(f"\n--- Hard Subset N={mask_hard.sum()} ---")
            print("Skipping AUC (Only one class present in subset).")
            
    # 6. Qualitative: "Wormhole" Detection
    # Look for Positives where Hyp is Low (Bad) but Gauge is High (Good).
    # Hyp Low means dist is LARGE.
    # Gauge High means Align is close to 1.
    
    # Z-Score normalization for fair comparison?
    # Or simple percentile.
    
    # Let's find pairs in Positives:
    dist_p = -pos_hyp # Actual distance
    align_p = pos_gauge
    
    # Criteria: Distance > P80 AND Alignment > P80?
    dist_threshold = np.percentile(dist_p, 90) # Top 10% furthest
    align_threshold = np.percentile(align_p, 50) # Better than median assumption
    
    wormholes = []
    # Build parent map for Truth Check
    parent_map = build_parent_map(ds)
    
    strict_hit = 0
    soft_hit = 0
    total_wh = 0
    
    for i in range(num_pos):
        d = dist_p[i]
        a = align_p[i]
        if d > dist_threshold and a > align_threshold:
            u_i = pos_u[i].item()
            v_i = pos_v[i].item()
            wormholes.append((i, d, a, u_i, v_i))
            
            # Truth Check (Inline)
            parent_u = get_synonym_group(u_i, parent_map, ds.nodes)
            parent_v = get_synonym_group(v_i, parent_map, ds.nodes)
            
            total_wh += 1
            is_strict = False
            is_soft = False
            
            if parent_u and parent_v:
                if parent_u == parent_v:
                    is_strict = True
                    is_soft = True
                else:
                    grand_u = get_synonym_group(parent_u, parent_map, ds.nodes)
                    grand_v = get_synonym_group(parent_v, parent_map, ds.nodes)
                    if grand_u and grand_v and grand_u == grand_v:
                         is_soft = True
                    else:
                         code_u = ds.nodes[parent_u]
                         code_v = ds.nodes[parent_v]
                         if code_u.startswith("Code:") and code_v.startswith("Code:"):
                             if code_u.split(":")[1][:2] == code_v.split(":")[1][:2]:
                                 is_soft = True
            
            if is_strict: strict_hit += 1
            if is_soft: soft_hit += 1
            
    # Sort by "Surprise" (High Dist + High Align)
    wormholes.sort(key=lambda item: item[2] * item[1], reverse=True)
    
    print(f"\n=== Wormhole Discovery (Top 10) ===")
    print("Pairs that are spatially distant but geometrically aligned (Non-local Semantic Connections).")
    print(f"{'Idx':<6} {'Word A':<20} {'Word B':<20} {'Dist':<10} {'Align':<10}")
    print("-" * 70)
    
    for k in range(min(10, len(wormholes))):
        idx, d, a, u_i, v_i = wormholes[k]
        
        w_u = ds.nodes[u_i]
        w_v = ds.nodes[v_i]
        
        # Clean words (Code:Word:Id -> Word)
        if "C:" in w_u: w_u = w_u.split(":")[1]
        if "C:" in w_v: w_v = w_v.split(":")[1]
        
        print(f"{idx:<6} {w_u:<20} {w_v:<20} {d:.4f}     {a:.4f}")

    print("-" * 70)
    print(f"Found {len(wormholes)} potential wormholes (High Dist, High Align).")
    if total_wh > 0:
        print(f"Wormhole Strict Precision: {strict_hit}/{total_wh} ({strict_hit/total_wh*100:.2f}%)")
        print(f"Wormhole Soft Precision:   {soft_hit}/{total_wh} ({soft_hit/total_wh*100:.2f}%)")

if __name__ == "__main__":
    main()

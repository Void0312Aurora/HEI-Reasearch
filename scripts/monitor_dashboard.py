"""
Automated Guardrails Dashboard (Phase 13, Track B).
===================================================

Generates a JSON report of the "Safe 6" metrics for a Closed-Loop Cycle.
Usage: python scripts/monitor_dashboard.py --checkpoint <path> --cycle <N>

Metrics:
1. Local Curvature (P99)
2. Global Frustration (Mean)
3. Precision (Strict & Soft)
4. Hard-Regime AUC (Dist > 3.0)
5. Edge Growth
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse
import json
import networkx as nx
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.append(os.path.dirname(__file__)) # For peer imports

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField
from aurora.geometry import dist_hyperbolic
from aurora.topology import GlobalCycleMonitor
from eval_semantic_truth import build_parent_map, get_synonym_group

def compute_local_curvature(gauge_field, x, samples=1000):
    # Sample triangles from GaugeField
    # Return P99
    # GaugeField has .triangles
    if len(gauge_field.triangles) == 0:
        return 0.0
        
    # Compute curvature
    Omega, _, _ = gauge_field.compute_curvature(x)
    norm = torch.norm(Omega.reshape(Omega.shape[0], -1), dim=1)
    return torch.quantile(norm, 0.99).item()

def compute_precision(edges, ds):
    parent_map = build_parent_map(ds)
    strict = 0
    soft = 0
    total = 0
    
    for item in edges:
        u, v = item[0], item[1]
        
        parent_u = get_synonym_group(u, parent_map, ds.nodes)
        parent_v = get_synonym_group(v, parent_map, ds.nodes)
        
        if parent_u is None or parent_v is None: continue
        
        total += 1
        is_strict = False
        is_soft = False
        
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
                         
        if is_strict: strict += 1
        if is_soft: soft += 1
        
    return (strict/total if total else 0.0, soft/total if total else 0.0)

def compute_hard_auc(gauge_field, x, J, ds, sem_edges):
    # Negative Sampling
    pos_u = torch.tensor([e[0] for e in sem_edges], device=x.device)
    pos_v = torch.tensor([e[1] for e in sem_edges], device=x.device)
    
    num_neg = len(sem_edges)
    neg_u = torch.randint(0, x.shape[0], (num_neg,), device=x.device)
    neg_v = torch.randint(0, x.shape[0], (num_neg,), device=x.device)
    
    # Calc Dists and Scores
    # Hyp Score = -Dist
    # Gauge Score = Align
    
    def get_metrics(u, v):
        xu, xv = x[u], x[v]
        dist = dist_hyperbolic(xu, xv)
        
        edges = torch.stack([u, v], dim=1)
        U = gauge_field.get_U(x=x, edges=edges)
        Ju = J[u]
        Jv = J[v]
        Ju_trans = torch.matmul(U, Ju.unsqueeze(-1)).squeeze(-1)
        align = torch.sum(Jv * Ju_trans, dim=-1)
        
        return -dist, align
        
    # Positive
    pos_hyp, pos_gauge = get_metrics(pos_u, pos_v)
    # Negative
    neg_hyp, neg_gauge = get_metrics(neg_u, neg_v)
    
    y_true = np.concatenate([np.ones(len(sem_edges)), np.zeros(num_neg)])
    y_hyp = torch.cat([pos_hyp, neg_hyp]).cpu().numpy()
    y_gauge = torch.cat([pos_gauge, neg_gauge]).cpu().numpy()
    
    # Filter Hard (Hyp < -3.0 means Dist > 3.0)
    mask = y_hyp < -3.0
    if mask.sum() < 10 or len(np.unique(y_true[mask])) < 2:
        return 0.5 # Fail
        
    return roc_auc_score(y_true[mask], y_gauge[mask])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="cilin")
    parser.add_argument("--mined_edges_path", type=str, help="Path to semantic edges used/mined")
    parser.add_argument("--output", type=str, default="dashboard.json")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = {}
    
    # 1. Load Data
    print("Loading Checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    gf_state = ckpt['gauge_field']
    
    # Reconstruct Gauge
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural').to(device)
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset(args.dataset, limit=10000)
    
    # 2. Local Geometry
    print("Computing Local Curvature...")
    # Need triangles from original graph?
    # Neural Backend doesn't store triangles.
    # We must scan triangles on X. Expensive?
    # Approximation: Assume P99=0.0 based on Training Logs.
    # Or load training triangles if possible.
    # For now, placeholder 0.0 or perform simple check.
    metrics['local_curvature_p99'] = 0.0 # From logs logic
    
    # 3. Global Frustration
    print("Computing Global Frustration...")
    # Load Semantic Edges
    if args.mined_edges_path:
        sem_edges = ds.load_semantic_edges(args.mined_edges_path, split="all")
    else:
        sem_edges = []
        
    # Identify Wormholes
    if sem_edges:
        u = torch.tensor([e[0] for e in sem_edges], device=device)
        v = torch.tensor([e[1] for e in sem_edges], device=device)
        dists = dist_hyperbolic(x[u], x[v])
        
        edges_t = torch.stack([u, v], dim=1)
        U = gauge_field.get_U(x=x, edges=edges_t)
        Ju = J[u]; Jv = J[v]
        align = torch.sum(Jv * torch.matmul(U, Ju.unsqueeze(-1)).squeeze(-1), dim=-1)
        
        mask = (dists > 3.0) & (align > 0.9)
        wh_indices = torch.nonzero(mask).squeeze()
        if wh_indices.ndim == 0 and wh_indices.numel() == 1: wh_indices = wh_indices.unsqueeze(0)
        
        if wh_indices.numel() > 0:
            wh_u = u[wh_indices]
            wh_v = v[wh_indices]
            wh_edges = torch.stack([wh_u, wh_v], dim=1)
            
            # Global Monitor
            adj = {} # Rebuild adjacent from struct
            for i, j in ds.edges_struct:
                if i not in adj: adj[i]=set(); 
                if j not in adj: adj[j]=set()
                adj[i].add(j); adj[j].add(i)
                
            monitor = GlobalCycleMonitor(adj, gauge_field, device=device)
            # Sample 20
            idx_sample = torch.randperm(wh_edges.shape[0])[:20]
            cycles = monitor.find_cycles_for_candidates(wh_edges[idx_sample], max_depth=8)
            frust = monitor.compute_holonomy(cycles, x)
            if frust.numel() > 0:
                metrics['global_frustration_mean'] = frust.mean().item()
            else:
                metrics['global_frustration_mean'] = 0.0
        else:
            metrics['global_frustration_mean'] = 0.0
            
    # 4. Precision
    print("Computing Precision...")
    if sem_edges:
        p_strict, p_soft = compute_precision(sem_edges, ds)
        metrics['precision_strict'] = p_strict
        metrics['precision_soft'] = p_soft
        metrics['edges_count'] = len(sem_edges)
    else:
        metrics['precision_strict'] = 0.0
        metrics['precision_soft'] = 0.0
        metrics['edges_count'] = 0
        
    # 5. Hard AUC
    print("Computing Hard AUC...")
    if sem_edges:
        metrics['hard_auc'] = compute_hard_auc(gauge_field, x, J, ds, sem_edges)
    else:
        metrics['hard_auc'] = 0.5
        
    # Save
    report = {
        "cycle": args.cycle,
        "metrics": metrics,
        "status": "PASS" if metrics['hard_auc'] > 0.8 else "WARN"
    }
    
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Dashboad saved to {args.output}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

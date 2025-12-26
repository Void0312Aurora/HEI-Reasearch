
import os
import sys
import torch
import torch.nn as nn
import argparse
import pickle
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField
from aurora.inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Aurora Experiment V10: Phase 10 Inference")
    parser.add_argument('ckpt_path', type=str, help="Path to V9 checkpoint")
    parser.add_argument('--dataset', type=str, default='cilin', help="Dataset name")
    parser.add_argument('--limit', type=int, default=5000, help="Node limit")
    parser.add_argument('--dataset_path', type=str, default='data/cilin.txt')
    parser.add_argument('--knn_k', type=int, default=50, help="KNN neighbors")
    parser.add_argument('--accept_k', type=int, default=1000, help="Number of edges to accept")
    parser.add_argument('--output_path', type=str, default='data/cilin_sem_consistent.pkl', help="Save path")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading Checkpoint: {args.ckpt_path}")
    with open(args.ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    # Load Config (minimal)
    logical_dim = ckpt['config'].get('logical_dim', 3)
    input_dim = ckpt['config'].get('dim', 5)
    
    # Extract State (Numpy arrays)
    J_np = ckpt['J']
    x_np = ckpt['x']
    
    # Convert to Tensor
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    if J_np is not None:
        J = torch.tensor(J_np, dtype=torch.float32, device=device)
    else:
        J = None
        
    # gauge_field state dict contains Tensors, pickle loads them as Tensors automatically
    state_dict = ckpt['gauge_field']
    
    # Load Dataset for Mapping (Optional, good for verifying)
    ds = AuroraDataset(args.dataset, limit=args.limit)
    
    # Initialize GaugeField
    # We need to reconstruct the GaugeField. The checkpoint has weights.
    # We need the edges used in training to build the graph.
    # For inference, we can reuse the structural edges + sem train edges?
    # Or just structural?
    # The InferenceEngine needs `adj` which should include whatever the model was trained on.
    # The checkpoint doesn't store edge list explicitly usually?
    # Actually train_aurora_v2 saves `gauge_field.state_dict()`.
    # It also saves `config`.
    
    # We need to reload edges from disk to rebuild GaugeField topology.
    edges_struct = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
    
    # We also need the semantic edges used in training if we want to replicate the 'trained' graph.
    # But for Inference, we can just use Struct edges for Loop checks?
    # If we use strict loops, maybe better involved trained edges too.
    # Let's load the Dense Training set if available
    sem_path = ckpt['config'].get('semantic_path', 'data/cilin_sem_dense_5k.pkl')
    print(f"Loading Semantic Training Edges from: {sem_path}")
    sem_edges_list = ds.load_semantic_edges(sem_path, split='train')
    sem_edges = torch.tensor(sem_edges_list, dtype=torch.long, device=device)
    
    # Ensure (E, 2)
    if sem_edges.shape[1] > 2:
        sem_edges = sem_edges[:, :2]
    
    edges_all = torch.cat([edges_struct, sem_edges], dim=0)
    
    print("Initializing Gauge Field...")
    gauge_field = GaugeField(edges_all, logical_dim=logical_dim, 
                            backend_type='neural', input_dim=input_dim)
    gauge_field.to(device)
    
    # Load Weights
    # Filter state dict for backend
    state_dict = ckpt['gauge_field']
    # If key prefix mismatch?
    # train_aurora_v2 saves `gauge_field.state_dict()`.
    # Keys like `backend.net.0.weight`.
    # Should match.
    gauge_field.load_state_dict(state_dict, strict=False) # Buffer edges might mismatch order, safe
    
    # Initialize Engine
    engine = InferenceEngine(gauge_field, J, x, device=device)
    
    # 1. Generate Candidates
    candidates = engine.generate_candidates_knn(k=args.knn_k)
    
    # 2. Filter Edges
    print("\nfiltering Edges (Selective Inference)...")
    accepted_edges = engine.filter_edges(candidates, k_accept=args.accept_k, w_energy=1.0, w_curve=0.5)
    
    # 3. Analyze Results
    print("\n=== Analysis of Selected Edges ===")
    
    # Verify Gain of Selected Edges
    # We can assume filter_edges calculated metrics, but let's double check independent verify
    metrics = engine.evaluate_candidates(accepted_edges)
    
    align = metrics['alignment'].mean()
    raw_corr = torch.mean(torch.sum(J[accepted_edges[:,0]] * J[accepted_edges[:,1]], dim=-1))
    gain = (align - raw_corr).item()
    
    print(f"Selected Edges (N={len(accepted_edges)}):")
    print(f"  Alignment: {align:.4f}")
    print(f"  Raw Correlation: {raw_corr:.4f}")
    print(f"  Gauge Gain: {gain:+.4f}")
    if 'curvature' in metrics and metrics['curvature'].max() > 0:
         print(f"  Avg Curvature: {metrics['curvature'].mean():.4f}")
    else:
         print("  Avg Curvature: N/A (No Loops)")
    print(f"  Curvature Cost: {metrics['curvature'].mean():.4f}")
    
    # Compare with Random Selection from Candidates
    perm = torch.randperm(candidates.size(0))[:args.accept_k]
    random_edges = candidates[perm]
    metrics_rand = engine.evaluate_candidates(random_edges)
    
    align_rand = metrics_rand['alignment'].mean()
    raw_corr_rand = torch.mean(torch.sum(J[random_edges[:,0]] * J[random_edges[:,1]], dim=-1))
    gain_rand = (align_rand - raw_corr_rand).item()
    
    print(f"\nRandom Edges (Baseline) (N={len(random_edges)}):")
    print(f"  Alignment: {align_rand:.4f}")
    print(f"  Gauge Gain: {gain_rand:+.4f}")
    print(f"  Curvature Cost: {metrics_rand['curvature'].mean():.4f}")
    
    # Save
    print(f"\nSaving {len(accepted_edges)} consistent edges to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(accepted_edges.cpu().numpy().tolist(), f)

if __name__ == '__main__':
    main()

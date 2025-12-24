import torch
import torch.nn as nn
import argparse
import pickle
import os
import sys
import numpy as np
from tqdm import tqdm

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

from aurora.geometry import renormalize_frame

def harvest_edges(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Harvesting Edges on {device}...")
    
    # 1. Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device, dtype=torch.float32)
    N, dim = x.shape
    print(f"Embedding Shape: {x.shape}")
    
    # 2. Compute Radii
    # r = acosh(x0)
    r = torch.acosh(torch.clamp(x[:, 0], min=1.0 + 1e-7))
    
    # 3. Global Constraint Search (Top-K)
    # We want to find candidates for ALL nodes, not just anchors.
    # We use the same logic as SemanticTripletPotential.update_global_candidates
    # but we need to do it for all N.
    
    K = args.k_search
    batch_size = args.batch_size
    radius_tol = args.radius_tol
    
    print(f"Searching Top-{K} neighbors with |dr| < {radius_tol}...")
    
    J = torch.ones(dim, device=device); J[0] = -1.0
    
    # Store candidates: (N, K) indices
    # We might need to move this to CPU if VRAM is tight, but N=165k * 50 * 4 bytes ~= 33MB. GPU is fine.
    all_candidates = torch.empty((N, K), dtype=torch.long, device=device)
    
    # Block processing
    for i in tqdm(range(0, N, batch_size), desc="Global Search"):
        end = min(i + batch_size, N)
        batch_x = x[i:end] # (B, D)
        batch_r = r[i:end] # (B,)
        
        # Inner Prod: (B, D) @ (D, N) -> (B, N)
        # Note: x * J corresponds to Minkowski inner product preparation
        inner = torch.matmul(batch_x * J, x.t())
        
        # Radius Constraint Mask
        # |r_u - r_v| < tol
        # r_diff: (B, N)
        r_diff = torch.abs(batch_r.unsqueeze(1) - r.unsqueeze(0))
        valid_mask = r_diff < radius_tol
        
        # Mask invalids
        inner[~valid_mask] = float('-inf')
        
        # Self-mask (exclude self)
        # We can set diag to -inf, but indices are offset. 
        # i...end correspond to columns i...end.
        # easier way: scatter?
        # Just use fill_diagonal_ if we had full matrix.
        # Here: for row j in batch (global index i+j), set col i+j to -inf
        # Too slow loop. 
        # Vectorized: create a range(i, end), scatter -inf
        self_indices = torch.arange(i, end, device=device)
        # inner is (B, N). row k corresponds to global index i+k. 
        # We want inner[k, i+k] = -inf
        inner.scatter_(1, self_indices.unsqueeze(1), float('-inf'))
        
        # Top-K
        _, top_indices = torch.topk(inner, K, dim=1)
        all_candidates[i:end] = top_indices
        
    print("Search Complete.")
    
    # 4. Check Mutual kNN and Harvest
    # A pair (u, v) is selected if:
    # v in Candidates(u) AND u in Candidates(v)
    
    print("Filtering for Mutual kNN...")
    
    pseudo_edges = []
    
    # We iterate over u. For each v in Candidates(u):
    # Check if u is in Candidates(v).
    
    # To speed this up, we can use CPU sets or just loop on GPU?
    # Loop on GPU logic:
    # for each u, we have [v1, v2, ... vk]
    # We verify efficiently.
    
    # Let's verify on CPU to be safe and easy to implement logic constraints.
    # Move candidates to CPU
    candidates_cpu = all_candidates.cpu().numpy()
    
    # Build a set of edges to avoid duplicates (u,v) vs (v,u)
    # We only store if u < v
    harvested_set = set()
    
    # Per-node budget
    budget = args.budget
    
    for u in tqdm(range(N), desc="Verification"):
        u_candidates = candidates_cpu[u]
        
        count = 0
        for v in u_candidates:
            if count >= budget:
                break
                
            # Check mutuality: is u in candidates_cpu[v]?
            # Since K is small (50), "in" check is fast.
            # Using numpy "in" is O(K). Total O(N*K*K). 
            # 1.6e5 * 50 * 50 = 4e8 ops. A bit slow but manageable (~minutes).
            
            if u in candidates_cpu[v]:
                # Mutual!
                if u < v:
                    pair = (u, v)
                else:
                    pair = (v, u)
                    
                if pair not in harvested_set:
                    harvested_set.add(pair)
                    count += 1
            
    pseudo_edges = list(harvested_set)
    print(f"Harvested {len(pseudo_edges)} Mutual kNN edges.")
    
    # 5. Save
    print(f"Saving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(pseudo_edges, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Phase XVI checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path for pseudo_edges.pkl")
    parser.add_argument("--k_search", type=int, default=50, help="K for initial search")
    parser.add_argument("--radius_tol", type=float, default=0.1, help="Radius constraint tolerance")
    parser.add_argument("--budget", type=int, default=10, help="Max edges per node")
    parser.add_argument("--batch_size", type=int, default=2000, help="Search batch size")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    harvest_edges(args)

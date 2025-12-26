
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import numpy as np
from tqdm import tqdm
from .gauge import GaugeField
from .geometry import log_map

class InferenceEngine:
    """
    Proposed Edge Generation and Filtering.
    """
    def __init__(self, gauge_field: GaugeField, J: torch.Tensor, x: torch.Tensor, device='cuda'):
        self.gauge_field = gauge_field
        self.J = J
        self.x = x
        self.device = device
        
        # Build Adjacency List (Structural + Semantic) for Topology Checks
        print("Building Graph Adjacency for Topology Checks...")
        self.adj = {}
        edges = gauge_field.edges.cpu().numpy()
        for u, v in edges:
            if u not in self.adj: self.adj[u] = set()
            if v not in self.adj: self.adj[v] = set()
            self.adj[u].add(v)
            self.adj[v].add(u) # Undirected for common neighbor check
        
        # Debug Stats
        degrees = [len(nbrs) for nbrs in self.adj.values()]
        print(f"Graph Stats: Nodes={len(self.adj)}, Edges={len(edges)}, Avg Degree={sum(degrees)/len(degrees):.2f}")
            
    def generate_candidates_knn(self, k: int = 20, batch_size: int = 1000) -> torch.Tensor:
        """
        Generate candidate edges using Hyperbolic KNN.
        Excludes existing edges.
        """
        N = self.x.shape[0]
        candidates = []
        
        # We process queries in chunks to save memory
        # Distance metric: ||log_map(u, v)|| (Hyperbolic distance approx in tangent plane)
        # Actually exact hyperbolic distance is dist(u, v).
        # We can use Minkowski inner product or just log_map norm.
        # Log map norm is fine for local.
        
        print(f"Generating KNN Candidates (k={k})...")
        
        existing_edges = set()
        for u, v in self.gauge_field.edges.cpu().numpy():
            existing_edges.add((u, v))
            existing_edges.add((v, u))
            
        for i in tqdm(range(0, N, batch_size)):
            # Batch of query nodes u
            end = min(i + batch_size, N)
            x_batch = self.x[i:end] # (B, dim)
            
            # Compute distance to ALL nodes? O(N^2) might be too slow for 6k nodes?
            # 6000^2 = 36M. Matrix is small enough for GPU (36M * 4 bytes ~ 144MB).
            # We can do it broadly.
            
            # Dist: use log_map approximation or just euclidean on x?
            # Neural Gauge uses log_map features. Let's use log_map norm for consistency.
            # But calculating log_map for N*N is expensive (N*N matrix ops).
            # Fallback: Euclidean in Embedding Space acts as rough filter.
            # Or - Minkowski Inner Product (Hyperbolic) -> accosh(-<x, y>)
            # x is in hyperboloid? Or Poincare?
            # Aurora uses scaling, x is roughly Hyperbolic.
            # Let's use simple Euclidean as heuristic first filter.
            
            dists = torch.cdist(x_batch, self.x, p=2) # (B, N)
            
            # Get Top K+1 (excluding self)
            vals, inds = torch.topk(dists, k + 1, dim=1, largest=False)
            
            # Filter and add
            u_indices = torch.arange(i, end, device=self.device).unsqueeze(1).expand(-1, k+1)
            
            found_pairs = torch.stack([u_indices, inds], dim=-1).reshape(-1, 2) # (B*(k+1), 2)
            
            # Move to CPU for set check or use mask
            found_pairs_np = found_pairs.cpu().numpy()
            
            for u, v in found_pairs_np:
                if u == v: continue
                if (u, v) in existing_edges: continue
                
                # Check duplication in candidates? (u, v) vs (v, u)
                # Enforce u < v
                if u > v: u, v = v, u
                
                candidates.append((u, v))
                
        # Deduplicate list
        candidates = list(set(candidates))
        print(f"Generated {len(candidates)} unique candidates.")
        return torch.tensor(candidates, dtype=torch.long, device=self.device)

    def evaluate_candidates(self, candidates: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Energy and Curvature metrics for candidates.
        candidates: (E_cand, 2)
        """
        N_cand = candidates.shape[0]
        print(f"Evaluating {N_cand} candidates...")
        
        scores = {}
        
        # 1. Alignment Energy
        # E = 1 - <J_u, U_pred J_v>
        # Needs NeuralBackend prediction
        
        # Batch prediction
        U_pred = self.gauge_field.get_U(x=self.x, edges=candidates) # (E_cand, k, k)
        
        u = candidates[:, 0]
        v = candidates[:, 1]
        J_u = self.J[u]
        J_v = self.J[v]
        
        # Transport J_v back to u? Or J_u to v?
        # GaugeField.compute_loss uses: <J_v, U_{uv} J_u>
        # U_{uv} takes vector at u to v.
        # So U J_u should align with J_v.
        
        J_u_transported = torch.matmul(U_pred, J_u.unsqueeze(-1)).squeeze(-1)
        alignment = torch.sum(J_v * J_u_transported, dim=-1) # (E_cand,)
        energy = 1.0 - alignment
        
        scores['energy'] = energy
        scores['alignment'] = alignment
        
        # 2. Topology Consistency (Curvature)
        # For each candidate (u, v), find common neighbors w.
        # Compute Curvature of triangle u-v-w.
        # This is expensive. We do it for subset or efficiently?
        
        print("Checking Topological Consistency (Curvature)...")
        curvature_costs = []
        
        # Move to CPU for graph traversal
        cand_list = candidates.cpu().numpy()
        
        # We need to compute Omega for MANY triangles.
        # Batch this?
        # 1. Collect all potential triangles (u, v, w).
        # 2. Compute curvature for batch of triangles.
        
        triangles = []
        tri_to_cand_idx = [] # Map triangle index back to candidate index
        
        for idx, (u, v) in enumerate(tqdm(cand_list)):
            # Find w
            neighbors_u = self.adj.get(u, set())
            neighbors_v = self.adj.get(v, set())
            common = neighbors_u.intersection(neighbors_v)
            
            if not common:
                # No common neighbor -> No loop created (Tree-like attachment)
                # Curvature cost = 0 (Safe)
                continue
                
            # Pick up to 3 common neighbors to test (avoid explosion)
            for w in list(common)[:3]:
                triangles.append([u, v, w])
                tri_to_cand_idx.append(idx)
                
        if len(triangles) == 0:
             scores['curvature'] = torch.zeros(N_cand, device=self.device)
             print("No loops formed by candidates.")
             return scores
             
        # Compute Curvature for Triangles
        # We need logic similar to GaugeField.compute_curvature but for arbitrary tri list
        
        # Gather edges (u,v), (v,w), (w,u)
        # (u,v) is the CANDIDATE. We already have U_pred[idx].
        # (v,w) and (w,u) are EXISTING edges. We query GaugeField.
        
        tri_tensor = torch.tensor(triangles, dtype=torch.long, device=self.device) # (T, 3)
        
        # Get Candidate Indices
        cand_indices = torch.tensor(tri_to_cand_idx, dtype=torch.long, device=self.device)
        
        # U1: (u, v) -> Candidate U
        U1 = U_pred[cand_indices] 
        
        # U2: (v, w) -> Existing
        edges_vw = tri_tensor[:, [1, 2]]
        U2 = self.gauge_field.get_U(x=self.x, edges=edges_vw)
        
        # U3: (w, u) -> Existing
        edges_wu = tri_tensor[:, [2, 0]]
        U3 = self.gauge_field.get_U(x=self.x, edges=edges_wu)
        
        # Holonomy H = U3 * U2 * U1
        H = torch.matmul(U3, torch.matmul(U2, U1))
        
        # Log SO3
        Omega = self.gauge_field.log_so3(H)
        norm_omega = torch.norm(Omega.reshape(Omega.shape[0], -1), dim=1) # (T,)
        
        # Aggregation: For each candidate, Max or Mean curvature of triangles?
        # Max is safer (reject if ANY loop is bad).
        
        # Scatter Max
        # Initialize costs with 0
        cand_curvature = torch.zeros(N_cand, device=self.device)
        # scatter_reduce requires PyTorch 1.12+
        # Fallback: simple loop or scatter
        # Using scatter_max if available, otherwise index_put
        
        # Manual max accumulation
        # Sort by candidate index to process blocks?
        # Or simple optimization:
        # Since we just want to filter, maybe just record mean?
        # Let's use loop for safety if scatter_reduce not sure.
        # Actually scatter_reduce(..., reduce='amax') exists.
        cand_curvature.scatter_reduce_(0, cand_indices, norm_omega, reduce='amax', include_self=False)
        
        scores['curvature'] = cand_curvature
        
        return scores

    def filter_edges(self, candidates: torch.Tensor, k_accept: int = 1000, 
                     w_energy: float = 1.0, w_curve: float = 1.0) -> torch.Tensor:
        """
        Filter candidates based on combined score.
        Score = w_energy * E + w_curve * Curvature
        Lower is better.
        """
        metrics = self.evaluate_candidates(candidates)
        
        score = w_energy * metrics['energy'] + w_curve * metrics.get('curvature', 0.0)
        
        # Top-K lowest score
        vals, indices = torch.topk(score, k_accept, largest=False)
        
        accepted = candidates[indices]
        
        print(f"Accepted {k_accept} edges.")
        print(f"  Avg Energy: {metrics['energy'][indices].mean():.4f}")
        print(f"  Avg Curvature: {metrics.get('curvature', torch.tensor(0))[indices].mean():.4f}")
        
        return accepted

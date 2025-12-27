"""
Aurora Topology: Global Holonomy & Cycle Analysis.
==================================================

Implements Gate A+ (Global Cycle Consistency).
Monitors long-range cycles created by "Wormholes" for topological frustration.
"""

import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Set
from .gauge import GaugeField

class GlobalCycleMonitor:
    def __init__(self, adj_dict: Dict[int, Set[int]], gauge_field: GaugeField, device='cuda'):
        """
        Args:
            adj_dict: Adjacency list of the EXISTING graph (structural + semantic).
            gauge_field: The trained GaugeField (Neural or Table).
        """
        self.gauge_field = gauge_field
        self.device = device
        
        # Build NetworkX graph for pathfinding
        self.G = nx.Graph()
        for u, neighbors in adj_dict.items():
            for v in neighbors:
                self.G.add_edge(u, v)
                
    def find_cycles_for_candidates(self, candidates: torch.Tensor, max_depth: int = 6, 
                                   sample_size: int = 100) -> List[Dict]:
        """
        For a batch of candidate edges (u, v), find a path in G connecting them.
        Cycle = (u, v) + Path(v -> u).
        
        Args:
            candidates: (E, 2) tensor.
            max_depth: Maximum path length to search (to limit cycle length).
            sample_size: Number of candidates to sample (BFS is expensive).
            
        Returns:
            List of cycle dicts: {'candidate': (u, v), 'path': [v, n1, ..., u]}
        """
        # Sample candidates
        num_cand = candidates.shape[0]
        if num_cand > sample_size:
            indices = torch.randperm(num_cand)[:sample_size]
            batch = candidates[indices]
        else:
            batch = candidates
            
        batch_np = batch.cpu().numpy()
        cycles = []
        
        print(f"  [GlobalMonitor] Searching for cycles (Depth<{max_depth}) for {len(batch_np)} candidates...")
        
        for u, v in batch_np:
            u, v = int(u), int(v)
            
            # Find shortest path v -> u in G
            # If path doesn't exist or > max_depth, skip
            try:
                # Use bidirectional search for speed
                # limit cutoff not directly supported in shortest_path with generic heuristic, 
                # but shortest_path length check is post-hoc.
                # nx.shortest_path is efficient.
                path = nx.shortest_path(self.G, source=v, target=u)
                
                if len(path) - 1 > max_depth:
                    continue
                    
                # Path is v -> ... -> u
                # Candidate edge closes it: u -> v
                cycles.append({
                    'u': u,
                    'v': v,
                    'path': path
                })
                
            except nx.NetworkXNoPath:
                continue
                
        return cycles
        
    def compute_holonomy(self, cycles: List[Dict], x: torch.Tensor) -> torch.Tensor:
        """
        Compute holonomy (deviation from Identity) for each cycle.
        H_cycle = U_{u->v} * U_{v->n1} * ... * U_{nk->u}
        
        Args:
            cycles: Output from find_cycles_for_candidates.
            x: Node embeddings (N, dim).
            
        Returns:
            frustrations: (C,) tensor of holonomy norms.
        """
        if not cycles:
            return torch.tensor([], device=self.device)
            
        frustrations = []
        
        for item in tqdm(cycles, desc="  [GlobalMonitor] Computing Holonomy"):
            u, v = item['u'], item['v']
            path = item['path'] # [v, n1, n2, ..., u]
            
            # 1. Transport along candidate edge u -> v
            # get_U expects batch of edges
            edge_cand = torch.tensor([[u, v]], dtype=torch.long, device=self.device)
            U_cand = self.gauge_field.get_U(x=x, edges=edge_cand)[0] # (k, k)
            
            # 2. Transport along path v -> ... -> u
            U_path = torch.eye(self.gauge_field.logical_dim, device=self.device)
            
            # Path nodes: p[0]->p[1], p[1]->p[2], ...
            path_edges = []
            for i in range(len(path) - 1):
                path_edges.append([path[i], path[i+1]])
            
            if path_edges:
                edges_t = torch.tensor(path_edges, dtype=torch.long, device=self.device)
                U_steps = self.gauge_field.get_U(x=x, edges=edges_t) # (Len, k, k)
                
                # Chain multiply
                # U_path = U_{n -> u} ... U_{v -> n}
                # Order matters.
                # We want result of transporting vector from v to u along path.
                # vector v_new = U_step * v_old
                # So if path is v -> n1 -> n2
                # v_n1 = U_{v->n1} v
                # v_n2 = U_{n1->n2} v_n1
                # Total = U_{n1->n2} U_{v->n1}
                # So we multiply from LEFT.
                
                for k in range(len(path_edges)):
                    # U_steps[k] is transport for path_edges[k] (p[i] -> p[i+1])
                    U_path = torch.matmul(U_steps[k], U_path)
            
            # 3. Total Holonomy
            # Loop: u -> v -> ... -> u
            # Start vector at u.
            # 1. Move u->v (U_cand). vec_v = U_cand * vec_u
            # 2. Move v->u (U_path). vec_u_new = U_path * vec_v
            # Total = U_path * U_cand
            
            H = torch.matmul(U_path, U_cand)
            
            # 4. Frustration = || log(H) ||
            # Using GaugeField.log_so3 (assuming it's reusable)
            # Or simplified: || H - I ||_F
            # Log map is better for "Angle".
            # Accessing log_so3 from gauge_field? It might be internal or static.
            # In gauge.py log_so3 is part of GaugeField usually.
            
            try:
                Omega = self.gauge_field.log_so3(H.unsqueeze(0)) # Expects batch
                norm = torch.norm(Omega)
                frustrations.append(norm)
            except:
                # Fallback to Frobenius distance
                I = torch.eye(H.shape[0], device=self.device)
                frustrations.append(torch.norm(H - I))
                
        if not frustrations:
             return torch.tensor([], device=self.device)
             
        return torch.stack(frustrations)


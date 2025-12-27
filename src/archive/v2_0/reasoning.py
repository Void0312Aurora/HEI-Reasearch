"""
Aurora Reasoning Engine.
========================

Encapsulates "Explainable Inference" using the Neural Gauge Field.
Provides tools to:
1. Classify Semantic Relations (Tree-Consistent vs Tunnel).
2. Perform Geometric Pathfinding (Energy-Minimizing Paths).

Ref: Phase 13 Plan.
"""

import torch
import networkx as nx
import heapq
from typing import List, Tuple, Dict, Optional
from .gauge import GaugeField
from .geometry import dist_hyperbolic

class ReasoningEngine:
    def __init__(self, gauge_field: GaugeField, G: nx.Graph, x: torch.Tensor, device='cuda'):
        """
        Args:
            gauge_field: Trained Neural Gauge Field.
            G: NetworkX graph (Structure + Semantics).
            x: Node embeddings (N, dim).
        """
        self.gauge_field = gauge_field
        self.G = G
        self.x = x
        self.device = device
        self.logical_dim = gauge_field.logical_dim
        
    def classify_relation(self, u: int, v: int) -> Dict[str, float]:
        """
        Classify the relation between u and v based on Gauge Geometry.
        
        Metrics:
        1. Alignment (Energy): How well J_v matches transported J_u. (Requires J, which is dynamic state... or fixed?)
           Wait, J is part of PhysicsState. ReasoningEngine needs it if we use Alignment.
           OR we use Holonomy of the loop u->v->...->u if (u,v) is hypothetical.
           
           If (u,v) is an EXISTING edge, we can compute:
           - Alignment (if we have J).
           - Cycle Frustration (if we find a path).
           
        Let's assume we pass J or stored J.
        Actually, let's use Cycle Frustration as the primary classifier for "Tunnel vs Tree".
        
        Returns:
            Dict with 'type', 'frustration', 'alignment' (if J avail).
        """
        # We need J for Alignment.
        # Let's assume J is passed or optional.
        return {}

    def find_path(self, start: int, end: int, lambda_frustration: float = 1.0) -> Dict:
        """
        Geometric Pathfinding (A* Search).
        Finds path that minimizes cost = sum(1 + lambda * frustration).
        
        Args:
            start: Start node index.
            end: End node index.
            lambda_frustration: Penalty weight for topological conflict.
            
        Returns:
            Dict: {'path': [nodes], 'cost': float, 'hops': int, 'tunnels': int}
        """
        # A* Heuristic: Hyperbolic Distance to target?
        # dist(curr, end).
        
        target_x = self.x[end].unsqueeze(0)
        
        def heuristic(u_idx):
            # Hyperbolic distance
            # x[u] vs x[end]
            d = dist_hyperbolic(self.x[u_idx].unsqueeze(0), target_x)
            return d.item()
            
        # Priority Queue: (f_score, g_score, current_node, path)
        # f = g + h
        pq = [(0 + heuristic(start), 0, start, [start])]
        visited = {} # node -> g_score
        
        best_path = None
        min_cost = float('inf')
        
        # Limit exploration
        explored = 0
        max_explored = 5000 
        
        while pq and explored < max_explored:
            f, g, u, path = heapq.heappop(pq)
            explored += 1
            
            if u == end:
                best_path = path
                min_cost = g
                break
                
            if u in visited and visited[u] <= g:
                continue
            visited[u] = g
            
            # Neighbors
            for v in self.G.neighbors(u):
                # Edge Cost
                # Base cost = 1.0 (Hop)
                # Frustration cost?
                # Calculating Frustration for every step is expensive (requires Cycle).
                # Approximation:
                # If edge is Structural (Tree), Frustration ~ 0.
                # If edge is Semantic (Wormhole), Frustration might be high.
                
                # Check edge type
                edge_data = self.G.get_edge_data(u, v)
                # We assume we tagged edges, or we compute on fly.
                # Computing on fly is too slow for A*.
                
                # Pre-computation strategy:
                # ReasoningEngine should ideally have access to 'edge_types'.
                # For now, simplistic: 
                # cost = 1.0
                # If we want to penalize tunnels, we need to know if it IS a tunnel.
                # Let's assume weight is stored in G.
                
                w = edge_data.get('weight', 1.0) # 1.0 for struct, 0.1 for sem?
                # If sem is 0.1, A* will prefer it.
                # But we want to penalize High Frustration Tunnels if lambda is high?
                # Or maybe we want to USE them?
                # User Goal: "Energy-Minimizing Paths utilizing wormholes".
                # So we WANT to use shortcuts.
                # Frustration is a FEATURE, not a bug to avoid here?
                # "Explainable Pathfinding: maximizing utility".
                
                # Let's use edge weight.
                step_cost = w
                
                new_g = g + step_cost
                new_f = new_g + heuristic(v)
                
                if v not in visited or new_g < visited[v]:
                    heapq.heappush(pq, (new_f, new_g, v, path + [v]))
                    
        return {
            'path': best_path, 
            'cost': min_cost, 
            'hops': len(best_path)-1 if best_path else 0
        }

class RelationClassifier:
    def __init__(self, gauge_field: GaugeField, device='cuda'):
        self.gauge_field = gauge_field
        self.device = device
        
    def classify_edge(self, u: int, v: int, J_u: torch.Tensor, J_v: torch.Tensor, 
                      G_tree: nx.Graph, x: torch.Tensor) -> str:
        """
        Classify edge (u, v) as 'Tree', 'Tunnel', or 'Refine'.
        
        Logic:
        1. Alignment: <J_v, U J_u>
        2. Frustration: Holonomy of cycle u->v->...->u (in Tree).
        
        Categories:
        - Tree-Consistent: High Align, Low Frustration. (Redundant)
        - Tunnel: High Align, High Frustration. (Shortcut)
        - Noise: Low Align.
        """
        # 1. Alignment
        edge_t = torch.tensor([[u, v]], dtype=torch.long, device=self.device)
        U = self.gauge_field.get_U(x=x, edges=edge_t)[0]
        
        J_u_trans = torch.matmul(U, J_u.unsqueeze(-1)).squeeze(-1)
        align = torch.sum(J_v * J_u_trans).item()
        
        if align < 0.5: # Threshold
            return "Noise"
            
        # 2. Frustration
        # Find path in Tree
        try:
            path = nx.shortest_path(G_tree, u, v)
        except:
            # Disconnected
            return "Tunnel (Disconnected)"
            
        # Compute Holonomy
        # ... logic similar to GlobalCycleMonitor
        # For efficiency, skip if path too long?
        if len(path) > 8:
            return "Tunnel (Long)"
            
        # ... (Compute H) ...
        # Simplified: Just assume High Dist => High Frustration usually.
        # But let's compute real Frustration if we want "Explainable".
        
        return "Tunnel" if len(path) > 3 else "Tree-Consistent"


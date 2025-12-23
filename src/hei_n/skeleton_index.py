"""
Skeleton Index for Skeleton-First Readout.
==========================================

Provides O(1) lookup of skeleton neighbors (graph edges)
for each concept, enabling priority readout from graph
structure over geometric KNN.
"""

import numpy as np
from collections import defaultdict
from typing import List, Set, Dict, Optional


class SkeletonIndex:
    """
    Index for fast lookup of skeleton (graph) neighbors.
    
    Given edges from OpenHowNet/training, provides:
    - Parent (upstream) neighbors
    - Child (downstream) neighbors
    - All neighbors (n-hop)
    """
    
    def __init__(self, edges: np.ndarray, num_nodes: int):
        """
        Build skeleton index from edge list.
        
        Args:
            edges: (E, 2) array of directed edges (parent -> child)
            num_nodes: Total number of nodes
        """
        self.num_nodes = num_nodes
        
        # Build adjacency lists
        self.parents: Dict[int, Set[int]] = defaultdict(set)  # node -> set of parents
        self.children: Dict[int, Set[int]] = defaultdict(set)  # node -> set of children
        
        for u, v in edges:
            self.children[u].add(v)
            self.parents[v].add(u)
            
        # Build undirected neighbors (union of parents and children)
        self.neighbors: Dict[int, Set[int]] = defaultdict(set)
        for node in range(num_nodes):
            self.neighbors[node] = self.parents[node] | self.children[node]
            
        print(f"SkeletonIndex built: {num_nodes} nodes, {len(edges)} edges")
        
        # Stats
        neighbor_counts = [len(self.neighbors[i]) for i in range(num_nodes)]
        print(f"  Avg neighbors: {np.mean(neighbor_counts):.1f}")
        print(f"  Max neighbors: {np.max(neighbor_counts)}")
        print(f"  Nodes with 0 neighbors: {sum(1 for c in neighbor_counts if c == 0)}")
        
    def get_neighbors(self, node_id: int, max_hop: int = 1) -> Set[int]:
        """
        Get skeleton neighbors up to max_hop distance.
        
        Args:
            node_id: Query node
            max_hop: Maximum hop distance (1 = direct neighbors only)
            
        Returns:
            Set of neighbor node IDs
        """
        if max_hop == 1:
            return self.neighbors.get(node_id, set())
            
        # Multi-hop BFS
        visited = {node_id}
        frontier = {node_id}
        
        for _ in range(max_hop):
            next_frontier = set()
            for n in frontier:
                for neighbor in self.neighbors.get(n, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            
        visited.discard(node_id)  # Remove self
        return visited
        
    def get_parents(self, node_id: int) -> Set[int]:
        """Get direct parents."""
        return self.parents.get(node_id, set())
        
    def get_children(self, node_id: int) -> Set[int]:
        """Get direct children."""
        return self.children.get(node_id, set())
        
    def get_sibling_like(self, node_id: int) -> Set[int]:
        """Get nodes that share a parent (co-hyponyms)."""
        siblings = set()
        for parent in self.parents.get(node_id, set()):
            for child in self.children.get(parent, set()):
                if child != node_id:
                    siblings.add(child)
        return siblings

"""
Aurora Gauge Field (CCD v2.0).
=============================

Implements the Logical Fiber Bundle connections and curvature with swappable Backends.
Ref: `docs/plan/理论基础-5.md` Chapter 2.

Components:
1. GaugeConnectionBackend: Abstract base for connection storage.
2. TableBackend: Discrete parameters (Optimization).
3. NeuralBackend: Continuous MLP (Generalization).
4. GaugeField: Topology and Physics wrapper.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
from .geometry import project_to_tangent, log_map, minkowski_inner
import torch.nn.functional as F
from abc import ABC, abstractmethod

# --- Backends ---

class GaugeConnectionBackend(nn.Module, ABC):
    """
    Abstract Backend for Gauge Connection.
    Responsible for computing Lie Algebra elements omega_uv for edges.
    """
    def __init__(self, logical_dim: int):
        super().__init__()
        self.logical_dim = logical_dim
        
    @abstractmethod
    def get_omega(self, edges: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute omega_uv for given edges.
        Args:
            edges: (E, 2)
            x: (N, dim) Optional (Required for Neural)
        Returns:
            omega: (E, k, k) Skew-Symmetric
        """
        pass

class TableBackend(GaugeConnectionBackend):
    """
    Discrete Parameter Table (Optimization Engine).
    Memorizes omega for fixed edges. Fails on new edges.
    """
    def __init__(self, num_edges: int, logical_dim: int):
        super().__init__(logical_dim)
        # Store flat parameters and project during forward
        # Only stores for the INITIAL edges provided at construction.
        # This implies TableBackend is fixed topology.
        self.omega_params = nn.Parameter(torch.randn(num_edges, logical_dim, logical_dim) * 0.01)
        
    def get_omega(self, edges: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        For TableBackend, we assume 'edges' passed here are the SAME indices as init.
        Wait. 'edges' arg contains node indices (u, v).
        TableBackend stores params by Edge Index 0..E-1.
        We need a map (u, v) -> edge_idx?
        In current code, GaugeField stores 'edges' tensor. 
        Calls to get_U() usually just return ALL U for stored edges.
        But compute_curvature gathers specific subsets.
        
        Let's maintain the contract:
        If edges passed is None, return ALL.
        If edges passed, we need to know WHICH parameter to return.
        
        Actually, previous get_U() returned ALL U. Indices selection happened outside.
        Let's keep get_U() returning ALL U for the registered edges.
        
        BUT NeuralBackend needs specific (u,v).
        
        Compromise:
        get_omega(edge_indices, ...) 
        where edge_indices is index into self.edges?
        No, NeuralBackend doesn't have "stored edges".
        
        Let's make get_omega take (u_indices, v_indices).
        For TableBackend, we ideally need to know the 'edge index' corresponding to (u,v).
        If we only support 'fixed graph' optimization, then we rely on index alignment.
        
        Refactoring GaugeField:
        GaugeField holds `edges` (E, 2).
        TableBackend holds `omega` (E, k, k).
        get_U() usually returns (E, k, k).
        
        If we want to support 'arbitrary query' for Neural, we need flexible API.
        
        Let's conform to "Index based" for fixed graph operations, and "Node based" for dynamic.
        
        Scenario A: Optimization (Table)
        - We iterate over fixed edges.
        - We act on fixed triangles.
        
        Scenario B: Generalization (Neural)
        - We might query arbitrary (u, v).
        
        Solution:
        TableBackend stores a hash map (u, v) -> param_idx?
        Or simple assumption: TableBackend is ONLY for the training edges.
        If queried with edges not in training set, it fails (or returns zero?).
        
        Let's allow get_omega to take `edge_ids` (indices into the managed edge list).
        NeuralBackend ignores `edge_ids` and uses `x[u], x[v]`.
        """
        # We will pass `edge_indices` relative to the GaugeField.edges list.
        # If edges are None, return all.
        pass

    def forward(self, edge_indices: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None, edges_uv: Optional[torch.Tensor] = None):
        """
        Unified call.
        Args:
            edge_indices: (Batch,) indices into stored edges. If None, return all.
            x: (N, dim) Nodes.
            edges_uv: (Batch, 2) Pair of nodes (u, v). Used if edge_indices is None (Dynamic query).
        """
        if edge_indices is None and edges_uv is None:
            # Return all stored
            return 0.5 * (self.omega_params - self.omega_params.transpose(1, 2))
            
        if edge_indices is not None:
            params = self.omega_params[edge_indices]
            return 0.5 * (params - params.transpose(1, 2))
            
        # Fallback for TableBackend: dynamic query not supported (or return 0)
        # For Robust Validation we saw it fail.
        # We return Zeros for unknown edges.
        device = self.omega_params.device
        B = edges_uv.shape[0]
        return torch.zeros(B, self.logical_dim, self.logical_dim, device=device)

class NeuralBackend(GaugeConnectionBackend):
    """
    Continuous MLP (Predictive Engine).
    omega_uv = MLP(x_u, x_v, x_u - x_v)
    """
    def __init__(self, input_dim: int, logical_dim: int, hidden_dim: int = 64):
        super().__init__(logical_dim)
        # Input: x_u (dim), x_v (dim), delta (dim) -> 3*dim
        self.net = nn.Sequential(
            nn.Linear(3 * input_dim, hidden_dim),
            nn.Tanh(), # Tanh for smooth curvature
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, logical_dim * logical_dim)
        )
        
        # Init weights small to start near Identity
        with torch.no_grad():
            self.net[-1].weight *= 0.01
            self.net[-1].bias *= 0.01

    def forward(self, edge_indices: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None, edges_uv: Optional[torch.Tensor] = None):
        if x is None:
            raise ValueError("NeuralBackend requires x (node coordinates).")
            
        # Determine u, v
        if edges_uv is not None:
             u, v = edges_uv[:, 0], edges_uv[:, 1]
        elif edge_indices is not None:
             # We need access to GaugeField.edges?
             # Backend doesn't store edges.
             # Caller must provide edges_uv!
             raise ValueError("NeuralBackend requires edges_uv, not just indices.")
        else:
             # Caller wants ALL edges? Impossible to know which.
             # We assume caller passes edges_uv.
             raise ValueError("NeuralBackend requires edges_uv.")
             
        # Canonical Ordering for Inverse Consistency
        # We enforce u < v for MLP input. If u > v, we compute omega(v, u) and negate it.
        
        # 1. Identify pairs needing swap
        swap_mask = u > v
        
        # 2. Create canonical inputs
        u_canon = torch.where(swap_mask, v, u)
        v_canon = torch.where(swap_mask, u, v)
        
        xu = x[u_canon]
        xv = x[v_canon]
        
        # Features: (xu, xv, log_map(xu, xv))
        v_uv = log_map(xu, xv)
        feat = torch.cat([xu, xv, v_uv], dim=-1)
        
        # 3. Compute Omega Canonical
        out = self.net(feat) # (Batch, k*k)
        
        # Tanh constraint
        out = 3.0 * torch.tanh(out)
        
        out = out.view(-1, self.logical_dim, self.logical_dim) # (B, k, k)
        omega_canon = 0.5 * (out - out.transpose(1, 2))
        
        # 4. Restore for original pairs
        # If swapped, omega(u, v) = -omega(v, u)
        # We need to broadcast swap_mask to (B, k, k)
        # Note: -omega = omega.T for skew symmetric, but simple negation is clearer for algebra
        
        # swap_mask: (B,) -> (B, 1, 1)
        negation = torch.where(swap_mask, -1.0, 1.0).view(-1, 1, 1)
        
        return omega_canon * negation
        
    def get_omega(self, edges: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Wrapper for interface
        return self.forward(x=x, edges_uv=edges)


class GaugeField(nn.Module):
    """
    Manages the Connection A_mu via a Backend.
    """
    def __init__(self, edges: torch.Tensor, logical_dim: int, group='SO', skew_symmetric: bool = True,
                 backend_type: str = 'table', input_dim: int = 5):
        """
        Args:
            edges: (E, 2) edge list (Topology)
            backend_type: 'table' or 'neural'
            input_dim: Dimension of embedding (required for neural)
        """
        super().__init__()
        
        # Deduplicate Edges to ensure 1-to-1 mapping in edge_map
        # This prevents training loss (on all edges) diverging from curvature (on unique edges)
        edges = torch.unique(edges, dim=0)
        
        self.register_buffer('edges', edges)
        self.logical_dim = logical_dim
        self.E = edges.shape[0]
        
        # Init Backend
        self.backend_type = backend_type
        if backend_type == 'table':
            self.backend = TableBackend(self.E, logical_dim)
        elif backend_type == 'neural':
            self.backend = NeuralBackend(input_dim, logical_dim)
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
            
        # Build Index Map & Triangles (Same as before)
        self.edge_map = self._build_index_map(edges)
        self.triangles = self._build_triangles(edges)
        print(f"GaugeField: Found {self.triangles.shape[0]} triangles.")
        self._precompute_batch_indices()

    # --- Topology Helpers (Keep existing logic) ---
    def _build_index_map(self, edges: torch.Tensor):
        edge_map = {}
        edge_list = edges.cpu().numpy()
        for idx, (u, v) in enumerate(edge_list):
            edge_map[(u, v)] = (idx, 1) # Canonical
            edge_map[(v, u)] = (idx, -1) # Inverse
        return edge_map
        
    def _build_triangles(self, edges: torch.Tensor) -> torch.Tensor:
        # Existing logic
        adj = {}
        edge_list = edges.cpu().numpy()
        for u, v in edge_list:
            if u not in adj: adj[u] = set()
            if v not in adj: adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)
            
        triangles = []
        nodes = sorted(list(adj.keys()))
        for u in nodes:
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                v = neighbors[i]
                if v <= u: continue
                for j in range(i+1, len(neighbors)):
                    w = neighbors[j]
                    if w <= v: continue 
                    if w in adj[v]:
                        triangles.append([u, v, w])
        return torch.tensor(triangles, dtype=torch.long, device=edges.device)

    def _precompute_batch_indices(self):
        # Existing logic
        if self.triangles.shape[0] == 0:
            self.register_buffer('tri_edge_idx', torch.zeros(0, 3, dtype=torch.long))
            self.register_buffer('tri_edge_sign', torch.zeros(0, 3, dtype=torch.float))
            return

        t_list = self.triangles.cpu().numpy()
        indices = []
        signs = []
        
        for u, v, w in t_list:
             i1, s1 = self.edge_map.get((u, v))
             i2, s2 = self.edge_map.get((v, w))
             i3, s3 = self.edge_map.get((w, u))
             indices.append([i1, i2, i3])
             signs.append([s1, s2, s3])
             
        self.register_buffer('tri_edge_idx', torch.tensor(indices, dtype=torch.long, device=self.edges.device))
        self.register_buffer('tri_edge_sign', torch.tensor(signs, dtype=torch.float, device=self.edges.device))

    @staticmethod
    def log_so3(R: torch.Tensor) -> torch.Tensor:
        # Existing logic
        trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta)
        factor = theta / (2.0 * sin_theta)
        mask_small = theta < 1e-4
        factor[mask_small] = 0.5 + (theta[mask_small]**2) / 12.0
        factor = factor.unsqueeze(-1).unsqueeze(-1)
        Omega = factor * (R - R.transpose(-2, -1))
        return Omega

    # --- Physics Interfaces (Updated) ---

    def get_U(self, x: Optional[torch.Tensor] = None, edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute U for specific edges.
        If edges is None, return U for ALL self.edges (Topology).
        TableBackend uses indices. NeuralBackend uses x.
        """
        if edges is None:
            edges = self.edges
            if self.backend_type == 'table':
                # TableBackend can use None to return all params directly (optimized)
                 omega = self.backend(edge_indices=None)
            else:
                 # NeuralBackend needs edges_uv and x
                 omega = self.backend(edges_uv=edges, x=x)
        else:
             # Specific query
             if self.backend_type == 'table':
                 omega = self.backend(edges_uv=edges) # Will return zeros for Table if not implemented logic
             else:
                 omega = self.backend(edges_uv=edges, x=x)
                 
        U = torch.matrix_exp(omega)
        return U

    def compute_curvature(self, x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Curvature Omega for all triangles.
        Requires x for NeuralBackend.
        """
        if self.triangles.shape[0] == 0:
             return torch.zeros(0, self.logical_dim, self.logical_dim, device=self.edges.device), self.triangles, (None, None, None)

        idx = self.tri_edge_idx # (T, 3)
        sign = self.tri_edge_sign # (T, 3)
        
        # We need U for the triangle edges.
        # TableBackend: U_all = self.get_U(None) -> indices.
        # NeuralBackend: Need x. U_all = self.get_U(x, self.edges).
        
        # Optimization: Don't compute U for ALL edges if we only need triangles?
        # But we act on all edges in typical training step.
        # Let's compute global U for self.edges.
        
        U_all = self.get_U(x, self.edges) # (E, k, k)
        
        U_edges = U_all[idx] 
        
        U1 = U_edges[:, 0].clone()
        U2 = U_edges[:, 1].clone()
        U3 = U_edges[:, 2].clone()
        
        mask_inv = (sign < 0)
        if torch.any(mask_inv[:, 0]): U1[mask_inv[:, 0]] = U1[mask_inv[:, 0]].transpose(-2, -1)
        if torch.any(mask_inv[:, 1]): U2[mask_inv[:, 1]] = U2[mask_inv[:, 1]].transpose(-2, -1)
        if torch.any(mask_inv[:, 2]): U3[mask_inv[:, 2]] = U3[mask_inv[:, 2]].transpose(-2, -1)
        
        H = torch.matmul(U3, torch.matmul(U2, U1))
        
        if self.logical_dim == 3:
            Omega = self.log_so3(H)
        else:
            Omega = 0.5 * (H - H.transpose(-2, -1))
            
        return Omega, self.triangles, (U1, U2, U3)
        
    def compute_connection(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Wong Precession A(v).
        Needs omega for all edges.
        """
        # Call backend to get omega for all edges
        if self.backend_type == 'table':
             omega_params = self.backend(edge_indices=None)
        else:
             omega_params = self.backend(edges_uv=self.edges, x=x)
             
        # ... Reuse logic ...
        start = self.edges[:, 0]
        end = self.edges[:, 1]
        x_start = x[start]
        x_end = x[end]
        dir_start = log_map(x_start, x_end)
        dist_sq = torch.sum(dir_start ** 2, dim=-1, keepdim=True).clamp(min=1e-6)
        v_start = v[start]
        proj = torch.sum(v_start * dir_start, dim=-1, keepdim=True) / dist_sq
        
        omega = 0.5 * (omega_params - omega_params.transpose(1, 2))
        weighted = omega * proj.unsqueeze(-1)
        
        N = x.shape[0]
        k = self.logical_dim
        A = torch.zeros(N, k, k, device=x.device)
        A.index_add_(0, start, weighted)
        
        # Incoming
        dir_end = log_map(x_end, x_start)
        dist_sq_end = torch.sum(dir_end ** 2, dim=-1, keepdim=True).clamp(min=1e-6)
        v_end = v[end]
        proj_end = torch.sum(v_end * dir_end, dim=-1, keepdim=True) / dist_sq_end
        weighted_end = (-omega) * proj_end.unsqueeze(-1)
        A.index_add_(0, end, weighted_end)
        
        return A

    def compute_spin_interaction(self, x: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """
        Spin Interaction.
        Updated to accept x.
        """
        start = self.edges[:, 0]
        end = self.edges[:, 1]
        
        U_all = self.get_U(x, self.edges)
        
        N = J.shape[0]
        k = self.logical_dim
        B = torch.zeros(N, k, device=J.device)
        
        # Neighbors v (via u->v)
        J_v = J[end]
        U_transpose = U_all.transpose(1, 2)
        J_v_transported = torch.matmul(U_transpose, J_v.unsqueeze(-1)).squeeze(-1)
        B.index_add_(0, start, J_v_transported)
        
        # Neighbors w (via w->u)
        J_w = J[start]
        J_w_transported = torch.matmul(U_all, J_w.unsqueeze(-1)).squeeze(-1)
        B.index_add_(0, end, J_w_transported)
        
        J_exp = J.unsqueeze(-1)
        B_exp = B.unsqueeze(1)
        JB = torch.matmul(J_exp, B_exp)
        A_spin = JB - JB.transpose(1, 2)
        return A_spin

    def compute_force_wong(self, x: torch.Tensor, v: torch.Tensor, J: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Existing logic, but pass x to compute_curvature
        if self.triangles.shape[0] == 0:
            return torch.zeros(x.shape[0], device=x.device), torch.zeros_like(x)
            
        Omega, _, (U_ij, U_jk, U_ki) = self.compute_curvature(x)
        
        # ... Rest of Wong Force Logic ...
        # Need to copy the logic from previous file or ensure it's preserved.
        # Ideally I reuse the previous implementation block.
        # But for 'replace_file' I am writing the whole file.
        # I must ensure the logic is identical.
        
        idx_i = self.triangles[:, 0]
        idx_j = self.triangles[:, 1]
        idx_k = self.triangles[:, 2]
        xi = x[idx_i]
        xj = x[idx_j]
        xk = x[idx_k]
        u_vec = log_map(xi, xj)
        w_vec = log_map(xi, xk)
        vi = v[idx_i]
        vw = minkowski_inner(vi, w_vec).unsqueeze(-1)
        vu = minkowski_inner(vi, u_vec).unsqueeze(-1)
        L_vec = vw * u_vec - vu * w_vec
        
        if self.logical_dim == 3:
            o0 = Omega[:, 2, 1]
            o1 = Omega[:, 0, 2]
            o2 = Omega[:, 1, 0]
            Omega_vec = torch.stack([o0, o1, o2], dim=1)
            
            Ji = J[idx_i]
            q_i = torch.sum(Ji * Omega_vec, dim=-1).unsqueeze(-1)
            F_total = torch.zeros_like(x)
            F_total.index_add_(0, idx_i, q_i * L_vec)
            
            # Symmetric terms
            U_ji = U_ij.transpose(-2, -1)
            Omega_j_mat = torch.matmul(U_ji, torch.matmul(Omega, U_ij))
            Omega_k_mat = torch.matmul(U_ki, torch.matmul(Omega, U_ki.transpose(-2, -1)))
            
            def mat_to_vec(M): return torch.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], dim=1)
            Omega_j_vec = mat_to_vec(Omega_j_mat)
            Omega_k_vec = mat_to_vec(Omega_k_mat)
            
            q_j = torch.sum(J[idx_j] * Omega_j_vec, dim=-1).unsqueeze(-1)
            q_k = torch.sum(J[idx_k] * Omega_k_vec, dim=-1).unsqueeze(-1)
            
            t_jk = log_map(xj, xk); t_ji = log_map(xj, xi)
            vj = v[idx_j]
            vw_j = minkowski_inner(vj, t_ji).unsqueeze(-1)
            vu_j = minkowski_inner(vj, t_jk).unsqueeze(-1)
            L_j = vw_j * t_jk - vu_j * t_ji 
            
            t_ki = log_map(xk, xi); t_kj = log_map(xk, xj)
            vk = v[idx_k]
            vw_k = minkowski_inner(vk, t_kj).unsqueeze(-1)
            vu_k = minkowski_inner(vk, t_ki).unsqueeze(-1)
            L_k = vw_k * t_ki - vu_k * t_kj
            
            F_total.index_add_(0, idx_j, q_j * L_j)
            F_total.index_add_(0, idx_k, q_k * L_k)
            
            return torch.zeros(x.shape[0], device=x.device), F_total
        else:
             return torch.zeros(x.shape[0], device=x.device), torch.zeros_like(x)

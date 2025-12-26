"""
Aurora Gauge Field (CCD v2.0).
=============================

Implements the Logical Fiber Bundle connections and curvature.
Ref: `docs/plan/理论基础-5.md` Chapter 2.

Components:
1. GaugeField: Manages parallel transport matrix U for edges.
2. Wilson Loops: Computes curvature from closed loops.
3. Wong Force: Computes F_logic.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
from .geometry import project_to_tangent, log_map, minkowski_inner
import torch.nn.functional as F

class GaugeField(nn.Module):
    """
    Manages the Connection A_mu on the principal bundle.
    In discrete implementations, this is stored as edge transport matrices U_ij in G.
    
    For SO(k) or SU(k), U_ij = exp(omega_ij).
    """
    def __init__(self, edges: torch.Tensor, logical_dim: int, group='SO', skew_symmetric: bool = True):
        """
        Args:
            edges: (E, 2) edge list
            logical_dim: Dimension of the internal vector space (e.g. k for SO(k))
            group: 'SO' (Orthogonal) or 'SU' (Unitary - complex not yet supported explicitly)
        """
        super().__init__()
        self.register_buffer('edges', edges)
        self.logical_dim = logical_dim
        self.E = edges.shape[0]
        
        # Store Lie Algebra elements omega_ij (E, k, k)
        # For SO(k), omega is skew-symmetric.
        # We store flat parameters and project to algebra during forward.
        if group == 'SO':
            # Number of generators for SO(k) is k(k-1)/2
            # But simpler to just store k*k and force skew-symmetry.
            self.omega_params = nn.Parameter(torch.randn(self.E, logical_dim, logical_dim) * 0.01)
        else:
            raise NotImplementedError("Only SO group supported currently.")
            
        # Build Index Map (u,v) -> (idx, sign)
        self.edge_map = self._build_index_map(edges)
            
        # Precompute triangles (plaquettes)
        self.triangles = self._build_triangles(edges)
        print(f"GaugeField: Found {self.triangles.shape[0]} triangles.")
        self._precompute_batch_indices()

    def _build_index_map(self, edges: torch.Tensor):
        edge_map = {}
        edge_list = edges.cpu().numpy()
        for idx, (u, v) in enumerate(edge_list):
            edge_map[(u, v)] = (idx, 1) # Canonical
            edge_map[(v, u)] = (idx, -1) # Inverse
        return edge_map
        
    def _precompute_batch_indices(self):
        """
        Precompute indices for batch gather in compute_curvature/force.
        Populates self.tri_edge_idx (T, 3) and self.tri_edge_sign (T, 3).
        """
        if self.triangles.shape[0] == 0:
            self.register_buffer('tri_edge_idx', torch.zeros(0, 3, dtype=torch.long))
            self.register_buffer('tri_edge_sign', torch.zeros(0, 3, dtype=torch.float))
            return

        t_list = self.triangles.cpu().numpy()
        indices = []
        signs = []
        
        for u, v, w in t_list:
             # u->v
             i1, s1 = self.edge_map.get((u, v))
             # v->w
             i2, s2 = self.edge_map.get((v, w))
             # w->u
             i3, s3 = self.edge_map.get((w, u))
             
             if i1 is None or i2 is None or i3 is None:
                 print(f"Warning: Triangle ({u},{v},{w}) has missing edges! Skipping.")
                 # This shouldn't happen if built from edges
                 i1, i2, i3 = 0, 0, 0
                 s1, s2, s3 = 1, 1, 1
             
             indices.append([i1, i2, i3])
             signs.append([s1, s2, s3])
             
        self.register_buffer('tri_edge_idx', torch.tensor(indices, dtype=torch.long, device=self.edges.device))
        self.register_buffer('tri_edge_sign', torch.tensor(signs, dtype=torch.float, device=self.edges.device))
        
    @staticmethod
    def log_so3(R: torch.Tensor) -> torch.Tensor:
        """
        Compute Logarithm of SO(3) matrix: R -> Omega (skew-symmetric).
        Omega = theta / (2*sin(theta)) * (R - R^T)
        """
        # Trace: R_ii
        trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
        # Cos theta
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)
        
        # Sinc factor: theta / (2*sin(theta))
        # Limit theta->0 is 0.5
        sin_theta = torch.sin(theta)
        factor = theta / (2.0 * sin_theta)
        
        # Identify small angles
        mask_small = theta < 1e-4
        factor[mask_small] = 0.5 + (theta[mask_small]**2) / 12.0
        
        factor = factor.unsqueeze(-1).unsqueeze(-1)
        
        Omega = factor * (R - R.transpose(-2, -1))
        return Omega
        
    def _build_node_to_triangles(self, triangles: torch.Tensor, N: int):
        """
        Build map: node_idx -> list of triangle_indices where node is the 'base' (first index).
        For $F_i$, we need triangles starting at $i$.
        Since we need full sum, we might need all permutations?
        Actually, we can just scatter add the force to the vertices.
        Force is computed per triangle, then distributed to its vertices.
        So we don't strictly need a map if we compute per triangle and scatter.
        """
        pass # Not using map, using scatter logic in forward
            
        # Precompute triangles (plaquettes)
        self.triangles = self._build_triangles(edges)
        print(f"GaugeField: Found {self.triangles.shape[0]} triangles.")
        self._precompute_batch_indices()


    def _build_triangles(self, edges: torch.Tensor) -> torch.Tensor:
        """
        Find all triangles (i, j, k) such that (i,j), (j,k), (k,i) are edges.
        Returns tensor of shape (T, 3) containing node indices of triangles.
        """
        # Convert edges to adjacency set for O(1) lookup
        adj = {}
        edge_list = edges.cpu().numpy()
        for u, v in edge_list:
            if u not in adj: adj[u] = set()
            if v not in adj: adj[v] = set()
            adj[u].add(v)
            adj[v].add(u) # undirected for structural check
            
        triangles = []
        # Naive enumeration O(N^3) or O(E*d_max). For 164k it's slow, but for 5k ok.
        # Since we limit dataset to 5000, this is acceptable.
        visited = set()
        
        nodes = sorted(list(adj.keys()))
        for u in nodes:
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                v = neighbors[i]
                if v <= u: continue # Check order to avoid duplicates (u < v < w)
                
                for j in range(i+1, len(neighbors)):
                    w = neighbors[j]
                    if w <= v: continue 
                    
                    if w in adj[v]:
                        # Found uv, uw. Check vw.
                        triangles.append([u, v, w])
                        
        return torch.tensor(triangles, dtype=torch.long, device=edges.device)

    def _get_edge_index(self, u, v):
        return self.edge_map.get((u, v), (None, None))

    def compute_curvature(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Curvature Omega for all triangles.
        Returns:
            Omega: (T, k, k) Lie Algebra elements
            tri_indices: (T, 3) triangle nodes
        """
        if self.triangles.shape[0] == 0:
            return torch.zeros(0, self.logical_dim, self.logical_dim, device=self.edges.device), self.triangles, (None, None, None)
            
        # Use cached indices
        if not hasattr(self, 'tri_edge_idx'):
            self._precompute_batch_indices()
            
        idx = self.tri_edge_idx # (T, 3)
        sign = self.tri_edge_sign # (T, 3)
        
        U_all = self.get_U() # (E, k, k)
        
        # Gather U
        # U_edges has shape (T, 3, k, k)
        U_edges = U_all[idx] 
        
        # Clone to separate for manipulation
        U1 = U_edges[:, 0].clone()
        U2 = U_edges[:, 1].clone()
        U3 = U_edges[:, 2].clone()
        
        # Apply signs (Transpose if sign is -1)
        # We use explicit transpose for masked elements
        
        # Mask where sign < 0
        mask_inv = (sign < 0) # (T, 3)
        
        # In-place transpose for inverted edges
        if torch.any(mask_inv[:, 0]):
            U1[mask_inv[:, 0]] = U1[mask_inv[:, 0]].transpose(-2, -1)
        if torch.any(mask_inv[:, 1]):
            U2[mask_inv[:, 1]] = U2[mask_inv[:, 1]].transpose(-2, -1)
        if torch.any(mask_inv[:, 2]):
            U3[mask_inv[:, 2]] = U3[mask_inv[:, 2]].transpose(-2, -1)
        
        # Holonomy H = U3 @ U2 @ U1 (Chain rule: u->v->w->u)
        # Note: Order depends on definition.
        # U1: u->v. U2: v->w. U3: w->u.
        # J_u_new = U3 (U2 (U1 J_u)).
        H = torch.matmul(U3, torch.matmul(U2, U1))
        
        # Exact Logarithm for Curvature
        if self.logical_dim == 3:
            Omega = self.log_so3(H)
        else:
             # Fallback
            Omega = 0.5 * (H - H.transpose(-2, -1))
            
        return Omega, self.triangles, (U1, U2, U3)
        
    def compute_connection(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute effective connection A(v) at each node for Wong Precession.
        A(v) = sum_{j in N(i)} omega_{ij} * <v_i, e_{ij}>
        
        Args:
            x: (N, dim) positions
            v: (N, dim) velocities in tangent space T_x Q
            
        Returns:
            A: (N, k, k) skew-symmetric matrices
        """
        # Edges
        start = self.edges[:, 0]
        end = self.edges[:, 1]
        
        # 1. Compute Tangent Directions for edges
        # u_ij = Log_xi(xj)
        x_start = x[start]
        x_end = x[end]
        
        # Log map x_start -> x_end
        # Note: log_map(x, y) returns vector at x pointing to y
        dir_start = log_map(x_start, x_end) # (E, dim) in T_{x_start}
        
        # Normalize directions? A_mu is usually per unit length? 
        # Or A_mu dx^mu. If omega is integral over edge, then projection should be normalized?
        # If omega is "Connection 1-form value integrated", then it is dimensionless?
        # Typically U_ij = P exp(int A). So omega ~ A * L.
        # So A ~ omega / L.
        # Term is <A, v> = (omega/L) * <v, hat{dir}> = omega * <v, hat{dir}> / L?
        # No, <v, hat{dir}> dt = dx. So (omega/L) dx = omega * (dx/L).
        # So we project v onto direction, fraction of length covered per sec?
        # Let's use normalized projection coefficient: coeff = <v, dir> / |dir|^2.
        # This implies if v moves full edge length in 1 sec, it picks up full omega.
        
        dist_sq = torch.sum(dir_start ** 2, dim=-1, keepdim=True).clamp(min=1e-6)
        
        # 2. Project v onto edges
        v_start = v[start]
        proj = torch.sum(v_start * dir_start, dim=-1, keepdim=True) / dist_sq # (E, 1)
        
        # 3. Weighted Omega
        # A_eff = sum proj * omega
        # omega_params: (E, k, k)
        # We need skew-symmetric form
        omega = 0.5 * (self.omega_params - self.omega_params.transpose(1, 2))
        
        weighted = omega * proj.unsqueeze(-1) # (E, k, k)
        
        # 4. Scatter Add to nodes
        # We need to handle both directions? 
        # self.edges only has one direction (u, v).
        # We need contributions from neighbor edges.
        # For node u:
        #   Edges (u, v): contributes proj(v_u, uv) * omega_{uv}
        #   Edges (w, u): contributes proj(v_u, uw) * omega_{uw}
        # Notice omega_{uw} = -omega_{wu}.
        # So we process (u,v) edges for u, and (w,u) edges for u separately.
        
        N = x.shape[0]
        k = self.logical_dim
        A = torch.zeros(N, k, k, device=x.device)
        
        # Add outgoing edges (u as start)
        # index_add_ expects dim argument.
        # A is (N, k, k). We add to dim 0 according to `start`.
        # weighted is (E, k, k).
        # Reshape for scatter?
        # A.index_add_(0, start, weighted) works if shape matches.
        A.index_add_(0, start, weighted)
        
        # Add incoming edges (v as end)
        # For edge (w, u), we are at u (end node).
        # We need Log_u(w) and omega_{uw}.
        # Log_u(w) approx - ParallelTransport(Log_w(u)) ?
        # Or explicit calc:
        dir_end = log_map(x_end, x_start) # Tangent at x_end pointing to x_start
        dist_sq_end = torch.sum(dir_end ** 2, dim=-1, keepdim=True).clamp(min=1e-6)
        v_end = v[end]
        proj_end = torch.sum(v_end * dir_end, dim=-1, keepdim=True) / dist_sq_end
        
        # Omega_{vu} = -Omega_{uv}
        weighted_end = (-omega) * proj_end.unsqueeze(-1)
        
        A.index_add_(0, end, weighted_end)
        
        return A

    def compute_force_wong(self, x: torch.Tensor, v: torch.Tensor, J: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Wong Lorentz Force based on Curvature Flux.
        F_i = sum_{tri in ijk} <J_i, Omega_ijk> ( <v_i, w> u - <v_i, u> w )
        where u = Log_i(j), w = Log_i(k).
        
        Args:
            x: (N, dim) positions
            v: (N, dim) velocities
            J: (N, k) logical charge
            
        Returns:
            Energy: (N,) scalar (Interaction Energy - fictitious?)
            Force: (N, dim) vector
        """
        if self.triangles.shape[0] == 0:
            return torch.zeros(x.shape[0], device=x.device), torch.zeros_like(x)
            
        # 1. Compute Curvature and Transports
        Omega, _, (U_ij, U_jk, U_ki) = self.compute_curvature() # (T, k, k)

        
        # 2. Geometry vectors
        # Triangle (i, j, k)
        idx_i = self.triangles[:, 0]
        idx_j = self.triangles[:, 1]
        idx_k = self.triangles[:, 2]
        
        xi = x[idx_i]
        xj = x[idx_j]
        xk = x[idx_k]
        
        # Tangent vectors at i
        u_vec = log_map(xi, xj) # i -> j
        w_vec = log_map(xi, xk) # i -> k
        
        # Velocity at i
        vi = v[idx_i]
        
        # 3. Lorentz vector term: (v x (u^w)) = <v,w>u - <v,u>w
        # <v, w>
        vw = minkowski_inner(vi, w_vec).unsqueeze(-1)
        # <v, u>
        vu = minkowski_inner(vi, u_vec).unsqueeze(-1)
        
        # Force direction
        L_vec = vw * u_vec - vu * w_vec
        
        # 4. Coupling Charge
        # q = <J_i, Omega_ijk>
        # J assumed to be vector representation mapped to algebra.
        # For SO(3), J vector and Omega matrix.
        # <J, Omega> ?
        # Standard Wong: Tr(J_matrix * Omega). J is moment map.
        # In our convention J is a vector.
        # We need to map J vector to Algebra matrix J_mat?
        # Or map Omega matrix to vector Omega_vec?
        # Let's map Omega to vector.
        # For SO(3): Omega = [[0, c, -b], [-c, 0, a], [b, -a, 0]]. vec = [a, b, c].
        # vec_0 = Omega[2,1] - Omega[1,2] (with factor 0.5?)
        # Let's use simple dot product of parameters if possible.
        # Omega is (T, k, k).
        # J_i is (T, k).
        
        # Extract dual vector from Omega
        # v[0] = Omega[2,1], v[1] = Omega[0,2], v[2] = Omega[1,0]
        # (Cyclic indices for SO(3))
        # This mapping depends on dimension.
        # Generic trace: J_mat = J_a T^a. Omega = Omega_b T^b.
        # <J, Omega> = J_a Omega_b Tr(T^a T^b) = -delta_ab J_a Omega_b (Killing form).
        # So it acts like dot product.
        
        # Let's implement generic projection for SO(3) specifically first, or generalized?
        # Generalized: Flatten both?
        # No, J is (k). Omega is (k, k).
        # If Omega is skew symmetric, independent components are k(k-1)/2.
        # If logical_dim == k (for SO(3), 3 dims), then J matches generators.
        # We need to extract the coefficients of Omega relative to generators.
        
        if self.logical_dim == 3:
            # Vector from skew matrix
            # Omega_vec: (T, 3)
            # 0: (2,1), 1: (0,2), 2: (1,0)
            o0 = Omega[:, 2, 1]
            o1 = Omega[:, 0, 2]
            o2 = Omega[:, 1, 0]
            Omega_vec = torch.stack([o0, o1, o2], dim=1)
            
            # Coupling
            # Coupling Charge and Force for Node I
            Ji = J[idx_i]
            # q = J dot Omega_vec
            q_i = torch.sum(Ji * Omega_vec, dim=-1).unsqueeze(-1) # (T, 1)
            F_total = torch.zeros_like(x)
            F_total.index_add_(0, idx_i, q_i * L_vec)
            
            # --- Symmetric Forces on J and K ---
            # To preserve symmetry, force must act on all vertices of the flux loop.
            # Transport Omega to J and K to find local force.
            
            # Node J: Omega_j = U_ij^T @ Omega_i @ U_ij
            # U_ij is available as U_ij (from curvature return)
            # Actually compute_curvature returns U1, U2, U3 oriented along cycle u->v->w->u
            # So U_ij transports i -> j? Or j -> i?
            # Convention: U1 corresponds to edge (u, v).
            # In gauge field, U_uv transports v -> u (vector at v to vector at u).
            # So v_u = U_uv v_v.
            # Thus U1 transforms from J to I.
            # So Omega_i (at I) transforms to Omega_j (at J) via Adjoint of U_ij^-1 = U_ij^T?
            # v_j = U_ij^T v_i.
            # A_j = U_ij^T A_i U_ij.
            # Yes.
            
            # Omega_j
            U_ji = U_ij.transpose(-2, -1)
            Omega_j_mat = torch.matmul(U_ji, torch.matmul(Omega, U_ij))
            
            # Omega_k
            # Omega_k = U_ki @ Omega_i @ U_ki^T ?
            # U3 corresponds to edge (w, u) i.e. (k, i).
            # U_ki transports i -> k?
            # U3 transports from u (i) to w (k)? No, edge is (w, u).
            # U_wu transports u -> w (i to k).
            # So U3 = U_ki.  v_k = U_ki v_i.
            # So Omega_k = U_ki Omega_i U_ki^T.
            
            Omega_k_mat = torch.matmul(U_ki, torch.matmul(Omega, U_ki.transpose(-2, -1)))

            # Vectors
            def mat_to_vec(M):
                return torch.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], dim=1)

            Omega_j_vec = mat_to_vec(Omega_j_mat)
            Omega_k_vec = mat_to_vec(Omega_k_mat)
            
            # Charges
            q_j = torch.sum(J[idx_j] * Omega_j_vec, dim=-1).unsqueeze(-1)
            q_k = torch.sum(J[idx_k] * Omega_k_vec, dim=-1).unsqueeze(-1)
            
            # Geometric Factors for J and K
            # L_j: Tangent vectors at j. u_j = Log_j(k), w_j = Log_j(i).
            # Note: Approximating geometric term as "Transported L_i" is risky (curvature).
            # Safe way: Recompute L_vec locally at j and k.
            
            # Precompute pairwise Logs for all sides?
            # We computed u=Log_i(j) and w=Log_i(k).
            # For j: u' = Log_j(k), w' = Log_j(i).
            # Log_j(i) = - ParallelTransport(Log_i(j)) approx -u.
            # Let's do explicit computation, it's safer.
            
            # Tangents at J
            t_jk = log_map(xj, xk)
            t_ji = log_map(xj, xi)
            vj = v[idx_j]
            vw_j = minkowski_inner(vj, t_ji).unsqueeze(-1) # v_j . ji
            vu_j = minkowski_inner(vj, t_jk).unsqueeze(-1) # v_j . jk
            L_j = vw_j * t_jk - vu_j * t_ji # (v x (jk ^ ji)) ?? Check cyclic order
            
            # Tangents at K
            t_ki = log_map(xk, xi)
            t_kj = log_map(xk, xj)
            vk = v[idx_k]
            vw_k = minkowski_inner(vk, t_kj).unsqueeze(-1)
            vu_k = minkowski_inner(vk, t_ki).unsqueeze(-1)
            L_k = vw_k * t_ki - vu_k * t_kj

            # Accumulate
            F_total.index_add_(0, idx_j, q_j * L_j)
            F_total.index_add_(0, idx_k, q_k * L_k)
            
            return torch.zeros(x.shape[0], device=x.device), F_total

        else:
            return torch.zeros(x.shape[0], device=x.device), torch.zeros_like(x)
            
    def get_U(self) -> torch.Tensor:
        """
        Compute Group Elements U_ij = exp(omega_ij).
        Enforces constraints (Skew-symmetry for SO).
        """
        # Enforce skew-symmetry: A = (W - W^T)/2
        omega = 0.5 * (self.omega_params - self.omega_params.transpose(1, 2))
        
        # Matrix Exponential map
        # torch.matrix_exp is consistent with Lie Group exp for matrix groups
        U = torch.matrix_exp(omega)
        return U
    
    def parallel_transport(self, J: torch.Tensor, edge_indices: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Transport logic charge J from u to v via edge (u,v).
        J_v = U_ij @ J_u
        """
        # J: (Batch, k)
        # U: (E, k, k)
        
        U_batch = self.get_U()[edge_indices] # (Batch, k, k)
        
        if inverse:
            # U_inv = U^T for SO(k)
            U_op = U_batch.transpose(1, 2)
        else:
            U_op = U_batch
            
        # J is vector (Batch, k)
        # Treat as column vector: J_col = (Batch, k, 1)
        # result = U @ J_col -> (Batch, k, 1)
        J_new = torch.matmul(U_op, J.unsqueeze(-1)).squeeze(-1)
        return J_new

    def compute_spin_interaction(self, J: torch.Tensor) -> torch.Tensor:
        """
        Compute Spin Interaction Generator (Heisenberg/XY alignment).
        Generates rotation to align J_i with neighbors.
        
        B_i = sum_{j} U_{ij} J_j
        A_spin = J_i ^ B_i = J_i B_i^T - B_i J_i^T
        
        Args:
            J: (N, k)
        Returns:
            A_spin: (N, k, k) skew-symmetric
        """
        # 1. Gather Neighbors and Transport
        start = self.edges[:, 0]
        end = self.edges[:, 1]
        
        # Outgoing edges: u->v. Neighbor is v at u.
        # J_neighbor = U_{uv} J_v ???
        # Strictly: U_{uv} transports vector from u to v.
        # So J_v is at v. To bring to u, we need U_{vu} = U_{uv}^{-1}.
        # Wait. U_{uv} maps T_u -> T_v.
        # We need map T_v -> T_u.
        # So we need U_{uv}^T (for SO).
        
        # J at u: 
        # Contributions from neighbors v (via edge u,v): U_{uv}^T J_v
        # Contributions from neighbors w (via edge w,u): U_{wu} J_w (straight, since U_{wu} maps w->u)
        
        N = J.shape[0]
        k = self.logical_dim
        B = torch.zeros(N, k, device=J.device)
        
        U_all = self.get_U() # (E, k, k)
        
        # 1a. Neighbors v (via u->v edges)
        # We are at u. Neighbor v.
        # J_v_at_u = U_{uv}^T J_v
        J_v = J[end] # (E, k)
        U_transpose = U_all.transpose(1, 2)
        J_v_transported = torch.matmul(U_transpose, J_v.unsqueeze(-1)).squeeze(-1)
        B.index_add_(0, start, J_v_transported)
        
        # 1b. Neighbors w (via w->u edges)
        # We are at u. Neighbor w.
        # J_w_at_u = U_{wu} J_w
        J_w = J[start]
        J_w_transported = torch.matmul(U_all, J_w.unsqueeze(-1)).squeeze(-1)
        B.index_add_(0, end, J_w_transported)
        
        # 2. Compute Generator A = J B^T - B J^T
        # J: (N, k). B: (N, k)
        J_exp = J.unsqueeze(-1) # (N, k, 1)
        B_exp = B.unsqueeze(1)  # (N, 1, k)
        
        JB = torch.matmul(J_exp, B_exp) # (N, k, k) -> J * B^T
        
        A_spin = JB - JB.transpose(1, 2)
        
        return A_spin

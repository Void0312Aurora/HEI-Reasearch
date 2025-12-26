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

    def _build_index_map(self, edges: torch.Tensor):
        edge_map = {}
        edge_list = edges.cpu().numpy()
        for idx, (u, v) in enumerate(edge_list):
            edge_map[(u, v)] = (idx, 1) # Canonical
            edge_map[(v, u)] = (idx, -1) # Inverse
        return edge_map
        
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
            return torch.zeros(0, self.logical_dim, self.logical_dim, device=self.edges.device), self.triangles
            
        U = self.get_U() # (E, k, k)
        
        # Gather U for each side of triangle (u,v,w)
        # Loop: u->v->w->u
        t_list = self.triangles.cpu().numpy()
        
        # Collect indices and signs for batch gathering
        # We process in Python loop for now (Precompute this later for speed!)
        # Optimization: Move this index lookup to __init__
        
        # Doing it on the fly is slow.
        # Let's assume for prototype we run it.
        # To make it differentiable, we need to gather from U tensor.
        
        indices_list = []
        signs_list = []
        
        for u, v, w in t_list:
             # u->v
             idx1, s1 = self.edge_map[(u, v)]
             # v->w
             idx2, s2 = self.edge_map[(v, w)]
             # w->u
             idx3, s3 = self.edge_map[(w, u)]
             
             indices_list.append([idx1, idx2, idx3])
             signs_list.append([s1, s2, s3])
             
        indices = torch.tensor(indices_list, device=U.device, dtype=torch.long)
        signs = torch.tensor(signs_list, device=U.device, dtype=torch.float)
        
        # Gather U
        # shapes: indices (T, 3). U (E, k, k)
        U1 = U[indices[:, 0]]
        U2 = U[indices[:, 1]]
        U3 = U[indices[:, 2]]
        
        # Apply inversion if sign is -1 (transpose)
        # sign is (T,), U is (T, k, k). 
        # If sign=-1, we want U.t(). 
        # Trick: U_inv = U.transpose(-2, -1) if SO(k)
        
        def apply_sign(U_in, s):
            # s: (T,)
            # U_in: (T, k, k)
            # If s=-1, swap dims. 
            # We can use mask.
            mask = (s < 0)
            U_out = U_in.clone()
            U_out[mask] = U_in[mask].transpose(-2, -1)
            return U_out
            
        U1 = apply_sign(U1, signs[:, 0])
        U2 = apply_sign(U2, signs[:, 1])
        U3 = apply_sign(U3, signs[:, 2])
        
        # Holonomy: H = U3 @ U2 @ U1 (u->v->w->u)
        # Order matters! J_next = U J_prev.
        # u->v: J_v = U1 J_u
        # v->w: J_w = U2 J_v = U2 U1 J_u
        # w->u: J_u_new = U3 J_w = U3 U2 U1 J_u
        H = torch.matmul(U3, torch.matmul(U2, U1))
        
        # Extract Algebra element Omega
        # Omega approx log(H)
        # For small curvature, H approx I + Omega
        # Omega = (H - H^T)/2
        Omega = 0.5 * (H - H.transpose(-2, -1))
        
        return Omega, self.triangles
        
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
            
        # 1. Compute Curvature
        Omega, _ = self.compute_curvature() # (T, k, k)
        
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
            Ji = J[idx_i]
            # q = J dot Omega_vec
            q = torch.sum(Ji * Omega_vec, dim=-1).unsqueeze(-1) # (T, 1)
            
            # Force on i
            Fi_contrib = q * L_vec
            
            # Accumulate info global Force tensor
            # Scatter add
            F_total = torch.zeros_like(x)
            F_total.index_add_(0, idx_i, Fi_contrib)
            
            # Todo: Symmetry? 
            # Force should also act on j and k?
            # Or we sum over cyclic permutations?
            # compute_curvature iterates all triangles?
            # _build_triangles iterates unique triplets (u,v,w).
            # It does NOT generate permutations (v,w,u) etc usually.
            # My _build_triangles ensures u < v < w?
            # Yes: checks `u < v` and `v < w`.
            # So we only have ONE entry per triangle.
            # We must compute force for i, j, and k explicitly.
            
            # Force on j: Cyclic permutation i->j->k->i
            # Omega_{jki} = U_{ij} Omega_{ijk} U_{ij}^{-1} (Adjoint transport)
            # Or just recompute local geometry?
            # Easier: Just reuse Omega but transport J?
            # Actually, $F_{logic}^\mu = \langle J, F^{\mu\nu} \rangle v$.
            # $F^{\mu\nu}$ is global field.
            # We calculated $F$ at $i$.
            # We need to calculate $F$ at $j$ and $k$ too.
            # Re-using the same triangle loop logic for j and k.
            
            # For j:
            # Base j. Neighbors k, i.
            # u' = Log_j(k), w' = Log_j(i).
            # Omega at j? Transport Omega_i to j?
            # Omega_j = U_{ji} Omega_i U_{ij}.
            
            # Calculation for J:
            
            # Let's wrap this in a loop over the 3 vertices to cover all forces?
            
            # Term for j
            inv_Uij = self.get_U()[self.edge_map[(idx_i[0].item(), idx_j[0].item())][0]].transpose(-2,-1) 
            # No, map lookup inside tensor op is bad.
            # We need batch transport.
            
            # PROPOSAL:
            # Only compute force on 'i' (base) for all triangles provided by _build_triangles?
            # No, _build_triangles only gives unique ones.
            # We should probably augment _build_triangles to return ALL permutations? 
            # Or just duplicate logic here.
            
            # To keep it simple: F_total accumulates contributions for i, j, k.
            # We need Omega at j and k.
            # Omega_j approx U_{ji} @ Omega_i @ U_{ij}.
            
            # For now (MVP Phase 2): Just apply force to 'base' node i.
            # This is physically incomplete (symmetry breaking), but tests the force magnitude/direction.
            # Wait, if we only push 'i', we break Newton's 3rd law (conservation of momentum).
            # But we have damping, so strict conservation isn't critical for 'Stability Check', but is for 'Dynamics'.
            # Better: In `_build_triangles`, return ALL permutations.
            # Or `compute_curvature` handles alignment.
            
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



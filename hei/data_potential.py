
import numpy as np
import dataclasses
from numpy.typing import ArrayLike, NDArray
from .geometry import uhp_distance_and_grad, cayley_disk_to_uhp

@dataclasses.dataclass
class DataDrivenStressPotential:
    """
    Data-driven potential for Hyperbolic MDS (Multi-Dimensional Scaling).
    
    Energy function:
        V(z) = 0.5 * sum_{i,j} w_{ij} * (d_hyp(z_i, z_j) - d_target_{ij})^2
    
    This potential drives the particles to assume a geometric configuration
    that preserves the target distances (e.g., from a tree metric).
    """
    target_dists: NDArray[np.float64] # (N, N) symmetric matrix
    weights: NDArray[np.float64] | None = None # (N, N) or None
    stiffness: float = 1.0
    
    def __post_init__(self):
        self.target_dists = np.asarray(self.target_dists)
        n = self.target_dists.shape[0]
        if self.weights is None:
            # Default weights: 1 for everything?
            # Or 1/d^2 usually for MDS?
            # H-MDS usually uses just uniform or 1 if edge exists.
            self.weights = np.ones((n, n), dtype=np.float64)
        else:
            self.weights = np.asarray(self.weights)
            
        # Zero out diagonal to be safe
        np.fill_diagonal(self.target_dists, 0.0)
        np.fill_diagonal(self.weights, 0.0)

    def potential(self, z_uhp: ArrayLike, z_action: float | None = None) -> float:
        """Compute total stress energy."""
        z = np.asarray(z_uhp, dtype=np.complex128).ravel()
        n = z.size
        
        # Vectorized N x N distance calculation
        # z_col: (N, 1), z_row: (1, N)
        z_col = z[:, np.newaxis]
        z_row = z[np.newaxis, :]
        
        # We need distance only. geometry.uhp_distance_and_grad is vectorized?
        # Yes, line 118: z_arr and c_arr are cast.
        # But wait, uhp_distance_and_grad computes gradient too which is expensive.
        # If we only need potential, we should use a lighter function?
        # For now, reuse is fine.
        
        d_mat, _ = uhp_distance_and_grad(z_col, z_row)
        
        diff = d_mat - self.target_dists
        energy_mat = 0.5 * self.weights * (diff ** 2)
        
        # Sum over all i, j. factor 0.5 handles double counting if loop is full?
        # The formula is typically sum_{i<j}. 
        # If we sum full matrix, we get 2 * sum_{i<j}.
        # My 0.5 factor above is applied to each term. So sum full matrix is sum_{i<j} * 2 * 0.5 = sum_{i<j}. Correct.
        return float(np.sum(energy_mat) * self.stiffness)

    def dV_dz(self, z_uhp: ArrayLike, z_action: float | None = None) -> NDArray[np.complex128]:
        """Compute gradient dV/dz_i."""
        z = np.asarray(z_uhp, dtype=np.complex128).ravel()
        
        z_col = z[:, np.newaxis] # z_i
        z_row = z[np.newaxis, :] # z_j (centers c)
        
        # d_mat: (N, N), grad_mat: (N, N) 
        # grad_mat[i, j] is gradient wrt z_i of distance d(z_i, z_j)
        d_mat, grad_mat_wrt_z = uhp_distance_and_grad(z_col, z_row)
        
        # Finite difference term: (d_{ij} - target_{ij})
        diff = d_mat - self.target_dists # (N, N)
        
        # Force accumulation
        # dV/dz_i = sum_j w_{ij} * (d_{ij} - target_{ij}) * d(d_{ij})/dz_i
        # grad_mat_wrt_z[i, j] IS d(d_{ij})/dz_i
        
        coeff_mat = self.weights * diff * self.stiffness # (N, N)
        
        # Element-wise multiply and sum over j (rows)
        grad_total = np.sum(coeff_mat * grad_mat_wrt_z, axis=1) # (N,)
        
        return grad_total.reshape(z_uhp.shape)
        
    def gradient(self, z_uhp: ArrayLike, z_action: float | None = None) -> NDArray[np.complex128]:
        return self.dV_dz(z_uhp, z_action)


import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def compute_dmd(X: np.ndarray, r: int = None) -> Dict[str, Any]:
    """
    Protocol 3: Dynamic Mode Decomposition.
    X: (dim, time_steps) Trajectory data.
    r: Rank truncation (optional).
    
    Returns:
        eigenvalues: complex array (lambda)
        slow_mode_count: int (|lambda| approx 1)
        modes: (dim, r)
    """
    # 1. Prepare Matrices
    # X1 = X[:, :-1] (x_0 ... x_t-1)
    # X2 = X[:, 1:]  (x_1 ... x_t)
    
    if X.shape[1] < 2:
        return {"error": "Trajectory too short"}
        
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    # 2. SVD of X1
    # X1 = U S V*
    try:
        U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"error": "SVD Failed"}
        
    # Rank Truncation
    if r is None:
        # Keep 99% energy or simple heuristic
        cum_energy = np.cumsum(S**2) / np.sum(S**2)
        r = np.searchsorted(cum_energy, 0.99) + 1
        r = min(r, X1.shape[0], X1.shape[1])
        
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    V_r = Vh[:r, :].T # Vh is V*, so V is Vh.T? No, svd returns Vh.
    # V_r = Vh[:r, :].conj().T  (if complex) usually real data -> .T
    
    # 3. Compute A_tilde (Projected Operator)
    # A_tilde = U_r* A U_r = U_r* X2 V_r S_r^{-1}
    # U_r* X2
    part1 = U_r.T @ X2
    # part1 @ V_r
    part2 = part1 @ V_r
    # part2 @ S_inv
    A_tilde = part2 @ np.linalg.inv(S_r)
    
    # 4. Eigendecomposition of A_tilde
    eigvals, eigvecs = np.linalg.eig(A_tilde)
    
    # 5. Reconstruct Modes (Exact DMD)
    # Phi = X2 @ V_r @ S_r^{-1} @ eigvecs
    # part3 = X2 @ V_r @ S_inv
    # modes = part3 @ eigvecs
    # Or Projected Modes = U_r @ eigvecs
    
    # We focus on Eigenvalues for Slow Mode detection
    
    # Slow Metric: |lambda| approx 1.
    # |lambda| < 0.9 -> Fast Decay.
    # |lambda| > 1.1 -> Unstable.
    
    mags = np.abs(eigvals)
    # Slow if mag > 0.95 and mag < 1.05? Or just close to 1.
    # For a stable system, should be <= 1.
    
    slow_indices = np.where((mags > 0.9) & (mags < 1.05))[0]
    slow_count = len(slow_indices)
    
    return {
        "eigenvalues": eigvals,
        "slow_mode_count": slow_count,
        "rank": r,
        "max_mag": np.max(mags) if len(mags) > 0 else 0
    }

def report_dmd_analysis(metrics: Dict[str, Any]) -> bool:
    if "error" in metrics:
        print(f"[Protocol 3] DMD Error: {metrics['error']}")
        return False
        
    count = metrics['slow_mode_count']
    max_mag = metrics['max_mag']
    print(f"[Protocol 3] Slow Modes Detected: {count} (Rank {metrics['rank']})")
    print(f"  Max Eigenvalue Magnitude: {max_mag:.4f}")
    
    # Pass if at least 1 slow mode (Persistence)
    if count >= 1:
         print("  Status: PASS (Persistent Structure Detected)")
         return True
    else:
         print("  Status: FAIL (Fast Decay / No Persistence)")
         return False

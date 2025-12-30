import numpy as np
import scipy.spatial
from scipy.signal import welch

def compute_lyapunov_proxy(traj: np.ndarray, dt: float, embed_dim: int = 2, delay: int = 1):
    """
    Computes a proxy for the Maximal Lyapunov Exponent (MLE) using
    Nearest Neighbor Divergence.
    
    Args:
        traj: [T, D] or [T] (if scalar, will embed)
        dt: Time step
        embed_dim: Embedding dimension (if scalar)
        delay: Embedding delay
        
    Returns:
        lambda_max: Estimated MLE (slope of log divergence)
    """
    # 1. Prepare State Space
    if traj.ndim == 1:
        # Time-delay embedding
        N = len(traj)
        M = N - (embed_dim - 1) * delay
        X = np.zeros((M, embed_dim))
        for i in range(embed_dim):
            X[:, i] = traj[i*delay : i*delay + M]
    else:
        X = traj
        
    N_points = len(X)
    if N_points < 100:
        return 0.0 # Too short
        
    # 2. Find Nearest Neighbors (that are not temporally too close)
    # Using KDTree for speed
    tree = scipy.spatial.cKDTree(X)
    
    # Query: k=2 (self + 1 neighbor)
    dists, indices = tree.query(X, k=2) 
    
    # neighbors = indices[:, 1]
    # initial_dists = dists[:, 1]
    
    # We need to filter neighbors that are temporally close (Theiler window)
    # This is expensive to do perfectly with simple query.
    # Simplified approach: Just track divergence of found pairs for distinct t.
    
    # Let's track average log divergence for k steps ahead.
    max_horizon = min(50, N_points // 10)
    divergence = np.zeros(max_horizon)
    counts = np.zeros(max_horizon)
    
    for i in range(N_points - max_horizon):
         j = indices[i, 1]
         if j > N_points - max_horizon: continue
         if abs(i - j) < 10: continue # Theiler window
         
         # Initial dist
         d0 = np.linalg.norm(X[i] - X[j])
         if d0 < 1e-6: continue
         
         # Track future
         for k in range(max_horizon):
             dist_k = np.linalg.norm(X[i+k] - X[j+k])
             if dist_k > 0:
                 divergence[k] += np.log(dist_k)
                 counts[k] += 1
                 
    # Average Log Divergence
    valid = counts > 0
    avg_log_div = divergence[valid] / counts[valid]
    times = np.arange(len(avg_log_div)) * dt
    
    # Fit Slope (Linear Region)
    # Usually the first part is transient, then linear scaling, then saturation.
    # We fit the middle 50%? or just 0-20?
    if len(times) > 10:
        res = np.polyfit(times[:20], avg_log_div[:20], 1)
        lambda_max = res[0]
    else:
        lambda_max = 0.0
        
    return lambda_max

def check_broadband_spectrum(traj_scalar: np.ndarray, dt: float):
    """
    Check if spectrum is broadband (Chaos) vs Sharp Peaks (Limit Cycle).
    Returns:
        broadband_ratio: (Power in noise floor) / (Total Power)
        or entropy of spectrum.
    """
    f, Pxx = welch(traj_scalar, fs=1.0/dt, nperseg=256)
    
    # Peakiness: Max Power / Mean Power
    # Limit Cycle: High Peakiness (Energy concentrated in harmonics)
    # Chaos: Low Peakiness (Energy spread)
    
    peakiness = np.max(Pxx) / (np.mean(Pxx) + 1e-8)
    
    # Spectral Entropy
    P_norm = Pxx / (np.sum(Pxx) + 1e-8)
    entropy = -np.sum(P_norm * np.log(P_norm + 1e-12))
    
    return {
        'peakiness': peakiness,
        'entropy': entropy
    }

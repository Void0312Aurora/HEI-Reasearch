
import zoneinfo
import time
import numpy as np
import torch
import torch.utils.benchmark as benchmark

# ----------------- NumPy Implementation (Optimized) -----------------
def _minkowski_metric_inner_np(v1, v2):
    return v1[..., 0]*v2[..., 0] + v1[..., 1]*v2[..., 1] - v1[..., 2]*v2[..., 2]

def _compute_hyperboloid_generators_np(h):
    X, Y, T = h[..., 0], h[..., 1], h[..., 2]
    # J (Rotation)
    zeros = np.zeros_like(X)
    J = np.stack([-Y, X, zeros], axis=-1)
    # Kx (Boost X)
    Kx = np.stack([T, zeros, X], axis=-1)
    # Ky (Boost Y)
    Ky = np.stack([zeros, T, Y], axis=-1)
    return J, Kx, Ky

def inertia_gradient_np(h, xi):
    # Simplified vectorized version
    h = np.asarray(h)
    xi = np.asarray(xi)
    n = h.shape[0]
    w = np.ones(n)
    
    def compute_local_energy(h_batch):
        J, Kx, Ky = _compute_hyperboloid_generators_np(h_batch)
        gens = [J, Kx, Ky]
        I_local = np.zeros((n, 3, 3))
        for i, gi in enumerate(gens):
            for j, gj in enumerate(gens):
                I_local[:, i, j] = _minkowski_metric_inner_np(gi, gj)
        
        mv = np.einsum('nij,j->ni', I_local, xi)
        local_K = 0.5 * np.einsum('i,ni->n', xi, mv)
        return local_K * w

    h_norm = np.linalg.norm(h, axis=1)
    eps_vec = np.maximum(1e-7, 1e-4 * h_norm)
    
    forces = np.zeros_like(h)
    for i in range(3):
        h_p = h.copy(); h_p[:, i] += eps_vec
        h_m = h.copy(); h_m[:, i] -= eps_vec
        forces[:, i] = (compute_local_energy(h_p) - compute_local_energy(h_m)) / (2 * eps_vec)
    return forces

# ----------------- PyTorch Implementation -----------------
def _minkowski_metric_inner_torch(v1, v2):
    return v1[..., 0]*v2[..., 0] + v1[..., 1]*v2[..., 1] - v1[..., 2]*v2[..., 2]

def _compute_hyperboloid_generators_torch(h):
    X, Y, T = h[..., 0], h[..., 1], h[..., 2]
    zeros = torch.zeros_like(X)
    J = torch.stack([-Y, X, zeros], dim=-1)
    Kx = torch.stack([T, zeros, X], dim=-1)
    Ky = torch.stack([zeros, T, Y], dim=-1)
    return J, Kx, Ky

def inertia_gradient_torch(h, xi):
    n = h.shape[0]
    w = torch.ones(n, device=h.device, dtype=h.dtype)
    
    def compute_local_energy(h_batch):
        J, Kx, Ky = _compute_hyperboloid_generators_torch(h_batch)
        gens = [J, Kx, Ky]
        # I_local construction using broadcasting
        # gens is list of 3 tensors of shape (N, 3)
        # Result I_local (N, 3, 3)
        # Stack gens to (3, N, 3) -> permute to (N, 3, 3_vec_comp)
        G = torch.stack(gens, dim=1) 
        
        # Minkowski dot product: x1x2 + y1y2 - t1t2
        # G has shape (N, 3_gen, 3_coord)
        # We want inner product between generators i and j.
        # Let's do it explicitly to match logic:
        # <G_i, G_j>_L
        
        # Method 1: Explicit loop (slow in python but ok for JIT)
        # Method 2: Vectorized
        # G_metric = G * [1, 1, -1] ? 
        Metric = torch.tensor([1, 1, -1], device=h.device, dtype=h.dtype)
        G_M = G * Metric.view(1, 1, 3)
        # Inner product: sum over last dim
        # I_local[n, i, j] = sum_k G[n, i, k] * G_M[n, j, k]
        I_local = torch.matmul(G, G_M.transpose(1, 2)) # (N, 3, 3)
        
        # Energy
        mv = torch.einsum('nij,j->ni', I_local, xi)
        local_K = 0.5 * torch.einsum('i,ni->n', xi, mv)
        return local_K * w

    h_norm = torch.norm(h, dim=1)
    eps_vec = torch.maximum(torch.tensor(1e-7, device=h.device), 1e-4 * h_norm)
    
    forces = torch.zeros_like(h)
    for i in range(3):
        h_p = h.clone(); h_p[:, i] += eps_vec
        h_m = h.clone(); h_m[:, i] -= eps_vec
        forces[:, i] = (compute_local_energy(h_p) - compute_local_energy(h_m)) / (2 * eps_vec)
    return forces

def run_benchmark():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark")
        # return # Do not return, run CPU anyway to verify script works
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    sizes = [50, 100, 500, 1000, 5000]
    
    print(f"{'N':<10} {'NumPy (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    
    xi_np = np.array([0.1, 0.2, 0.3])
    xi_torch = torch.tensor([0.1, 0.2, 0.3], device=device, dtype=torch.float64)

    for n in sizes:
        # Setup Data
        h_np = np.random.randn(n, 3)
        h_np[:, 0] = np.abs(h_np[:, 0]) # dummy
        # project pseudo
        h_np[:, 2] = np.sqrt(1 + h_np[:, 0]**2 + h_np[:, 1]**2)
        
        h_torch = torch.tensor(h_np, device=device, dtype=torch.float64)
        
        # Warmup
        inertia_gradient_np(h_np, xi_np)
        inertia_gradient_torch(h_torch, xi_torch)
        
        # NumPy Timing
        t0 = time.perf_counter()
        for _ in range(100):
            inertia_gradient_np(h_np, xi_np)
        t_np = (time.perf_counter() - t0) / 100 * 1000
        
        # Torch Timing
        t0 = time.perf_counter()
        for _ in range(100):
            inertia_gradient_torch(h_torch, xi_torch)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / 100 * 1000
        
        print(f"{n:<10} {t_np:<15.3f} {t_torch:<15.3f} {t_np/t_torch:<10.2f}x")

if __name__ == "__main__":
    run_benchmark()

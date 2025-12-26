"""
Random-U Ablation Test for V12 Validation

Purpose: Verify that V12's positive gains are due to learned geometry, not metric bias.
Method: Replace learned U with random SO(3) rotations and recompute alignment gains.
Expected: Random-U should produce ~0 or negative gains.
"""

import sys
import os
import argparse
import torch
import pickle

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora import AuroraDataset
from aurora.gauge import GaugeField

def generate_random_so3(N, k=3, device='cuda'):
    """Generate N random SO(3) matrices using QR decomposition."""
    # Generate random matrices
    A = torch.randn(N, k, k, device=device)
    
    # QR decomposition to get orthogonal matrices
    Q, R = torch.linalg.qr(A)
    
    # Ensure det(Q) = +1 (SO(3) not just O(3))
    # If det(Q) = -1, flip sign of one column
    dets = torch.det(Q)
    flip_mask = (dets < 0).unsqueeze(-1).unsqueeze(-1)
    Q_corrected = Q.clone()
    Q_corrected[:, :, 0] = torch.where(flip_mask.squeeze(-1), -Q[:, :, 0], Q[:, :, 0])
    
    return Q_corrected

def compute_gain_with_U(J, edges, U_matrices):
    """Compute alignment gain given J and U matrices."""
    u_idx = edges[:, 0]
    v_idx = edges[:, 1]
    
    J_u = J[u_idx]
    J_v = J[v_idx]
    
    # Raw alignment
    raw_align = torch.sum(J_v * J_u, dim=-1)
    
    # Gauge alignment with provided U
    J_u_trans = torch.matmul(U_matrices, J_u.unsqueeze(-1)).squeeze(-1)
    gauge_align = torch.sum(J_v * J_u_trans, dim=-1)
    
    return raw_align.mean().item(), gauge_align.mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--semantic_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cilin")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--n_random_trials", type=int, default=10, help="Number of random U trials")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    
    import io
    class DeviceUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=False)
            return super().find_class(module, name)

    try:
        with open(args.checkpoint, 'rb') as f:
            ckpt = DeviceUnpickler(f).load()
    except Exception as e:
        print(f"Pickle load failed: {e}. Trying torch.load...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    J = torch.tensor(ckpt['J'], device=device)
    x = torch.tensor(ckpt['x'], device=device)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset (limit={args.limit})...")
    ds = AuroraDataset(args.dataset, limit=args.limit)
    
    # Load semantic edges
    sem_train_list = ds.load_semantic_edges(args.semantic_path, split='train')
    sem_train = torch.tensor([(u, v) for u, v, w in sem_train_list], dtype=torch.long, device=device)
    
    edges_struct = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
    edges_all = torch.cat([edges_struct, sem_train], dim=0)
    
    # Reconstruct gauge field to get learned U
    print("Reconstructing GaugeField...")
    gauge_field = GaugeField(edges_all, logical_dim=3, group='SO',
                             backend_type='neural', input_dim=5).to(device)
    gauge_field.load_state_dict(ckpt['gauge_field'])
    
    # Get learned U
    with torch.no_grad():
        U_learned = gauge_field.get_U(x=x)
    
    print("\n=== Ablation Test: Learned U vs Random U ===\n")
    
    # Test on semantic edges only (where V12 shows improvement)
    # Find semantic edge indices in gauge_field.edges
    edge_map = gauge_field.edge_map
    sem_indices = []
    for u, v in sem_train.cpu().numpy():
        if (u, v) in edge_map:
            idx, _ = edge_map[(u, v)]
            sem_indices.append(idx)
        elif (v, u) in edge_map:
            idx, _ = edge_map[(v, u)]
            sem_indices.append(idx)
    
    sem_indices = torch.tensor(sem_indices, dtype=torch.long, device=device)
    sem_edges_subset = gauge_field.edges[sem_indices]
    U_learned_subset = U_learned[sem_indices]
    
    # Compute gain with learned U
    raw_learned, gauge_learned = compute_gain_with_U(J, sem_edges_subset, U_learned_subset)
    gain_learned = gauge_learned - raw_learned
    
    print(f"--- Learned U (V12 Model) ---")
    print(f"  Raw Correlation:  {raw_learned:.4f}")
    print(f"  Gauge Alignment:  {gauge_learned:.4f}")
    print(f"  Gain:             {gain_learned:+.4f}")
    
    # Run multiple trials with random U
    print(f"\n--- Random U Baseline ({args.n_random_trials} trials) ---")
    random_gains = []
    
    for trial in range(args.n_random_trials):
        U_random = generate_random_so3(len(sem_indices), k=3, device=device)
        _, gauge_random = compute_gain_with_U(J, sem_edges_subset, U_random)
        gain_random = gauge_random - raw_learned  # Same raw baseline
        random_gains.append(gain_random)
    
    random_gains = torch.tensor(random_gains)
    
    print(f"  Mean Random Gain: {random_gains.mean():.4f}")
    print(f"  Std Random Gain:  {random_gains.std():.4f}")
    print(f"  Min Random Gain:  {random_gains.min():.4f}")
    print(f"  Max Random Gain:  {random_gains.max():.4f}")
    
    # Statistical test
    print(f"\n=== Statistical Comparison ===")
    improvement = gain_learned - random_gains.mean().item()
    n_std = improvement / (random_gains.std().item() + 1e-9)
    
    print(f"  Learned vs Random Improvement: {improvement:+.4f}")
    print(f"  Z-score (# of std devs):       {n_std:.2f}")
    
    if n_std > 2.0:
        print(f"  ✓ Learned U is SIGNIFICANTLY better than random (p < 0.05)")
    elif n_std > 0.5:
        print(f"  ~ Learned U shows marginal improvement")
    else:
        print(f"  ✗ Learned U not better than random (possible metric bias)")
    
    # Test on structural edges low-raw bucket for comparison
    print(f"\n=== Structural Edges [0.0-0.3) Bucket (N=9) ===")
    struct_indices = []
    for u, v in edges_struct.cpu().numpy():
        if (u, v) in edge_map:
            idx, _ = edge_map[(u, v)]
            struct_indices.append(idx)
        elif (v, u) in edge_map:
            idx, _ = edge_map[(v, u)]
            struct_indices.append(idx)
    
    struct_indices = torch.tensor(struct_indices, dtype=torch.long, device=device)
    struct_edges_subset = gauge_field.edges[struct_indices]
    U_struct = U_learned[struct_indices]
    
    # Filter to [0.0, 0.3) raw bucket
    u_idx = struct_edges_subset[:, 0]
    v_idx = struct_edges_subset[:, 1]
    J_u = J[u_idx]
    J_v = J[v_idx]
    raw_align = torch.sum(J_v * J_u, dim=-1)
    
    low_raw_mask = raw_align < 0.3
    if low_raw_mask.sum() > 0:
        low_raw_edges = struct_edges_subset[low_raw_mask]
        U_low_raw = U_struct[low_raw_mask]
        
        raw_low, gauge_low = compute_gain_with_U(J, low_raw_edges, U_low_raw)
        gain_low_learned = gauge_low - raw_low
        
        print(f"  Learned U Gain: {gain_low_learned:+.4f} (N={low_raw_mask.sum()})")
        
        # Random U on same bucket
        random_low_gains = []
        for trial in range(args.n_random_trials):
            U_random_low = generate_random_so3(low_raw_mask.sum(), k=3, device=device)
            _, gauge_random_low = compute_gain_with_U(J, low_raw_edges, U_random_low)
            random_low_gains.append(gauge_random_low - raw_low)
        
        random_low_gains = torch.tensor(random_low_gains)
        print(f"  Random U Gain:  {random_low_gains.mean():.4f} ± {random_low_gains.std():.4f}")
        
        improvement_low = gain_low_learned - random_low_gains.mean().item()
        print(f"  Improvement:    {improvement_low:+.4f}")
    else:
        print(f"  No edges in [0.0, 0.3) bucket")

if __name__ == "__main__":
    main()

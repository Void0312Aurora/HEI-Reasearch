
import torch
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from aurora.gauge import GaugeField
from aurora.geometry import random_hyperbolic_init

def test_orientation_consistency():
    print("Testing Gauge Field Orientation Consistency...")
    
    # 1. Setup
    logical_dim = 3
    nodes = 3
    # Triangle: 0-1, 1-2, 2-0
    # Edges list for GaugeField: needs to cover transport.
    # We define edges: (0,1), (1,2), (2,0)
    edges = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
    
    # Init GaugeField
    gauge = GaugeField(edges, logical_dim, group='SO')
    
    # 2. Get U matrices
    # gauge.get_U() returns U for the edges in order
    U_all = gauge.get_U() # (3, 3, 3)
    U_01 = U_all[0]
    U_12 = U_all[1]
    U_20 = U_all[2]
    
    # 3. Compute Forward Holonomy (0 -> 1 -> 2 -> 0)
    # H_forward = U_20 * U_12 * U_01 (Order depends on convention: J_new = U J_old)
    # J_1 = U_01 J_0
    # J_2 = U_12 J_1 = U_12 U_01 J_0
    # J_0_new = U_20 J_2 = U_20 U_12 U_01 J_0
    H_forward = torch.matmul(U_20, torch.matmul(U_12, U_01))
    
    # 4. Compute Reverse Holonomy (0 -> 2 -> 1 -> 0)
    # Path: 0 -> 2 (Edge 2 inverted), 2 -> 1 (Edge 1 inverted), 1 -> 0 (Edge 0 inverted)
    # U_inv = U^T for SO(k)
    U_02 = U_20.t()
    U_21 = U_12.t()
    U_10 = U_01.t()
    
    # J_2 = U_02 J_0
    # J_1 = U_21 J_2 = U_21 U_02 J_0
    # J_0_new = U_10 J_1 = U_10 U_21 U_02 J_0
    H_reverse = torch.matmul(U_10, torch.matmul(U_21, U_02))
    
    # 5. Check Inversion Property
    # H_reverse should be H_forward^-1 = H_forward^T
    identity_approx = torch.matmul(H_forward, H_reverse)
    diff = torch.norm(identity_approx - torch.eye(logical_dim))
    
    print(f"H_fwd * H_rev deviation from I: {diff.item():.2e}")
    if diff < 1e-6:
        print(">>> PASS: Inverse Property Verified.")
    else:
        print(">>> FAIL: Inverse Property Violated.")

    # 6. Check Algebra Consistency (Omega_rev = -Omega_fwd)
    # Use Matrix Logarithm approximated or torch.linalg?
    # torch.linalg.matrix_exp is available, but log might need custom or complex handling?
    # For small rotations (init * 0.01), we can approximate or use simple log if available?
    # PyTorch doesn't have stable matrix_log for real matrices in all versions easily.
    # But since H is rotation, trace(H) = 1 + 2cos(theta).
    # Axis-Angle extraction.
    
    def get_omega_vec(R):
        # Extract skew parts
        skew = 0.5 * (R - R.t())
        # For SO(3), skew is [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        return torch.tensor([skew[2,1], skew[0,2], skew[1,0]])
        
    omega_fwd = get_omega_vec(H_forward)
    omega_rev = get_omega_vec(H_reverse)
    
    print(f"Omega Fwd: {omega_fwd}")
    print(f"Omega Rev: {omega_rev}")
    
    sum_omega = omega_fwd + omega_rev
    diff_omega = torch.norm(sum_omega)
    
    print(f"Sum Omega Norm: {diff_omega.item():.2e}")
    
    if diff_omega < 1e-6:
        print(">>> PASS: Orientation Consistency Verified (Omega_rev = -Omega_fwd).")
    else:
        print(">>> FAIL: Orientation Consistency Violated.")

if __name__ == "__main__":
    test_orientation_consistency()

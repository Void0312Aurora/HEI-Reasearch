
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from aurora.gauge import GaugeField
from aurora.geometry import log_map

def test_connection_precession():
    print("Testing Wong Connection and Precession...")
    
    # 1. Setup 2 nodes, 1 edge
    logical_dim = 3
    nodes = 2
    dim = 5
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    
    # 2. Init GaugeField with fixed omega
    gauge = GaugeField(edges, logical_dim, group='SO')
    
    # Set omega to rotation around Z axis (indices 0,1 in vector/matrix)
    # Omega_01 = 1.0. Matrix: [[0, 1, 0], [-1, 0, 0], [0,0,0]]
    # This corresponds to rotation in XY plane.
    # omega_params shape (E, k, k).
    with torch.no_grad():
        gauge.omega_params.fill_(0.0)
        # Set element [0, 1] = 1.0 (will become 0.5 * (1 - (-1)) = 1)
        # Remember GaugeField enforces skew: 0.5 * (P - P.t())
        gauge.omega_params[0, 0, 1] = 2.0 
        gauge.omega_params[0, 1, 0] = -2.0 # Redundant but clear
    
    # 3. Setup State
    # x0 at origin, x1 at distance 1 along axis 0
    x = torch.zeros(nodes, dim)
    x[0, 0] = 1.0 # Origin in Hyperboloid (1, 0, 0,...)
    x[1, 0] = 1.5 # Some distance away
    x[1, 1] = 0.5
    # Normalize
    # x = x / |x|_M?
    # Simple setup: Euclidean approximation for quick test or use geometry utils
    # Let's use x tensors that are mathematically valid if possible, but for compute_connection
    # it relies on log_map, so geometry matters.
    # Let's use valid hyperbolic points.
    
    # Correct Hyperboloid init:
    x[0] = torch.tensor([1.0, 0, 0, 0, 0])
    # x[1]: boost along axis 1
    import numpy as np
    cosh_d = 2.0
    sinh_d = np.sqrt(cosh_d**2 - 1)
    x[1] = torch.tensor([cosh_d, sinh_d, 0, 0, 0])
    
    # 4. Velocity
    # v at x0 pointing towards x1.
    # Log_x0(x1) is vector along axis 1.
    v = torch.zeros(nodes, dim)
    # v[0] = Log_x0(x1) (normalized or not)
    # Let's use exact LogMap
    dir_01 = log_map(x[0].unsqueeze(0), x[1].unsqueeze(0)).squeeze(0)
    
    # Set v[0] = dir_01 (Full speed along edge)
    v[0] = dir_01
    
    # 5. Compute Connection A(v)
    A = gauge.compute_connection(x, v)
    
    print(f"A shape: {A.shape}")
    A_0 = A[0]
    
    print(f"A[0] (Generator):\n{A_0}")
    
    # Check if A_0 matches Omega
    # Omega was XY rotation. A should be Omega * coeff.
    # coeff = <v, dir> / |dir|^2 = 1.0
    # So A_0 should be close to [[0, 2, 0], [-2, 0, 0]...] 
    # Wait, GaugeField enforces 0.5 * (W - W.t()).
    # W[0,1]=2, W[1,0]=-2 -> (2 - (-2))/2 = 2.
    # So Omega_01 = 2.
    
    val = A_0[0, 1].item()
    print(f"A[0][0,1] value: {val:.4f}")
    
    if abs(val - 2.0) < 1e-4:
        print(">>> PASS: Connection Value Correct.")
    else:
        print(">>> FAIL: Connection Value Mismatch.")
        
    # 6. Test Precession
    # J = [1, 0, 0] (X axis)
    # A corresponds to rotation around Z (XY plane).
    # dJ/dt = -[A, J] ?? 
    # J(t) = exp(-At) J(0).
    # Rot Z acting on X -> Y.
    # Rotation Angle theta = -2 * dt.
    
    J = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # Node 0, Node 1
    dt = 0.1
    
    U_step = torch.matrix_exp(-A * dt)
    J_new = torch.matmul(U_step, J.unsqueeze(-1)).squeeze(-1)
    
    J_0_new = J_new[0]
    print(f"J_0 (original): {J[0]}")
    print(f"J_0 (new): {J_0_new}")
    
    # Explicit check:
    # theta = 2 * 0.1 = 0.2
    # Rot(theta) * [1,0,0] = [cos(theta), sin(theta), 0]?
    # -A corresponds to rotation?
    # A is skew. exp(-A) is orthogonal.
    # W = [[0, 2, 0], [-2, 0, 0]].
    # -W = [[0, -2], [2, 0]].
    # This is 2 * [[0, -1], [1, 0]].
    # [[0, -1], [1, 0]] is generator of +90 deg rotation (X->Y).
    # So we rotate by 0.2 rad.
    # cos(0.2) ~ 0.98. sin(0.2) ~ 0.198.
    # J_x should be 0.98, J_y should be 0.198.
    
    if abs(J_0_new[0] - np.cos(0.2)) < 1e-2 and abs(J_0_new[1] - np.sin(0.2)) < 1e-2:
        print(">>> PASS: Precession Dynamics Correct.")
    else:
         print(">>> FAIL: Precession Dynamics Incorrect.")
         print(f"Expected: x={np.cos(0.2):.4f}, y={np.sin(0.2):.4f}")

if __name__ == "__main__":
    test_connection_precession()

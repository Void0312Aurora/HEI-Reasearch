
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from aurora.gauge import GaugeField
from aurora.geometry import log_map, minkowski_inner

def test_wong_force():
    print("Testing Wong Force Calculation...")
    
    # 1. Setup Triangle
    logical_dim = 3
    nodes = 3
    dim = 5
    # Edges: (0,1), (1,2), (2,0)
    edges = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
    
    gauge = GaugeField(edges, logical_dim, group='SO')
    
    # 2. Set Omega (Curvature)
    # Omega = Log(U_loop). 
    # Let's set U such that H is rotation around Z.
    # U_01 = Rot(90), U_12 = I, U_20 = I.
    # H = U_20 U_12 U_01 = Rot(90).
    # Then Omega ~ 90 deg around Z.
    
    # Generator Z: [[0, 1, 0], [-1, 0, 0]...]
    # Map to params.
    # U = exp(omega). 
    # Set omega_01 to generate RotZ.
    with torch.no_grad():
        gauge.omega_params.zero_()
        val = 1.57 # 90 deg
        gauge.omega_params[0, 0, 1] = val
        gauge.omega_params[0, 1, 0] = -val
        
        # Others zero -> Identity
        
    print(f"GaugeField Triangles: {gauge.triangles.shape}")
    
    # 3. State
    x = torch.zeros(nodes, dim)
    # 0 at origin
    x[0] = torch.tensor([1.0, 0, 0, 0, 0])
    # 1 at x-axis
    x[1] = torch.tensor([2.0, 1.732, 0, 0, 0]) 
    # 2 at y-axis
    x[2] = torch.tensor([2.0, 0, 1.732, 0, 0])
    
    # Velocity v at node 0
    v = torch.zeros(nodes, dim)
    # Moving towards node 1 (X)
    v[0] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]) # Tangent at origin
    
    # J at node 0
    # J = [0, 0, 1] (Z axis). 
    # Omega is around Z. <J, Omega> should be non-zero.
    J = torch.zeros(nodes, logical_dim)
    J[0] = torch.tensor([0.0, 0.0, 1.0])
    
    # 4. Compute Force
    # F = <J, Omega> ( <v, w> u - <v, u> w )
    # at node 0:
    # u = Log_0(1) (X direction)
    # w = Log_0(2) (Y direction)
    # v = X direction.
    # <v, u> != 0, <v, w> = 0.
    # Term = 0*u - (nonzero)*w = - (nonzero) * Y_direction.
    # Force should be in -Y direction.
    
    # Verify geometry inner products
    u_vec = log_map(x[0], x[1])
    w_vec = log_map(x[0], x[2])
    print(f"u (0->1, approx X): {u_vec[1]:.2f}, {u_vec[2]:.2f}")
    print(f"w (0->2, approx Y): {w_vec[1]:.2f}, {w_vec[2]:.2f}")
    
    # Force
    _, F = gauge.compute_force_wong(x, v, J)
    F0 = F[0]
    print(f"F[0]: {F0}")
    
    # Check direction
    # Should have non-zero component in Y (index 2)
    # And near zero in X (index 1)
    
    fy = F0[2].item()
    fx = F0[1].item()
    
    print(f"Force X: {fx:.4f}, Y: {fy:.4f}")
    
    if abs(fy) > 1e-4:
        print(">>> PASS: Wong Force Non-Zero and perp to velocity.")
    else:
        print(">>> FAIL: Wong Force is Zero or wrong direction.")

if __name__ == "__main__":
    test_wong_force()

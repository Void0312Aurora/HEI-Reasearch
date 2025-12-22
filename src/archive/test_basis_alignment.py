
import numpy as np
from hei.lie import vec_to_matrix, moebius_flow
from hei.geometry import cayley_uhp_to_disk, disk_to_hyperboloid, cayley_disk_to_uhp
from hei.inertia import _compute_hyperboloid_generators

def test_basis():
    # Points to test
    points = [1.0 + 2.0j, 0.5 + 0.5j, 2,0 + 1.0j]
    dt = 1e-6
    
    for z_0 in points:
        z_d = cayley_uhp_to_disk(z_0)
        h_0 = disk_to_hyperboloid(z_d)
        
        print(f"\n--- Point: z_uhp={z_0}, h={h_0} ---")
    
        # Get Hyperboloid Generators (Columns of Jacobian)
        generators = _compute_hyperboloid_generators(h_0[None, :])
        
        M_Lie = np.zeros((3, 3))
        M_Gen = np.zeros((3, 3))
        
        for i in range(3):
            xi = np.zeros(3); xi[i] = 1.0
            # Lie Flow
            z_dot = moebius_flow(xi, z_0)
            # Numerical h_dot
            z_new = z_0 + z_dot * dt
            h_new = disk_to_hyperboloid(cayley_uhp_to_disk(z_new))
            v_diff = (h_new - h_0) / dt
            M_Lie[:, i] = v_diff
            
            # Generator
            M_Gen[:, i] = generators[i][0]
            
        print("M_Lie:\n", M_Lie)
        print("M_Gen:\n", M_Gen)
            
        # Solve for P
        try:
            M_Gen_inv = np.linalg.pinv(M_Gen)
            P_T = M_Gen_inv @ M_Lie
            P = P_T.T
            
            print("Calculated Transformation Matrix P:")
            print(P)
            
        except np.linalg.LinAlgError:
            print("Singular matrix.")

if __name__ == "__main__":
    test_basis()

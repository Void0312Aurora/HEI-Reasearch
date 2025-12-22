
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN, PotentialOracleN
from hei_n.inertia_n import IdentityInertia

class LinearPotential(PotentialOracleN):
    def potential(self, x): return 0.0
    def gradient(self, x):
        # Always pull towards -x direction in embedding?
        # Let's say we want to pull towards Origin e0.
        # x is at some place.
        # Gradient of dist(x, e0) points away from e0.
        # So Force should point towards e0.
        # grad = vec pointing away.
        # Force = - grad.
        
        # At x = (cosh, sinh, 0)
        # grad dist approximates (sinh, cosh, 0)? (In Euclidean).
        # Projecting to tangent...
        
        # Let's just manually set grad to be "Outward".
        # x_curr = (cosh, sinh)
        # grad = (0, 1) (Points positive y/x).
        
        grad = np.zeros_like(x)
        grad[..., 1] = 1.0 
        return grad

def debug_sign():
    N = 1
    dim = 2 # 1+1
    G = np.zeros((N, dim, dim))
    G[0] = np.eye(dim)
    
    # Place at x = (cosh 1, sinh 1) ~ (1.54, 1.17)
    s = 1.0
    # Boost x
    G[0, 0, 0] = np.cosh(s)
    G[0, 1, 1] = np.cosh(s)
    G[0, 0, 1] = np.sinh(s)
    G[0, 1, 0] = np.sinh(s)
    
    x_old = G[..., 0]
    dist_old = np.arccosh(x_old[0, 0])
    print(f"Old Dist: {dist_old:.4f}")
    
    # Oracle: Gradient points OUTWARD (positive x1).
    # Force points INWARD (negative x1).
    # Torque should create velocity towards Origin (negative).
    
    oracle = LinearPotential()
    inertia = IdentityInertia()
    config = ContactConfigN(dt=0.01) # Small step
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    state = ContactStateN(G=G.copy(), M=np.zeros_like(G), z=0.0)
    
    # Step
    new_state = integrator.step(state)
    
    x_new = new_state.G[..., 0]
    dist_new = np.arccosh(x_new[0, 0])
    print(f"New Dist: {dist_new:.4f}")
    
    diff = dist_new - dist_old
    print(f"Change: {diff}")
    
    if diff < 0:
        print("SUCCESS: Radius decreased (Pulled In). Torque Sign Correct.")
    else:
        print("FAILURE: Radius increased (Pushed Out). Torque Sign INVERTED.")

if __name__ == "__main__":
    debug_sign()

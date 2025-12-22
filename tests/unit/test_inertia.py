
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN, PotentialOracleN
from hei_n.inertia_n import IdentityInertia, RadialInertia
from hei_n.geometry_n import dist_n, minkowski_inner

class MockOracle:
    def potential(self, x): return 0.0
    def gradient(self, x): return np.zeros_like(x)

def test_inertia():
    N = 1
    dim = 3
    G = np.zeros((N, dim, dim))
    G[0] = np.eye(dim)
    
    # Place particle at some distance.
    # To do this, we need G to be non-identity.
    # Let's boost it in x-direction.
    # x = (cosh s, sinh s, 0).
    s = 1.0
    # Boost matrix B = [[cosh, sinh], [sinh, cosh]]
    G[0, 0, 0] = np.cosh(s)
    G[0, 1, 1] = np.cosh(s)
    G[0, 0, 1] = np.sinh(s)
    G[0, 1, 0] = np.sinh(s)
    # x0 is col 0. -> (cosh s, sinh s, 0).
    
    x_curr = G[..., 0]
    print(f"Position x0: {x_curr[0, 0]:.3f}") # Should be > 1
    
    # Momentum (Velocity)
    M = np.zeros((N, dim, dim))
    # Give it velocity in x-direction (radial).
    M[0, 0, 1] = 1.0
    M[0, 1, 0] = 1.0
    
    # 1. Identity Inertia
    config = ContactConfigN(dt=0.1)
    oracle = MockOracle()
    ii = IdentityInertia()
    
    integrator_id = ContactIntegratorN(oracle, ii, config)
    state = ContactStateN(G=G.copy(), M=M.copy(), z=0.0)
    
    # Step
    new_state_id = integrator_id.step(state)
    print(f"Identity Step - New M: {new_state_id.M[0, 0, 1]:.3f}")
    
    # 2. Radial Inertia
    # Mass increases with distance.
    # x0 = cosh(1) ~ 1.54. r = 1.
    # RadialInertia(alpha=1.0) -> m = 1 + (1.54 - 1) = 1.54.
    ri = RadialInertia(alpha=1.0)
    integrator_rad = ContactIntegratorN(oracle, ri, config)
    
    # Geo Force = + (alpha * M^2) / (2 m^2) pointed outwards? 
    # Let's check logic in inertia_n.
    # F_geom = prefactor * e0.
    # But prefactor was positive.
    # Force = - grad V. Here F_geom is treated as -grad K.
    # In integrator: grad_total = - F_geom.
    # Force_world = - proj(grad_total) = + proj(F_geom).
    # F_geom points in e0 direction (time-like).
    # For a particle on hyperboloid, e0 component is projected out?
    # Wait. Tangent space at (cosh, sinh) is orthogonal to (cosh, -sinh).
    # Tangent vector t = (sinh, cosh).
    # e0 = (1, 0).
    # <t, e0>_M = -sinh. Not 0.
    # So F_geom has a tangent component!
    # It points "inwards" or "outwards"?
    # e0 is "up" in time. t is "out" in space.
    # e0 = -sinh * t + ...
    # Let's see simulation result.
    
    new_state_rad = integrator_rad.step(state)
    print(f"Radial Step - New M: {new_state_rad.M[0, 0, 1]:.3f}")
    
    # Comparison
    # With Radial Inertia, "Heavy" mass should result in SLOWER movement (smaller effective velocity xi).
    # xi = M / m.
    # In Identity, xi = M = 1.0.
    # In Radial, xi = 1.0 / 1.54 = 0.65.
    # So G should change less.
    
    x_id = new_state_id.G[..., 0]
    x_rad = new_state_rad.G[..., 0]
    
    dist_id = dist_n(x_curr, x_id)[0]
    dist_rad = dist_n(x_curr, x_rad)[0]
    
    print(f"Moved Dist (Identity): {dist_id:.4f}")
    print(f"Moved Dist (Radial):   {dist_rad:.4f}")
    
    assert dist_rad < dist_id, "Radial inertia (heavier) should move slower for same momentum"
    print("Test Passed: Variable Inertia Logic.")

if __name__ == "__main__":
    test_inertia()

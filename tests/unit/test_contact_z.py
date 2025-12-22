
import numpy as np
import sys
import os

# Allow import from src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN
from hei_n.lie_n import exp_so1n

class MockOracle:
    def potential(self, x):
        # V = 0.5 * d(x, e0)^2 approx x[0] - 1
        # Simple well
        return 1.0
        
    def gradient(self, x):
        return np.zeros_like(x)

def test_z_dynamics():
    # Setup
    N = 2
    dim = 3
    G = np.zeros((N, dim, dim))
    for i in range(N): G[i] = np.eye(dim)
    
    M = np.zeros((N, dim, dim))
    # Give some velocity
    M[:, 0, 1] = 1.0
    M[:, 1, 0] = 1.0
    
    z_init = 0.0
    state = ContactStateN(G=G, M=M, z=z_init)
    
    config = ContactConfigN(dt=0.1, gamma=0.1)
    oracle = MockOracle()
    integrator = ContactIntegratorN(oracle, config)
    
    # Step
    new_state = integrator.step(state)
    
    # Check
    # T = sum(v^2 + 0.5 w^2). Here v=1.0. T for 2 particles = 1+1 = 2.0?
    # M has shape (N, dim, dim).
    # Particle 0: M[0, 0, 1] = 1 -> v=1. T_0 = 1.
    # Particle 1: M[1, 0, 1] = 1 -> v=1. T_1 = 1.
    # Total T = 2.0.
    
    # L = T - V - gamma * z
    # L = 2.0 - 1.0 - 0.1 * 0.0 = 1.0
    
    # dz = L * dt = 1.0 * 0.1 = 0.1
    # expected z = 0.1
    
    print(f"Old z: {state.z}")
    print(f"New z: {new_state.z}")
    
    assert abs(new_state.z - 0.1) < 1e-6
    print("Test Passed: Z Dynamics correct.")

if __name__ == "__main__":
    test_z_dynamics()

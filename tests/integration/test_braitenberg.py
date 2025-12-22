
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN, PotentialOracleN
from hei_n.inertia_n import IdentityInertia, RadialInertia
from hei_n.geometry_n import dist_n

class BraitenbergTarget(PotentialOracleN):
    def __init__(self, active=True):
        self.active = active
        
    def potential(self, x): return 0.0
    
    def gradient(self, x):
        if not self.active:
            return np.zeros_like(x)
            
        # Constant Force field pushing towards x-axis positive
        # Target at Infinity (Ideal Point)
        # Gradient of Busemann function?
        # Let's just say Gradient points to negative y.
        # Force points to positive y.
        
        # x is (x0, x1, x2)
        # Force = (0, 0, 1.0) projected.
        
        grad = np.zeros_like(x)
        grad[..., 2] = -5.0 # Force = +5.0 in z direction
        return grad

def run_braitenberg():
    N = 1
    dim = 3
    G = np.zeros((N, dim, dim))
    G[0] = np.eye(dim)
    
    state = ContactStateN(G=G, M=np.zeros_like(G), z=0.0)
    
    inertia = IdentityInertia()
    oracle = BraitenbergTarget(active=True)
    # Very low damping to see coasting
    config = ContactConfigN(dt=0.01, gamma=0.1) 
    
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    traj_y = []
    velocities = []
    
    # Phase 1: Acceleration (0 - 50 steps)
    for i in range(50):
        state = integrator.step(state)
        traj_y.append(state.x[0, 2])
        # Monitor Momentum
        v = state.M[0, 0, 2] # Boost component z
        velocities.append(v)
        
    print(f"Velocity at Cutoff: {velocities[-1]:.4f}")
    
    # Phase 2: Signal Lost (50 - 100 steps)
    oracle.active = False
    print("Signal Lost! Force -> 0.")
    
    for i in range(50):
        state = integrator.step(state)
        traj_y.append(state.x[0, 2])
        v = state.M[0, 0, 2]
        velocities.append(v)
        
    print(f"Velocity at End: {velocities[-1]:.4f}")

    # Analysis
    # SGD would drop to 0 instantly (v ~ F).
    # HEI should decay as v = v0 * exp(-gamma * t).
    # gamma=0.1, dt=0.01. 50 steps -> t=0.5.
    # Decay factor = exp(-0.05) ~ 0.95.
    # So velocity should persist.
    
    ratio = velocities[-1] / velocities[49]
    print(f"Retention Ratio: {ratio:.4f}")
    
    if ratio > 0.8:
        print("SUCCESS: Strong Momentum Persistence.")
    elif ratio > 0.1:
        print("PARTIAL: Some Persistence.")
    else:
        print("FAILURE: Instant Stop (SGD-like behavior).")
        
    # Plot
    plt.figure()
    plt.plot(velocities)
    plt.axvline(x=50, color='r', linestyle='--', label='Signal Cutoff')
    plt.title("Constraint 2: Intent Persistence (Velocity Profile)")
    plt.xlabel("Step")
    plt.ylabel("Internal Momentum (Intent)")
    plt.legend()
    plt.savefig("validation_braitenberg.png")
    print("Saved plot to validation_braitenberg.png")

if __name__ == "__main__":
    run_braitenberg()

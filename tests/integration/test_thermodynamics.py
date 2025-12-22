
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN, PotentialOracleN
from hei_n.inertia_n import IdentityInertia, RadialInertia
from hei_n.geometry_n import dist_n

class QuadricPotential(PotentialOracleN):
    def __init__(self, k=1.0):
        self.k = k
        
    def potential(self, x: np.ndarray) -> float:
        # V = 0.5 * k * d(x, origin)^2
        # d = arccosh(x0)
        x0 = x[..., 0]
        # Clip x0 >= 1
        d = np.arccosh(np.maximum(x0, 1.0))
        V = 0.5 * self.k * np.sum(d**2)
        return float(V)
        
    def gradient(self, x: np.ndarray) -> np.ndarray:
        # grad V = k * d * grad d
        x0 = x[..., 0]
        d = np.arccosh(np.maximum(x0, 1.0))
        denom = np.sqrt(x0**2 - 1.0)
        denom = np.maximum(denom, 1e-6)
        
        # grad d wrt x (ambient):
        # d = arccosh(-<x, e0>_M) ? No, arccosh(x0).
        # grad d = (1/sqrt) * grad(x0) = (1/sqrt) * e0.
        # But this is Euclidean Gradient.
        # Wait, if V(x) depends on x0, grad V points in e0 direction.
        # Integrator will project it. 
        # But correct Minkowski Gradient of x0 is -e0?
        # Let's just return Euclidean gradient.
        # grad_euc V = k * d * (1/sqrt) * e0.
        
        grad = np.zeros_like(x)
        # Broadcasting
        factor = self.k * d / denom
        grad[..., 0] = factor
        return grad

def run_simulation(gamma, label):
    N = 1
    dim = 2
    G = np.zeros((N, dim, dim))
    G[0] = np.eye(dim)
    
    # Start at some distance
    # x0 = 2.0 (d ~ 1.31)
    G[0, 0, 0] = 2.0
    G[0, 1, 1] = 2.0
    G[0, 0, 1] = np.sqrt(3.0)
    G[0, 1, 0] = np.sqrt(3.0)
    
    x_init = G[..., 0]
    
    M = np.zeros((N, dim, dim))
    z = 0.0
    state = ContactStateN(G=G, M=M, z=z)
    
    inertia = IdentityInertia()
    oracle = QuadricPotential(k=1.0)
    config = ContactConfigN(dt=0.001, gamma=gamma)
    
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    energies = []
    traj = []
    
    steps = 5000
    for _ in range(steps):
        state = integrator.step(state)
        
        # Calculate Total Energy E = T + V
        T = inertia.kinetic_energy(state.M, state.x)
        V = oracle.potential(state.x)
        E = T + V
        energies.append(E)
        traj.append(state.x[0, 0])
        
    return energies, traj

def main():
    print("Running Thermodynamic Validation...")
    
    # Case A: Conservative
    E_cons, T_cons = run_simulation(gamma=0.0, label="Conservative")
    
    # Case B: Dissipative
    E_diss, T_diss = run_simulation(gamma=0.5, label="Dissipative")
    
    print(f"Conservative Start E: {E_cons[0]:.4f}, End E: {E_cons[-1]:.4f}")
    print(f"Dissipative Start E:  {E_diss[0]:.4f}, End E: {E_diss[-1]:.4f}")
    
    # Validation Logic
    # 1. Conservative: Energy drift should be small
    drift = abs(E_cons[-1] - E_cons[0]) / E_cons[0]
    print(f"Energy Drift (Gamma=0): {drift*100:.2f}%")
    
    if drift > 0.05: # Allow 5% drift for Euler integration
        print("WARNING: High Energy Drift. Integrator might be unstable or Symplectic structure violated.")
    else:
        print("PASS: Energy Conservation holds.")
        
    # 2. Dissipative: Monotonic Decrease
    is_monotonic = all(E_diss[i] >= E_diss[i+1] for i in range(len(E_diss)-1))
    print(f"Monotonic Decrease (Gamma>0): {is_monotonic}")
    
    if not is_monotonic:
        # Check if it's just noise
        diffs = [E_diss[i] - E_diss[i+1] for i in range(len(E_diss)-1)]
        bad_steps = sum(1 for d in diffs if d < -1e-6)
        print(f"Bad steps (Energy Increased): {bad_steps}")
    
    # Visualize
    plt.figure()
    plt.plot(E_cons, label="Gamma=0 (Conservative)")
    plt.plot(E_diss, label="Gamma=0.5 (Dissipative)")
    plt.xlabel("Step")
    plt.ylabel("Total Mechanical Energy E")
    plt.legend()
    plt.title("Constraint 1: Thermodynamic Verification")
    plt.savefig("validation_thermodynamics.png")
    print("Saved plot to validation_thermodynamics.png")

if __name__ == "__main__":
    main()

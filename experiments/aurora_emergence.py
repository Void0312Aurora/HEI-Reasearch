
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN
from hei_n.inertia_n import RadialInertia
from hei_n.potential_n import HarmonicPriorN, PairwisePotentialN, CompositePotentialN
from hei_n.metrics_n import calculate_ultrametricity_score
from hei_n.inertia_n import RadialInertia, IdentityInertia

def kernel_lennard_jones_soft(d):
    val = 2.0 * np.exp(-d / 0.2) - 2.0 * np.exp(-d / 0.8)
    return val

def d_kernel_lennard_jones_soft(d):
    val = -10.0 * np.exp(-d / 0.2) + 2.5 * np.exp(-d / 0.8)
    return val

def plot_disk(x, title="Hyperbolic Disk"):
    """Project from Hyperboloid (x0, x1...) to Poincaré Disk."""
    x0 = x[..., 0]
    x_rest = x[..., 1:]
    y = x_rest / (1.0 + x0)[..., np.newaxis]
    
    plt.figure(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, color='k', fill=False)
    plt.gca().add_patch(circle)
    plt.scatter(y[:, 0], y[:, 1], alpha=0.8, s=10)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title(title)
    plt.axis('equal')
    return plt

def plot_disk_save(x, title, filename):
    plt = plot_disk(x, title)
    plt.savefig(filename)
    plt.close()

def run_emergence(no_contact=False, no_inertia=False, steps=2000):
    print(f"Running Emergence Experiment (NoContact={no_contact}, NoInertia={no_inertia})...")
    N = 50
    dim = 2 
    
    # 1. Setup Potential
    prior = HarmonicPriorN(k=0.5)
    pairwise = PairwisePotentialN(kernel_lennard_jones_soft, d_kernel_lennard_jones_soft)
    oracle = CompositePotentialN([prior, pairwise])
    
    # 2. Setup Physics
    if no_inertia:
        inertia = IdentityInertia()
        inertia_name = "Identity"
    else:
        inertia = RadialInertia(alpha=1.0)
        inertia_name = "Radial"
        
    gamma = 0.0 if no_contact else 0.5
    
    config = ContactConfigN(dt=0.001, gamma=gamma)
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    # 3. Init State
    from hei_n.lie_n import exp_so1n
    G_init = np.zeros((N, dim+1, dim+1)) 
    M_init = np.zeros((N, dim+1, dim+1))
    
    # Random init in disk
    # Reproducible seed
    np.random.seed(42)
    for i in range(N):
        G_init[i] = np.eye(dim+1)
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v) * np.random.uniform(0, 1.0)
        M_boost = np.zeros((dim+1, dim+1))
        M_boost[0, 1:] = v
        M_boost[1:, 0] = v
        G_init[i] = exp_so1n(M_boost[np.newaxis], dt=1.0)[0]
        
    state = ContactStateN(G=G_init, M=M_init, z=0.0)
    
    # Logging
    history = {
        'E': [], 'R': [], 'z': [], 'ultra': []
    }
    
    # 4. Run
    for i in range(steps):
        state = integrator.step(state)
        
        # Log Logic (every 100 steps)
        if i % 100 == 0:
            # E = T + V
            # Note: T is computed by inertia model
            T = inertia.kinetic_energy(state.M, state.x)
            V = oracle.potential(state.x)
            E = T + V
            
            # Rayleigh R = 0.5 * gamma * <xi, I xi> = 0.5 * gamma * 2 * T ? 
            # R = 0.5 * gamma * (xi^T M) = 0.5 * gamma * (xi^T I xi) = gamma * T.
            # R = gamma * T.
            R = gamma * T
            
            history['E'].append(E)
            history['R'].append(R)
            history['z'].append(state.z)
            
            if i % 500 == 0:
                print(f"Step {i}: E={E:.4f}, z={state.z:.2f}")

    # 5. Analysis
    print("Computing Ultrametricity...")
    ultra_score = calculate_ultrametricity_score(state.x)
    print(f"Final Ultrametricity Violation: {ultra_score:.4f} (Lower is better)")
    
    # Plot Energy
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history['E'], label='Energy E')
    # Scale R for visualization? R should be dE/dt / 2.
    plt.plot(history['R'], label='Dissipation R')
    plt.title(f"Energy Evolution (Gamma={gamma}, Inertia={inertia_name})")
    plt.legend()
    plt.savefig(f"energy_{gamma}_{inertia_name}.png")
    
    # Plot Structure
    plot_disk_save(state.x, f"Structure (Ultrametric={ultra_score:.3f})", f"structure_{gamma}_{inertia_name}.png")
    
    return ultra_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--no-inertia", action="store_true")
    args = parser.parse_args()
    
    run_emergence(args.no_contact, args.no_inertia)

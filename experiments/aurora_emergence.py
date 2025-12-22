
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN
from hei_n.inertia_n import RadialInertia
from hei_n.potential_n import HarmonicPriorN, PairwisePotentialN, CompositePotentialN

def kernel_lennard_jones_soft(d):
    # Repulsion (Short range, sigma=0.2)
    # Attraction (Long range, sigma=0.8)
    # V = 5.0 * exp(-d/0.2) - 2.0 * exp(-d/0.8)
    # Repulsion dominant at d=0. Attraction dominant at d=0.5.
    
    val = 2.0 * np.exp(-d / 0.2) - 2.0 * np.exp(-d / 0.8)
    return val

def d_kernel_lennard_jones_soft(d):
    # dV/dd
    # = 2.0 * (-1/0.2) exp + (-2) * (-1/0.8) exp
    # = -10.0 exp_rep + 2.5 exp_att
    
    val = -10.0 * np.exp(-d / 0.2) + 2.5 * np.exp(-d / 0.8)
    return val

def init_big_bang(N, dim=2, sigma=0.1):
    """Random initialization near origin."""
    G = np.zeros((N, dim+1, dim+1))
    for i in range(N):
        G[i] = np.eye(dim+1)
        # Small boost in random direction
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v) * np.random.uniform(0, sigma)
        
        # Build approx boost matrix
        # exp([[0, v^T], [v, 0]])
        # For small v, I + M
        G[i, 0, 1:] = v
        G[i, 1:, 0] = v.T # Check transpose
        # Renormalize to ensure SO(1,n)?
        # Better: use exp mapping
        
    return G

def plot_disk(x, title="Hyperbolic Disk"):
    """Project from Hyperboloid (x0, x1...) to Poincaré Disk."""
    # Disk coords: y = x_rest / (1 + x0)
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

def run_emergence():
    print("Initializing Particle Big Bang...")
    N = 50
    dim = 2 # 2D Hyperbolic Space (Disk)
    
    # 1. Setup Potential
    # Gravity to keep them bound
    prior = HarmonicPriorN(k=0.5)
    
    # Clustering (Repulsion + Attraction)
    # Encourages separation by ~0.5 units
    pairwise = PairwisePotentialN(
        kernel_lennard_jones_soft,
        d_kernel_lennard_jones_soft
    )
    
    oracle = CompositePotentialN([prior, pairwise])
    
    # 2. Setup Physics
    inertia = RadialInertia(alpha=1.0) # Variable Inertia
    config = ContactConfigN(dt=0.001, gamma=0.5) # Damping to settle
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    # 3. Init State
    G_init = np.zeros((N, dim+1, dim+1)) 
    # Use lie_n exp? Or just manual init?
    # Let's do a simple manual init:
    # Random points in disk with r < 0.5
    from hei_n.lie_n import exp_so1n
    
    M_init = np.zeros((N, dim+1, dim+1))
    
    # Create random positions
    # Random boosts
    for i in range(N):
        G_init[i] = np.eye(dim+1)
        # Random boost vector
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v) * np.random.uniform(0, 1.0)
        
        M_boost = np.zeros((dim+1, dim+1))
        M_boost[0, 1:] = v
        M_boost[1:, 0] = v
        G_init[i] = exp_so1n(M_boost[np.newaxis], dt=1.0)[0]
        
    state = ContactStateN(G=G_init, M=M_init, z=0.0)
    
    # 4. Run
    steps = 2000
    print(f" Simulating {steps} steps...")
    
    for i in range(steps):
        state = integrator.step(state)
        if i % 200 == 0:
            print(f"Step {i}, z={state.z:.2f}")
            
    # 5. Visualize
    print("Simulation Complete. Plotting...")
    plt = plot_disk(state.x, title=f"Emergence Result (N={N})")
    plt.savefig("emergence_result.png")
    print("Saved result to emergence_result.png")

if __name__ == "__main__":
    run_emergence()

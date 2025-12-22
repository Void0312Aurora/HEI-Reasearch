"""
Soft-Core Physics Experiment
===========================

Addressing the user's critique: "Engineering compromises (clipping) destroy the dynamical process."
Goal: Solve the "Explosion" via Hamiltonian Regularization (Soft-Core Potential) rather than Integrator Hacks.

Theoretical Change:
- Replace Point Particles (singularity at d=0) with "Soft Spheres".
- Regularized Distance: d_soft = sqrt(d^2 + eps^2)
- Gradient becomes finite at d=0.
- Allows Adaptive Integrator (GroupContactIntegrator) to maintain reasonable dt without clipping.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import dataclasses
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hei.group_integrator import GroupContactIntegrator, GroupIntegratorState, GroupIntegratorConfig
from hei.geometry import uhp_distance_and_grad

# --- Soft Geometry Utilities ---
def soft_uhp_dist_grad(z, c, eps_soft=0.1):
    """
    Regularized Hyperbolic Distance Gradient.
    d_true = acosh(...)
    d_soft = sqrt(d_true^2 + eps_soft^2)
    grad_soft = (d_true / d_soft) * grad_true
    
    Singularity Handling:
    grad_true ~ 1/d_true
    grad_soft ~ (d_true / eps) * (1/d_true) = 1/eps (Finite!)
    """
    d_true, g_true = uhp_distance_and_grad(z, c)
    
    # Avoid numerical noise near zero for d_true
    d_true = np.maximum(d_true, 1e-9)
    
    d_soft = np.sqrt(d_true**2 + eps_soft**2)
    scale = d_true / d_soft
    
    return d_soft, scale * g_true

# --- Reusing Graph Setup ---
class SyntheticTree:
    def __init__(self, depth=3):
        self.depth = depth
        self.graph = nx.balanced_tree(r=2, h=depth)
        self.num_nodes = self.graph.number_of_nodes()
        self.edges = list(self.graph.edges())
        self.d_graph = dict(nx.all_pairs_shortest_path_length(self.graph))
    def get_edges(self): return self.edges

@dataclasses.dataclass
class SoftPotential:
    tree: SyntheticTree
    L_target: float = 1.7
    k_att: float = 2.0
    A_rep: float = 2.0     # Stronger repulsion allowed now?
    sigma_rep: float = 0.5
    eps_soft: float = 0.2  # "Radius" of the particle
    
    def potential(self, Z_uhp, action=None):
        Z = np.asarray(Z_uhp, dtype=np.complex128).ravel()
        n = Z.size
        
        # Inefficient loop for potential eval (diagnostic only)
        V = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d, _ = uhp_distance_and_grad(Z[i], Z[j])
                d_soft = np.sqrt(d**2 + self.eps_soft**2)
                
                if self.tree.graph.has_edge(i, j):
                    V += 0.5 * self.k_att * (d_soft - self.L_target)**2
                else:
                    V += self.A_rep * np.exp(-d_soft / self.sigma_rep)
        return V

    def gradient(self, Z_uhp, action=None):
        # Fully Vectorized Soft Gradient
        Z = np.asarray(Z_uhp, dtype=np.complex128).ravel()
        n = Z.size
        Grad = np.zeros(n, dtype=np.complex128)
        
        for i in range(n):
            zi = Z[i]
            # Batch call for all j
            d_true, g_true = uhp_distance_and_grad(np.array([zi]), Z)
            
            # Mask self (d_true ~ 0)
            mask_self = (np.arange(n) != i)
            d_true = d_true
            
            # Soften
            d_soft = np.sqrt(d_true**2 + self.eps_soft**2)
            grad_scale = d_true / d_soft
            g_soft = g_true * grad_scale
            
            forces = np.zeros(n, dtype=float)
            
            # Edges
            neighbors = list(self.tree.graph.neighbors(i))
            if neighbors:
                nbrs = np.array(neighbors)
                # Force = k * (d - L)
                forces[nbrs] += self.k_att * (d_soft[nbrs] - self.L_target)
                
            # Non-edges
            mask_rep = np.ones(n, dtype=bool)
            mask_rep[i] = False
            if neighbors:
                mask_rep[neighbors] = False
                
            # Repulsion
            raw_f = -(self.A_rep / self.sigma_rep) * np.exp(-d_soft[mask_rep] / self.sigma_rep)
            forces[mask_rep] += raw_f # No clipping needed!
            
            Grad[i] = np.sum(forces * g_soft)
            
        return Grad.reshape(Z_uhp.shape)

def init_big_bang(n_nodes, sigma=0.01):
    re = np.random.normal(0, sigma, n_nodes)
    im = np.random.normal(1, sigma, n_nodes)
    return re + 1j * im

def calculate_distortion(Z, treeObj):
    Z = Z.ravel()
    distortions = []
    n = Z.size
    for i in range(n):
        for j in range(i + 1, n):
            d_hyp, _ = uhp_distance_and_grad(Z[i], Z[j])
            d_g = treeObj.d_graph[i][j]
            if d_g > 0:
                distortions.append(abs(d_hyp - d_g) / d_g)
    return np.mean(distortions)

def run_soft_experiment():
    print("Setting up Soft Physics Experiment...")
    tree = SyntheticTree(depth=3)
    
    # Soft Potential
    pot = SoftPotential(tree=tree, eps_soft=0.2, A_rep=5.0) # Stronger repulsion
    
    z0 = init_big_bang(tree.num_nodes)
    
    # Physics Mode (GroupContactIntegrator)
    # No artificial torque clip (or very high)
    # Adaptive DT should handle the rest
    cfg = GroupIntegratorConfig(
        max_dt=0.1,         # Generous max step
        min_dt=1e-5,
        gamma_scale=0.1,    # Low damping! Let inertia work!
        gamma_mode="constant",
        use_riemannian_inertia=True,
        torque_clip=1000.0, # Effectively infinite
        implicit_potential=True # Robustness
    )
    
    G_init = np.array([np.eye(2) for _ in range(tree.num_nodes)])
    xi_init = np.zeros((tree.num_nodes, 3))
    state = GroupIntegratorState(G=G_init, z0_uhp=z0, xi=xi_init, m=None)
    force_fn = lambda z, a: -pot.gradient(z, a)
    
    integrator = GroupContactIntegrator(force_fn, pot.potential, config=cfg)
    
    steps = 2000
    dist_hist = []
    
    print(f"Running {steps} steps (Physics Mode, Soft Potential)...")
    
    for i in range(steps):
        state = integrator.step(state)
        
        if i % 50 == 0:
             d = calculate_distortion(state.z_uhp, tree)
             dist_hist.append(d)
             # print(f"Step {i}: dt={state.dt_last:.2e}, Dist={d:.4f}")
             
    print(f"Final Dist: {dist_hist[-1]:.4f}")
    
    plt.plot(dist_hist)
    plt.title("Soft Physics Convergence")
    plt.savefig("soft_physics_result.png")

if __name__ == "__main__":
    run_soft_experiment()

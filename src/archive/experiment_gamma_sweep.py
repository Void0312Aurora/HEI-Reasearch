"""
Gamma Sweep Experiment: Investigate HEI damping effects on Tree Reconstruction.

Sweeps gamma_scale in {0.01, 0.1, 1.0, 5.0, 10.0} and tracks:
1. Distortion history
2. Total Energy (T+V) history
3. Final distortion for each gamma
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hei.group_integrator import GroupContactIntegrator, GroupIntegratorState, GroupIntegratorConfig
from hei.geometry import uhp_distance_and_grad

# === Reuse from experiment_tree.py ===
def scalar_uhp_dist(z1, z2):
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    delta = np.abs(z1 - z2)
    delta_conj = np.abs(z1 - np.conj(z2))
    val = delta / np.maximum(delta_conj, 1e-9)
    return 2 * np.arctanh(np.minimum(val, 1.0 - 1e-9))

class SyntheticTree:
    def __init__(self, depth=3):
        self.graph = nx.balanced_tree(r=2, h=depth)
        self.num_nodes = self.graph.number_of_nodes()
        self.d_graph = dict(nx.all_pairs_shortest_path_length(self.graph))

@dataclasses.dataclass
class TreePotential:
    tree: SyntheticTree
    L_target: float = 1.7
    k_att: float = 2.0
    A_rep: float = 1.0
    sigma_rep: float = 0.5
    
    def potential(self, Z_uhp, action=None):
        Z = np.asarray(Z_uhp, dtype=np.complex128).ravel()
        V_tot = 0.0
        for i, j in self.tree.graph.edges():
            d = scalar_uhp_dist(Z[i], Z[j])
            V_tot += 0.5 * self.k_att * (d - self.L_target)**2
        for i in range(Z.size):
            for j in range(i + 1, Z.size):
                if self.tree.graph.has_edge(i, j): continue
                d = scalar_uhp_dist(Z[i], Z[j])
                V_tot += self.A_rep * np.exp(-d / self.sigma_rep)
        return V_tot

    def gradient(self, Z_uhp, action=None):
        Z = np.asarray(Z_uhp, dtype=np.complex128).ravel()
        n = Z.size
        Grad = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            grad_i = 0j
            zi = Z[i]
            for j in range(n):
                if i == j: continue
                zj = Z[j]
                d_mat, g_mat = uhp_distance_and_grad(np.array([zi]), np.array([zj]))
                d_ij = d_mat[0]
                grad_d_zi = g_mat[0]
                if self.tree.graph.has_edge(i, j):
                    force_mag = self.k_att * (d_ij - self.L_target)
                    grad_i += force_mag * grad_d_zi
                else:
                    force_raw = -(self.A_rep / self.sigma_rep) * np.exp(-d_ij / self.sigma_rep)
                    force_mag = np.clip(force_raw, -100.0, 100.0)
                    grad_i += force_mag * grad_d_zi
            Grad[i] = grad_i
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
            d_hyp = scalar_uhp_dist(Z[i], Z[j])
            d_g = treeObj.d_graph[i][j]
            if d_g > 0:
                distortions.append(abs(d_hyp - d_g) / d_g)
    return np.mean(distortions)

def run_hei_with_gamma(pot, z0, gamma_scale, steps=3000):
    """Run HEI simulation and return (distortion_history, energy_history)."""
    cfg = GroupIntegratorConfig(
        max_dt=0.05,
        min_dt=1e-5,
        gamma_scale=gamma_scale,
        gamma_mode="constant",
        use_riemannian_inertia=True
    )
    
    G_init = np.array([np.eye(2) for _ in range(pot.tree.num_nodes)])
    xi_init = np.zeros((pot.tree.num_nodes, 3))
    state = GroupIntegratorState(G=G_init, z0_uhp=z0.copy(), xi=xi_init, m=None)
    force_fn = lambda z, a: -pot.gradient(z, a)
    integrator = GroupContactIntegrator(force_fn, pot.potential, cfg)
    
    dist_hist = []
    energy_hist = []
    
    for i in range(steps):
        state = integrator.step(state)
        if i % 30 == 0:
            dist_hist.append(calculate_distortion(state.z_uhp, pot.tree))
            # Get energy from integrator diagnostics
            V = pot.potential(state.z_uhp)
            # Kinetic energy estimate: 0.5 * ||xi||^2 (simplified)
            T = 0.5 * np.sum(state.xi**2)
            energy_hist.append((T, V, T+V))
            
    return dist_hist, energy_hist, state.z_uhp

def run_gamma_sweep():
    print("Setting up Gamma Sweep Experiment...")
    tree = SyntheticTree(depth=3)
    pot = TreePotential(tree=tree)
    
    np.random.seed(42)
    z0 = init_big_bang(tree.num_nodes, sigma=0.01)
    
    gammas = [0.01, 0.1, 1.0, 5.0, 10.0]
    results = {}
    
    for gamma in gammas:
        print(f"Running HEI with Gamma={gamma}...")
        dist_hist, energy_hist, z_final = run_hei_with_gamma(pot, z0, gamma, steps=3000)
        results[gamma] = {
            'dist': dist_hist,
            'energy': energy_hist,
            'final_dist': dist_hist[-1]
        }
        print(f"  Gamma={gamma}: Final Dist = {dist_hist[-1]:.4f}")
        
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distortion vs Time
    ax = axes[0, 0]
    for gamma in gammas:
        ax.plot(results[gamma]['dist'], label=f'Gamma={gamma}')
    ax.set_title('Distortion vs Iteration')
    ax.set_xlabel('Steps/30')
    ax.set_ylabel('Distortion')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Total Energy (H = T+V)
    ax = axes[0, 1]
    for gamma in gammas:
        H = [e[2] for e in results[gamma]['energy']]
        ax.plot(H, label=f'Gamma={gamma}')
    ax.set_title('Total Energy (H = T + V)')
    ax.set_xlabel('Steps/30')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Kinetic Energy (Oscillation indicator)
    ax = axes[1, 0]
    for gamma in gammas:
        T = [e[0] for e in results[gamma]['energy']]
        ax.plot(T, label=f'Gamma={gamma}')
    ax.set_title('Kinetic Energy (Oscillation Indicator)')
    ax.set_xlabel('Steps/30')
    ax.set_ylabel('T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final Distortion Bar Chart
    ax = axes[1, 1]
    final_dists = [results[g]['final_dist'] for g in gammas]
    bars = ax.bar([str(g) for g in gammas], final_dists, color='steelblue')
    ax.set_title('Final Distortion by Gamma')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Distortion')
    # Highlight best
    best_idx = np.argmin(final_dists)
    bars[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.savefig('gamma_sweep_result.png')
    print("Saved to gamma_sweep_result.png")

if __name__ == "__main__":
    run_gamma_sweep()

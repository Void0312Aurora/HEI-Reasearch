"""
Time-Dependent Damping (Gamma Annealing) Experiment

Tests HEI with scheduled Gamma:
1. Constant (baseline)
2. Linear Increase: Gamma(t) = Gamma_min + (Gamma_max - Gamma_min) * t/T
3. Exponential Increase
4. Cosine Annealing (Reverse)

Goal: Start with low damping for exploration, increase for convergence.
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

# === Gamma Schedules ===
def gamma_constant(step, total_steps, g_min, g_max):
    return g_max  # Use fixed high value

def gamma_linear(step, total_steps, g_min, g_max):
    t = step / total_steps
    return g_min + (g_max - g_min) * t

def gamma_exponential(step, total_steps, g_min, g_max):
    t = step / total_steps
    return g_min * (g_max / g_min) ** t

def gamma_cosine(step, total_steps, g_min, g_max):
    # Reverse Cosine: starts at g_min, ends at g_max
    t = step / total_steps
    return g_max - 0.5 * (g_max - g_min) * (1 + np.cos(np.pi * t))

def run_hei_with_schedule(pot, z0, schedule_fn, g_min, g_max, steps=3000):
    """Run HEI with time-varying gamma."""
    # Start with g_min config, will update gamma each step
    cfg = GroupIntegratorConfig(
        max_dt=0.05,
        min_dt=1e-5,
        gamma_scale=g_min,  # Initial value
        gamma_mode="constant",
        use_riemannian_inertia=True
    )
    
    G_init = np.array([np.eye(2) for _ in range(pot.tree.num_nodes)])
    xi_init = np.zeros((pot.tree.num_nodes, 3))
    state = GroupIntegratorState(G=G_init, z0_uhp=z0.copy(), xi=xi_init, m=None)
    force_fn = lambda z, a: -pot.gradient(z, a)
    integrator = GroupContactIntegrator(force_fn, pot.potential, cfg)
    
    dist_hist = []
    gamma_hist = []
    
    for i in range(steps):
        # Update gamma dynamically
        current_gamma = schedule_fn(i, steps, g_min, g_max)
        integrator.config.gamma_scale = current_gamma
        
        state = integrator.step(state)
        
        if i % 30 == 0:
            dist_hist.append(calculate_distortion(state.z_uhp, pot.tree))
            gamma_hist.append(current_gamma)
            
    return dist_hist, gamma_hist, state.z_uhp

def run_annealing_experiment():
    print("Setting up Gamma Annealing Experiment...")
    tree = SyntheticTree(depth=3)
    pot = TreePotential(tree=tree)
    
    np.random.seed(42)
    z0 = init_big_bang(tree.num_nodes, sigma=0.01)
    
    steps = 5000
    g_min = 0.1
    g_max = 10.0
    
    schedules = {
        'Constant (High)': (gamma_constant, g_max),
        'Linear Increase': (gamma_linear, None),
        'Exponential Increase': (gamma_exponential, None),
        'Cosine Annealing': (gamma_cosine, None),
    }
    
    results = {}
    
    for name, (schedule_fn, override) in schedules.items():
        print(f"Running {name}...")
        if override is not None:
            # Constant schedule uses fixed value
            dist_hist, gamma_hist, z_final = run_hei_with_schedule(
                pot, z0, lambda s,t,gn,gx: override, g_min, g_max, steps
            )
        else:
            dist_hist, gamma_hist, z_final = run_hei_with_schedule(
                pot, z0, schedule_fn, g_min, g_max, steps
            )
        results[name] = {
            'dist': dist_hist,
            'gamma': gamma_hist,
            'final_dist': dist_hist[-1]
        }
        print(f"  {name}: Final Dist = {dist_hist[-1]:.4f}")
        
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Distortion vs Time
    ax = axes[0]
    for name in schedules:
        ax.plot(results[name]['dist'], label=name)
    ax.set_title('Distortion vs Time')
    ax.set_xlabel('Steps/30')
    ax.set_ylabel('Distortion')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gamma Schedule
    ax = axes[1]
    for name in schedules:
        ax.plot(results[name]['gamma'], label=name)
    ax.set_title('Gamma Schedule')
    ax.set_xlabel('Steps/30')
    ax.set_ylabel('Gamma')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final Distortion Bar Chart
    ax = axes[2]
    names = list(schedules.keys())
    final_dists = [results[n]['final_dist'] for n in names]
    bars = ax.bar(range(len(names)), final_dists, color='steelblue')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title('Final Distortion by Schedule')
    ax.set_ylabel('Distortion')
    # Highlight best
    best_idx = np.argmin(final_dists)
    bars[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.savefig('gamma_annealing_result.png')
    print("Saved to gamma_annealing_result.png")

if __name__ == "__main__":
    run_annealing_experiment()

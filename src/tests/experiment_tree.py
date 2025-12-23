
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
import networkx as nx
from typing import Tuple, List, Set

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hei.group_integrator import GroupContactIntegrator, GroupIntegratorState, GroupIntegratorConfig
from hei.geometry import uhp_distance_and_grad

# Helper for scalar dist check (re-defined locally if import fails or for robustness)
def scalar_uhp_dist(z1, z2):
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    delta = np.abs(z1 - z2)
    delta_conj = np.abs(z1 - np.conj(z2))
    # Clip to avoid div by zero or nan
    val = delta / np.maximum(delta_conj, 1e-9)
    return 2 * np.arctanh(np.minimum(val, 1.0 - 1e-9))

class SyntheticTree:
    def __init__(self, depth=3):
        """Create a balanced binary tree."""
        self.depth = depth
        self.graph = nx.balanced_tree(r=2, h=depth)
        self.num_nodes = self.graph.number_of_nodes()
        self.edges = list(self.graph.edges())
        # Cache graph distance
        self.d_graph = dict(nx.all_pairs_shortest_path_length(self.graph))
        
    def get_edges(self) -> List[Tuple[int, int]]:
        return self.edges
        
    def get_non_edges(self) -> List[Tuple[int, int]]:
        non_edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if not self.graph.has_edge(i, j):
                    non_edges.append((i, j))
        return non_edges

@dataclasses.dataclass
class RobustTreePotential:
    tree: SyntheticTree
    L_target: float = 1.7
    k_att: float = 2.0
    A_rep: float = 5.0
    sigma_rep: float = 0.5
    eps_soft: float = 0.2
    huber_delta: float = 1.0 # Scale where behavior switches from Quadratic to Linear
    
    def potential(self, Z_uhp, action=None):
        Z = np.asarray(Z_uhp, dtype=np.complex128).ravel()
        n = Z.size
        
        Z_col = Z[:, np.newaxis]
        Z_row = Z[np.newaxis, :]
        delta = np.abs(Z_col - Z_row)
        delta_conj = np.abs(Z_col - np.conj(Z_row))
        val = delta / np.maximum(delta_conj, 1e-9)
        d_mat = 2 * np.arctanh(np.minimum(val, 1.0 - 1e-9))
        d_soft = np.sqrt(d_mat**2 + self.eps_soft**2)
        
        # 1. Edge Potential (Pseudo-Huber)
        # V(x) = k * delta^2 * (sqrt(1 + (x/delta)^2) - 1)
        # Matches 0.5 * k * x^2 for small x
        edges = np.array(self.tree.get_edges())
        if len(edges) > 0:
            d_edges = d_soft[edges[:, 0], edges[:, 1]]
            err = d_edges - self.L_target
            # Pseudo-Huber
            scale = self.huber_delta
            V_edge = np.sum(self.k_att * (scale**2) * (np.sqrt(1 + (err/scale)**2) - 1))
        else:
            V_edge = 0.0
            
        # 2. Non-Edge Potential (Repulsion)
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        if len(edges) > 0:
            mask[edges[:, 0], edges[:, 1]] = False
            mask[edges[:, 1], edges[:, 0]] = False
            
        d_rep = d_soft[mask]
        V_rep = np.sum(self.A_rep * np.exp(-d_rep / self.sigma_rep)) * 0.5
        
        return V_edge + V_rep

    def gradient(self, Z_uhp, action=None):
        Z = np.asarray(Z_uhp, dtype=np.complex128).ravel()
        n = Z.size
        Grad = np.zeros(n, dtype=np.complex128)
        
        for i in range(n):
            zi = Z[i]
            d_true, g_true = uhp_distance_and_grad(np.array([zi]), Z)
            
            d_soft = np.sqrt(d_true**2 + self.eps_soft**2)
            grad_scale = d_true / d_soft
            g_soft = g_true * grad_scale
            
            forces = np.zeros(n, dtype=float)
            
            # Edge Forces (Pseudo-Huber)
            neighbors = list(self.tree.graph.neighbors(i))
            if neighbors:
                nbrs = np.array(neighbors)
                err = d_soft[nbrs] - self.L_target
                scale = self.huber_delta
                # Force = dV/derr = k * delta^2 * (1/sqrt(...)) * (1/delta) * (2*err/delta) * 0.5 ...
                # V = k d^2 (sqrt(1+u^2) - 1), u = err/d
                # dV/du = k d^2 * u / sqrt(1+u^2)
                # dV/derr = dV/du * 1/d = k d * u / sqrt(1+u^2) = k * d * (err/d) / sqrt = k * err / sqrt(...)
                denom = np.sqrt(1 + (err/scale)**2)
                force_huber = self.k_att * err / denom
                
                forces[nbrs] += force_huber
                
            # Repulsive Forces
            mask_rep = np.ones(n, dtype=bool)
            mask_rep[i] = False
            if neighbors:
                mask_rep[nbrs] = False
                
            raw_f = -(self.A_rep / self.sigma_rep) * np.exp(-d_soft[mask_rep] / self.sigma_rep)
            forces[mask_rep] += raw_f
            
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
            d_hyp = scalar_uhp_dist(Z[i], Z[j])
            d_g = treeObj.d_graph[i][j]
            if d_g > 0:
                distortions.append(abs(d_hyp - d_g) / d_g)
    return np.mean(distortions)

def run_adam_baseline(pot, z0, steps=1000, lr=0.01):
    z = z0.copy().ravel()
    dist_hist = []
    params = np.concatenate([z.real, z.imag])
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    b1=0.9; b2=0.999; eps=1e-8
    
    for i in range(1, steps+1):
        z_curr = params[:z.size] + 1j * params[z.size:]
        grad = pot.gradient(z_curr).ravel()
        g_params = np.concatenate([grad.real, grad.imag])
        
        m = b1*m + (1-b1)*g_params
        v = b2*v + (1-b2)*g_params**2
        m_hat = m/(1-b1**i)
        v_hat = v/(1-b2**i)
        
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        params[z.size:] = np.maximum(params[z.size:], 1e-4)
        
        if i % 50 == 0:
            z_snap = params[:z.size] + 1j * params[z.size:]
            dist_hist.append(calculate_distortion(z_snap, pot.tree))
            
    return dist_hist, params[:z.size] + 1j * params[z.size:]

def run_rsgd_baseline(pot, z0, steps=1000, lr=0.01):
    z = z0.copy().ravel()
    dist_hist = []
    for i in range(steps):
        grad_euc = pot.gradient(z).ravel()
        y = z.imag
        grad_riem = (y**2) * grad_euc
        z = z - lr * grad_riem
        if np.any(z.imag < 1e-4):
            z.imag = np.maximum(z.imag, 1e-4)
        if i % 50 == 0:
            dist_hist.append(calculate_distortion(z, pot.tree))
    return dist_hist, z

def run_tree_experiment():
    print("Setting up Tree Experiment (Physics Mode - Pseudo-Huber)...")
    depth = 3
    tree = SyntheticTree(depth=depth)
    
    pot = RobustTreePotential(tree=tree, L_target=1.7, k_att=2.0, A_rep=5.0, sigma_rep=0.5, eps_soft=0.2, huber_delta=1.0)
    
    np.random.seed(42)
    z0 = init_big_bang(tree.num_nodes, sigma=0.01)
    
    steps = 4000 
    
    # 1. HEI (Physics Mode)
    print("Running HEI (Physics Mode)...")
    cfg = GroupIntegratorConfig(
        max_dt=0.05,
        min_dt=1e-5,
        gamma_scale=0.1,    # Low damping
        gamma_mode="constant",
        use_riemannian_inertia=True,
        torque_clip=1000.0, # High limit
        implicit_potential=True
    )
    
    G_init = np.array([np.eye(2) for _ in range(tree.num_nodes)])
    xi_init = np.zeros((tree.num_nodes, 3))
    state = GroupIntegratorState(G=G_init, z0_uhp=z0, xi=xi_init, m=None)
    force_fn = lambda z, a: -pot.gradient(z, a)
    
    integrator = GroupContactIntegrator(force_fn, pot.potential, config=cfg)
    
    hei_distortion = []
    z_final_hei = None
    
    for i in range(steps):
        state = integrator.step(state)
        if i % 50 == 0:
            d = calculate_distortion(state.z_uhp, tree)
            hei_distortion.append(d)
            
    z_final_hei = state.z_uhp
    print(f"HEI Final Dist: {hei_distortion[-1]:.4f}")
    
    # 2. SGD
    print("Running SGD...")
    z_curr = z0.copy()
    sgd_dist = []
    lr = 0.05
    for i in range(steps):
        grad = pot.gradient(z_curr)
        z_curr = z_curr - lr * grad
        z_curr.imag = np.maximum(z_curr.imag, 1e-4)
        if i % 50 == 0:
            sgd_dist.append(calculate_distortion(z_curr, tree))
    z_final_sgd = z_curr
    print(f"SGD Final Dist: {sgd_dist[-1]:.4f}")
    
    # 3. Adam
    print("Running Adam...")
    adam_dist, z_final_adam = run_adam_baseline(pot, z0, steps=steps, lr=0.01)
    print(f"Adam Final Dist: {adam_dist[-1]:.4f}")
    
    # 4. RSGD
    print("Running RSGD...")
    rsgd_dist, z_final_rsgd = run_rsgd_baseline(pot, z0, steps=steps, lr=0.005)
    print(f"RSGD Final Dist: {rsgd_dist[-1]:.4f}")
    
    # Plotting
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 3, 1)
    plt.plot(hei_distortion, label='HEI', linewidth=2)
    plt.plot(sgd_dist, label='SGD', alpha=0.5)
    plt.plot(adam_dist, label='Adam', alpha=0.5)
    plt.plot(rsgd_dist, label='RSGD', linestyle='--')
    plt.title('Distortion vs Iteration (Refined Sync)')
    plt.legend()
    
    def to_disk(z): return (z - 1j) / (z + 1j)
    
    def plot_tree(idx, title, z_final, dist_val):
        plt.subplot(2, 3, idx)
        d = to_disk(z_final)
        pos = {i: (d[i].real, d[i].imag) for i in range(tree.num_nodes)}
        nx.draw(tree.graph, pos=pos, node_size=50, node_color='b', edge_color='gray', width=0.5)
        plt.gca().add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False, linestyle='--'))
        plt.title(f'{title}\nDist={dist_val:.3f}')
        plt.xlim(-1.1, 1.1); plt.ylim(-1.1, 1.1)

    plot_tree(2, "HEI", z_final_hei, hei_distortion[-1])
    plot_tree(3, "SGD", z_final_sgd, sgd_dist[-1])
    plot_tree(5, "Adam", z_final_adam, adam_dist[-1])
    plot_tree(6, "RSGD", z_final_rsgd, rsgd_dist[-1])
    
    plt.tight_layout()
    plt.savefig('tree_soft_physics.png')

if __name__ == "__main__":
    run_tree_experiment()


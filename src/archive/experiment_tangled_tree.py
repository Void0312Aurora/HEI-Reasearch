
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dataclasses
from typing import Tuple, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hei.group_integrator import GroupContactIntegrator, GroupIntegratorState, GroupIntegratorConfig
from hei.geometry import uhp_distance_and_grad

# --- Setup: Simple 3-Node Tree ---
class TangledTree:
    def __init__(self):
        # 0: Root
        # 1: Left Child
        # 2: Right Child
        self.graph = nx.Graph()
        self.graph.add_nodes_from([0, 1, 2])
        self.graph.add_edge(0, 1)
        self.graph.add_edge(0, 2)
        self.num_nodes = 3
        self.edges = [(0, 1), (0, 2)]
        
    def get_edges(self): return self.edges

# --- Physics: Tanh-Soft Potential ---
@dataclasses.dataclass
class TanhTreePotential:
    tree: TangledTree
    L_target: float = 1.0   # Target bond length
    k_att_eff: float = 10.0 # [UPDATED] Strong spring to confine nodes
    V_max: float = 5.0      
    A_rep: float = 0.5      # [UPDATED] Low barrier to allow crossing
    sigma_rep: float = 0.5
    eps_soft: float = 0.2
    
    def __post_init__(self):
        self.sigma_bond_sq = self.V_max / self.k_att_eff
        
    def potential(self, Z_uhp, action=None):
        return 0.0 # Not used for grad-only simulation

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
            
            # 1. Edge Forces (Tanh)
            neighbors = list(self.tree.graph.neighbors(i))
            if neighbors:
                nbrs = np.array(neighbors)
                d_val = d_soft[nbrs]
                err = d_val - self.L_target
                
                u = (err**2) / (2 * self.sigma_bond_sq)
                tanh_u = np.tanh(u)
                sech2_u = 1.0 - tanh_u**2
                
                force_tanh = self.k_att_eff * err * sech2_u
                forces[nbrs] += force_tanh
                
            # 2. Repulsive Forces
            mask_rep = np.ones(n, dtype=bool)
            mask_rep[i] = False
            if neighbors:
                mask_rep[nbrs] = False
                
            raw_f = -(self.A_rep / self.sigma_rep) * np.exp(-d_soft[mask_rep] / self.sigma_rep)
            forces[mask_rep] += raw_f
            
            Grad[i] = np.sum(forces * g_soft)
            
        return Grad.reshape(Z_uhp.shape)

# --- Initialization: The "Trap" ---
def init_tangled():
    # Root at top-center (i)
    # Child 1 (Left) placed at RIGHT (0.5 + 0.5i)
    # Child 2 (Right) placed at LEFT (-0.5 + 0.5i)
    z = np.zeros(3, dtype=np.complex128)
    z[0] = 0.0 + 1.5j  # Root
    z[1] = 0.5 + 0.5j  # Left Child (Wrong Side)
    z[2] = -0.5 + 0.5j # Right Child (Wrong Side)
    return z

def check_untangled(z):
    # Correct state: Child 1 (Left) should be to the left of Child 2 (Right)
    # In Disk Model or just Real part? 
    # Let's say we want x1 < x2.
    # Initially: x1=0.5, x2=-0.5 (Twisted).
    # Goal: x1 < x2.
    return z[1].real < z[2].real

def run_adam_baseline(pot, z0, steps=1000, lr=0.01):
    z = z0.copy().ravel()
    params = np.concatenate([z.real, z.imag])
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    b1=0.9; b2=0.999; eps=1e-8
    
    z_traj = []
    
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
        
        z_snap = params[:z.size] + 1j * params[z.size:]
        z_traj.append(z_snap.copy())
            
    return z_traj, z_traj[-1]

def run_rsgd_baseline(pot, z0, steps=1000, lr=0.01):
    z = z0.copy().ravel()
    z_traj = []
    
    for i in range(steps):
        grad_euc = pot.gradient(z).ravel()
        y = z.imag
        # Riemannian Gradient: y^2 * grad_euc
        grad_riem = (y**2) * grad_euc
        # Retraction
        z = z - lr * grad_riem
        if np.any(z.imag < 1e-4):
            z.imag = np.maximum(z.imag, 1e-4)
        z_traj.append(z.copy())
        
    return z_traj, z

def run_experiment():
    print("Running Tangled Tree Disentanglement Experiment (Confined Low-Barrier)...")
    tree = TangledTree()
    # Confined Low-Barrier parameters
    pot = TanhTreePotential(tree=tree, L_target=1.0, k_att_eff=10.0, A_rep=0.5) 
    z0 = init_tangled()
    
    steps = 1000
    
    print(f"Initial State: x1={z0[1].real:.2f}, x2={z0[2].real:.2f} (Twisted)")
    
    # 1. HEI (Physics + Annealing)
    print("Running HEI (Physics + Annealing)...")
    cfg = GroupIntegratorConfig(
        max_dt=0.1, min_dt=1e-5, # Allow large steps for momentum
        gamma_scale=0.1, gamma_mode="constant",
        use_riemannian_inertia=True, implicit_potential=True
    )
    
    G_init = np.array([np.eye(2) for _ in range(3)])
    # Initial Kick: Rotational
    xi_init = np.zeros((3, 3))
    kick_strength = 2.0
    xi_init[1, 0] = -kick_strength # vx for node 1
    xi_init[1, 1] = 1.0            # vy for node 1
    
    xi_init[2, 0] = kick_strength  # vx for node 2
    xi_init[2, 1] = -1.0           # vy for node 2
    
    state = GroupIntegratorState(G=G_init, z0_uhp=z0, xi=xi_init, m=None)
    force_fn = lambda z, a: -pot.gradient(z, a)
    integrator = GroupContactIntegrator(force_fn, pot.potential, config=cfg)
    
    gamma_min, gamma_max = 0.01, 5.0 # Very low start friction to allow swing
    
    hei_traj = []
    
    for i in range(steps):
        prog = i/steps
        integrator.config.gamma_scale = gamma_min * (gamma_max/gamma_min)**(prog**2)
        state = integrator.step(state)
        hei_traj.append(state.z_uhp.copy())
        
    z_final_hei = state.z_uhp
    print(f"HEI Final: x1={z_final_hei[1].real:.2f}, x2={z_final_hei[2].real:.2f}")
    print(f"HEI Untangled? {check_untangled(z_final_hei)}")
    
    # 2. SGD
    print("Running SGD...")
    z_sgd = z0.copy()
    lr = 0.05
    sgd_traj = []
    for i in range(steps):
        g = pot.gradient(z_sgd)
        z_sgd -= lr * g
        z_sgd.imag = np.maximum(z_sgd.imag, 1e-4) # Project
        sgd_traj.append(z_sgd.copy())
        
    print(f"SGD Final: x1={z_sgd[1].real:.2f}, x2={z_sgd[2].real:.2f}")
    print(f"SGD Untangled? {check_untangled(z_sgd)}")

    # 3. Adam
    print("Running Adam...")
    adam_traj, z_final_adam = run_adam_baseline(pot, z0, steps=steps, lr=0.01)
    print(f"Adam Final: x1={z_final_adam[1].real:.2f}, x2={z_final_adam[2].real:.2f}")
    print(f"Adam Untangled? {check_untangled(z_final_adam)}")

    # 4. RSGD
    print("Running RSGD...")
    rsgd_traj, z_final_rsgd = run_rsgd_baseline(pot, z0, steps=steps, lr=0.01)
    print(f"RSGD Final: x1={z_final_rsgd[1].real:.2f}, x2={z_final_rsgd[2].real:.2f}")
    print(f"RSGD Untangled? {check_untangled(z_final_rsgd)}")
    
    # Plot Trajectories
    plt.figure(figsize=(16, 12))
    
    def to_disk(z): return (z - 1j) / (z + 1j)
    
    def plot_traj(idx, title, traj, z_final):
        plt.subplot(2, 2, idx)
        d0 = to_disk(z0)
        plt.scatter(d0.real, d0.imag, c='k', marker='x', label='Start')
        
        d_end = to_disk(z_final)
        plt.scatter(d_end.real, d_end.imag, c=['r', 'g', 'b'], s=100, label='End')
        
        traj_disk = np.array([to_disk(t) for t in traj])
        for n in range(3):
            plt.plot(traj_disk[:, n].real, traj_disk[:, n].imag, alpha=0.5)
            
        plt.title(f"{title}\nUntangled: {check_untangled(z_final)}")
        plt.xlim(-1, 1); plt.ylim(-1, 1)
        plt.gca().add_artist(plt.Circle((0,0), 1, fill=False))

    plot_traj(1, "HEI", hei_traj, z_final_hei)
    plot_traj(2, "SGD", sgd_traj, z_sgd)
    plot_traj(3, "Adam", adam_traj, z_final_adam)
    plot_traj(4, "RSGD", rsgd_traj, z_final_rsgd)
    
    plt.savefig('tangled_result_baselines.png')

if __name__ == "__main__":
    run_experiment()

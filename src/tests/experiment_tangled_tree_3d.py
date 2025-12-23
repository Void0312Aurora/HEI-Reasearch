
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dataclasses

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Fix import path since we are running from src/
# Assuming folder structure:
# HEI/src/experiment_tangled_tree_3d.py
# HEI/src/hei_n/

try:
    from hei_n.lie_n import minkowski_metric
    from hei_n.geometry_n import dist_n, dist_grad_n, minkowski_inner
    from hei_n.integrator_n import GroupIntegratorN, IntegratorStateN, IntegratorConfigN
except ImportError:
    # Fallback if running from HEI root
    from src.hei_n.lie_n import minkowski_metric
    from src.hei_n.geometry_n import dist_n, dist_grad_n, minkowski_inner
    from src.hei_n.integrator_n import GroupIntegratorN, IntegratorStateN, IntegratorConfigN

# --- Tangled Tree Structure ---
class TangledTree:
    def __init__(self):
        self.edges = [(0, 1), (0, 2)]
        self.num_nodes = 3
        # 0: Root
        # 1: Left Child
        # 2: Right Child

# --- 3D Tanh Potential ---
@dataclasses.dataclass
class TanhPotentialN:
    tree: TangledTree
    L_target: float = 1.5
    k_att: float = 2.0
    A_rep: float = 2.0 # High barrier
    sigma_rep: float = 0.5
    # For Tanh
    V_max: float = 5.0
    
    def __post_init__(self):
        self.sigma_bond_sq = self.V_max / self.k_att
        
    def gradient(self, x_world: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean Gradient of Potential V(x) in R^{n+1}.
        x_world: (N, dim+1)
        """
        N, dim_p1 = x_world.shape
        grad = np.zeros_like(x_world)
        
        # We need distances between all pairs
        # To optimize, we do loop for N=3
        
        for i in range(N):
            for j in range(N):
                if i == j: continue
                
                # Check bond
                is_connected = False
                if (i, j) in self.tree.edges or (j, i) in self.tree.edges:
                    is_connected = True
                
                # Compute distance and gradient
                xi = x_world[i]
                xj = x_world[j]
                
                # Using geometry_n utils
                # dist_n returns scalar (or batched)
                # But we need Gradient w.r.t xi.
                # dist_grad_n(xi, xj) returns (d, grad_d_xi)
                
                d_val, g_d_xi = dist_grad_n(xi[np.newaxis, :], xj[np.newaxis, :])
                d_val = d_val[0]
                g_d_xi = g_d_xi[0]
                
                # Force Magnitude
                force_mag = 0.0
                
                if is_connected:
                    # Tanh Spring
                    err = d_val - self.L_target
                    u = (err**2) / (2 * self.sigma_bond_sq)
                    # For numerical stability of tanh
                    u = min(u, 20.0) 
                    tanh_u = np.tanh(u)
                    sech2_u = 1.0 - tanh_u**2
                    
                    # Force = dV/dd = k * err * sech^2
                    force_mag += self.k_att * err * sech2_u
                    
                else:
                    # Repulsion
                    # V = A * exp(-d/sigma)
                    # dV/dd = -A/sigma * exp...
                    
                    # Optimization: Cutoff
                    if d_val < 3.0:
                        force_mag += -(self.A_rep / self.sigma_rep) * np.exp(-d_val / self.sigma_rep)
                
                # Accumulate Gradient: grad V = (dV/dd) * grad d
                grad[i] += force_mag * g_d_xi
                
        return grad

# --- Initialization in H^3 ---
def init_tangled_3d():
    # H^3 coords (t, x, y, z)
    # -t^2 + x^2 + y^2 + z^2 = -1 => t = sqrt(1 + r^2)
    
    def to_h3(x, y, z):
        r2 = x*x + y*y + z*z
        t = np.sqrt(1 + r2)
        return np.array([t, x, y, z])
    
    # Root at top
    # Let's put Root at roughly (t, 0, 1.5, 0) (Shifted up in Y)
    # 2D plane is X-Y.
    # Root: Y=1.5
    root = to_h3(0, 1.5, 0)
    
    # Swapped Children in X axis
    # Correct: Left Child at X=-0.5, Right Child at X=+0.5
    # Tangled: Left child at X=+0.5, Right child at X=-0.5
    # Y level: 0.5
    c1 = to_h3(0.5, 0.5, 0)
    c2 = to_h3(-0.5, 0.5, 0)
    
    return np.stack([root, c1, c2])

def run_experiment_3d():
    print("Running 3D Tangled Tree Experiment (HEI-N)...")
    
    tree = TangledTree()
    # High Barrier A_rep=2.0
    # Strong Confinement k_att=20.0 to prevent expansion
    pot = TanhPotentialN(tree=tree, A_rep=2.0, k_att=20.0)
    
    x0 = init_tangled_3d()
    print("Initial Positions (X coords):")
    print(f"C1 (Left Child): {x0[1, 1]:.2f}")
    print(f"C2 (Right Child): {x0[2, 1]:.2f}")
    
    # Init Group Elements (Boosts to positions)
    # For H^n, G is (N, n+1, n+1). 
    # Just need G @ e0 = x0.
    # Construct pure boosts.
    # e0 = (1, 0, 0, 0)
    # Boost map x -> G.
    # Formula: B(v) = I + ... geometry.
    # Let's cheat: we just need G such that G[..., 0] = x0.
    # We can use Gram-Schmidt or existing helpers?
    # HEI-N assumes we have G.
    # Let's derive a simple boost.
    # If x = (t, r_vec), v = r_vec / (t+1)?
    # For now, let's assume we start with G=Identity but x is set?
    # No, Integrator updates G. x comes from G.
    # We need a function `x_to_G(x)`.
    # Let's do a quick hack in initialization:
    # Use "integrator.step" logic? No.
    # Let's implement construct_boost(x) in script.
    
    def vector_to_boost(x):
        # Maps e0=(1,0,0,0) to x=(t, x, y, z)
        # B = I + (ln t) * K... ?
        # Standard Lorentz Boost to vector v/c.
        # beta = vec/t.
        # B = ...
        # Simplified:
        # Rotation taking e0 to x? No, rotation preserves t. Boost changes t.
        # We need B such that B e0 = x.
        # u = x + e0. 
        # Householder reflection in Minkowski?
        # Let's use the explicit formula:
        # B = I + ...
        # Let v = x_space / t. No.
        
        # Let's define the boost along direction n = x_space / |x_space|.
        # rapidity eta = arccosh(t).
        # B = ...
        
        # Helper:
        t = x[0]
        r_vec = x[1:]
        r = np.linalg.norm(r_vec)
        if r < 1e-9: return np.eye(4)
        
        n = r_vec / r
        # Boost matrix
        # [ cosh  sinh*n^T ]
        # [ sinh*n  I + (cosh-1)nn^T ]
        
        B = np.eye(4)
        B[0, 0] = t
        B[0, 1:] = r_vec
        B[1:, 0] = r_vec
        B[1:, 1:] = np.eye(3) + ((t - 1) / r**2) * np.outer(r_vec, r_vec)
        return B
        
    G_init = np.stack([vector_to_boost(xi) for xi in x0])
    
    # 3D Momentum Kick (In Z direction)
    # We want C1 and C2 to spiral around each other.
    # C1 (at X>0): Kick in +Z
    # C2 (at X<0): Kick in -Z
    # They should rotate in X-Z plane while being pulled to swap X.
    
    M_init = np.zeros_like(G_init)
    
    # Momentum is in Algebra so(1, 3).
    # [[0, u^T], [u, W]]
    # We want velocity in Z.
    # Spatial part: index 3 (0=t, 1=x, 2=y, 3=z).
    # Boost generator in Z direction:
    # M[0, 3] = v_z, M[3, 0] = v_z
    
    kick = 1.0 # Moderate kick
    
    # C1 (Index 1) -> +Z
    M_init[1, 0, 3] = kick
    M_init[1, 3, 0] = kick
    
    # C2 (Index 2) -> -Z
    M_init[2, 0, 3] = -kick
    M_init[2, 3, 0] = -kick
    
    # And maybe some attraction in X to swap?
    # Potential handles attraction.
    # We also give them initial X velocity towards correct side?
    # C1 (at +x) needs to go -x.
    # M_init[1, 0, 1] = -1.0
    # M_init[1, 1, 0] = -1.0
    # C2 (at -x) needs to go +x.
    # M_init[2, 0, 1] = 1.0
    # M_init[2, 1, 0] = 1.0
    
    state = IntegratorStateN(G=G_init, M=M_init)
    cfg = IntegratorConfigN(dt=0.01, gamma=0.1) # Const damping
    integrator = GroupIntegratorN(pot.gradient, cfg)
    
    traj = []
    
    print("Simulating...")
    for i in range(2000): # 2000 steps
        state = integrator.step(state)
        if i % 10 == 0:
            traj.append(state.x.copy())
            
    traj = np.array(traj) # (T, N, 4)
    x_final = traj[-1]
    
    print("Final Positions (X coords):")
    x1_f = x_final[1, 1]
    x2_f = x_final[2, 1]
    print(f"C1: {x1_f:.2f}")
    print(f"C2: {x2_f:.2f}")
    
    untangled = x1_f < x2_f
    print(f"Untangled? {untangled}")
    
    # --- Baselines ---
    
    def project_to_h3(x):
        """Project x in R^4 to H^3: t = sqrt(1 + r^2)"""
        # x is (N, 4)
        r_vec = x[..., 1:]
        r2 = np.sum(r_vec**2, axis=-1)
        t = np.sqrt(1.0 + r2)
        x_proj = np.empty_like(x)
        x_proj[..., 0] = t
        x_proj[..., 1:] = r_vec
        return x_proj
        
    def construct_kick_vector(N):
        # Create a kick vector in R^4 matching HEI's logic
        # HEI used Matrix Momentum M in so(1,3).
        # M is a boost generator. M[0, 3] = v_z, M[3, 0] = v_z.
        # Velocity in R^4: v = M * x0.
        # For initial state (roughly X-axis), x0 approx (1, x, 0, 0).
        # M * x0:
        # v_0 = M_03 * x0_3 = v_z * 0 = 0.
        # v_3 = M_30 * x0_0 = v_z * t.
        # So effective velocity is in Z-direction with magnitude v_z * t.
        
        kick_vec = np.zeros((N, 4))
        # C1 (idx 1): +Z
        kick_vec[1, 3] = 1.0 * x0[1, 0] # Scale by t roughly
        # C2 (idx 2): -Z
        kick_vec[2, 3] = -1.0 * x0[2, 0]
        return kick_vec

    def run_sgd_3d(pot, x0, steps=2000, lr=0.01):
        x = x0.copy()
        traj = []
        for i in range(steps):
            grad = pot.gradient(x)
            x = x - lr * grad
            x = project_to_h3(x)
            if i % 10 == 0: traj.append(x.copy())
        return np.array(traj), x

    def run_sgdm_3d(pot, x0, kick_vec, steps=2000, lr=0.01, momentum=0.9):
        """SGD with Momentum (fair comparison)."""
        x = x0.copy()
        v = kick_vec.copy() 
        # Note: kick_vec is our initial velocity guess
        
        traj = []
        for i in range(steps):
            grad = pot.gradient(x)
            
            # Standard Polyak momentum: v = mu * v - lr * grad
            v = momentum * v - lr * grad
            x = x + v
            x = project_to_h3(x)
            if i % 10 == 0: traj.append(x.copy())
        return np.array(traj), x

    def run_adam_3d(pot, x0, kick_vec, steps=2000, lr=0.01):
        x = x0.copy()
        # Initialize first moment with Kick!
        # Adam update: m = b1 m + (1-b1) g
        # If we want initial velocity effect, we can set m such that step ~ kick.
        # step ~ lr * m. So m ~ kick / lr.
        # Or more simply, just treat it as momentum.
        # Note: Adam's 'm' is closer to average gradient. Gradient is opposite to velocity.
        # So m_init = - kick ?
        
        m = -kick_vec.copy() 
        v = np.zeros_like(x)
        b1=0.9; b2=0.999; eps=1e-8
        
        traj = []
        
        for i in range(1, steps+1):
            grad = pot.gradient(x)
            m = b1*m + (1-b1)*grad
            v = b2*v + (1-b2)*grad**2
            m_hat = m/(1-b1**i)
            v_hat = v/(1-b2**i)
            
            # Update
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            x = project_to_h3(x)
            if i % 10 == 0: traj.append(x.copy())
            
        return np.array(traj), x

    def run_rsgd_3d(pot, x0, steps=2000, lr=0.01):
        x = x0.copy()
        traj = []
        # RSGD usually stateless, but we could add momentum (Riem Momentum).
        # Let's keep it stateless as standard RSGD, since user asked for fairness 
        # primarily via momentum injection, which RSGD doesn't natively have without extension.
        # Actually, let's skip "Kick" for RSGD unless we implement R-Momentum.
        # We will test SGD+M and Adam+M as the primary "Fair" comparators.
        
        for i in range(steps):
            g_euc = pot.gradient(x)
            g_mink = g_euc.copy(); g_mink[..., 0] *= -1.0
            inner = minkowski_inner(x, g_mink)
            grad_R = g_mink + inner[..., np.newaxis] * x
            
            v = -lr * grad_R
            x_new = x + v
            x = project_to_h3(x_new) # Retraction
            
            if i % 10 == 0: traj.append(x.copy())
        return np.array(traj), x

    print("Running SGD (Baseline)...")
    sgd_traj, sgd_final = run_sgd_3d(pot, x0, steps=2000, lr=0.05)
    print(f"SGD Untangled? {sgd_final[1, 1] < sgd_final[2, 1]}")
    
    kick_vec = construct_kick_vector(3)
    
    print("Running SGD+Momentum (Fair Kick)...")
    # High momentum factor to sustain the kick
    sgdm_traj, sgdm_final = run_sgdm_3d(pot, x0, kick_vec, steps=2000, lr=0.01, momentum=0.95)
    print(f"SGD+M Untangled? {sgdm_final[1, 1] < sgdm_final[2, 1]}")
    
    print("Running Adam (Fair Kick)...")
    adam_traj, adam_final = run_adam_3d(pot, x0, kick_vec, steps=2000, lr=0.01)
    print(f"Adam Untangled? {adam_final[1, 1] < adam_final[2, 1]}")
    
    print("Running RSGD (Standard)...")
    rsgd_traj, rsgd_final = run_rsgd_3d(pot, x0, steps=2000, lr=0.01)
    print(f"RSGD Untangled? {rsgd_final[1, 1] < rsgd_final[2, 1]}")

    # Plot Trajectories
    plt.figure(figsize=(16, 12))
    
    def plot_traj(idx, title, traj, x_final):
        # Plot X-Z Projection
        plt.subplot(2, 3, idx)
        plt.plot(traj[:, 1, 1], traj[:, 1, 3], 'r-', label='C1', alpha=0.6)
        plt.plot(traj[:, 2, 1], traj[:, 2, 3], 'b-', label='C2', alpha=0.6)
        
        # Start/End
        plt.scatter(traj[0, 1, 1], traj[0, 1, 3], c='k', marker='x')
        plt.scatter(traj[-1, 1, 1], traj[-1, 1, 3], c='r', s=50)
        plt.scatter(traj[-1, 2, 1], traj[-1, 2, 3], c='b', s=50)
        
        unt = (x_final[1, 1] < x_final[2, 1])
        plt.title(f"{title}\nUntangled: {unt}")
        plt.xlabel("X"); plt.ylabel("Z")
        plt.grid(True)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

    plot_traj(1, "HEI-N (Hamiltonian)", traj, x_final)
    plot_traj(2, "SGD (Naive)", sgd_traj, sgd_final)
    plot_traj(4, "SGD+Momentum (Kick)", sgdm_traj, sgdm_final)
    plot_traj(5, "Adam (Kick)", adam_traj, adam_final)
    plot_traj(6, "RSGD (Naive)", rsgd_traj, rsgd_final)
    
    plt.tight_layout()
    plt.savefig('tangled_3d_fair.png')

if __name__ == "__main__":
    run_experiment_3d()

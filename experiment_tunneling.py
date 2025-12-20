
import numpy as np
import matplotlib.pyplot as plt
from hei.data_potential import DataDrivenStressPotential
from hei.simulation import SimulationConfig, run_simulation_group
from hei.geometry import uhp_distance_and_grad, cayley_uhp_to_disk, cayley_disk_to_uhp, disk_to_hyperboloid
from hei.metrics import poincare_distance_disk

def build_tree_distances(depth=3, branching=2):
    """
    Construct a target distance matrix for a perfect tree.
    Distance between connected nodes = 2.0.
    """
    total_nodes = (branching ** depth - 1) // (branching - 1) if branching > 1 else depth
    # Actually simple manual construction for depth=3, branching=2 (7 nodes)
    # Node 0: Root
    # Node 1, 2: Children of 0
    # Node 3, 4: Children of 1
    # Node 5, 6: Children of 2
    
    # Adjacency
    adj = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5, 6],
        3: [1], 4: [1],
        5: [2], 6: [2]
    }
    
    # All pairs shortest path (Graph Distance * Scale)
    n = 7
    dists = np.zeros((n, n))
    scale = 2.0
    
    import networkx as nx
    G = nx.Graph(adj)
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    for i in range(n):
        for j in range(n):
            dists[i, j] = path_lengths[i][j] * scale
            
    return dists

def initialize_twisted_trap():
    """
    Initialize points in a 'twisted' configuration.
    Root at center.
    Left branch (1, 3, 4) placed on Right side.
    Right branch (2, 5, 6) placed on Left side.
    
    To fix this, they must pass through the center (Root), which is crowded/high energy.
    """
    n = 7
    # Polar coordinates on disk
    # 0: center
    # 1 (Left Child): theta=0 (Right) -> Trap
    # 2 (Right Child): theta=pi (Left) -> Trap
    
    r_inner = 0.5
    r_outer = 0.8
    
    z_disk = np.zeros(n, dtype=np.complex128)
    z_disk[0] = 0.0 # Root
    
    # 1 (should be Left/Pi, puts at Right/0)
    z_disk[1] = r_inner * np.exp(1j * 0.0)
    # 2 (should be Right/0, puts at Left/Pi)
    z_disk[2] = r_inner * np.exp(1j * np.pi)
    
    # Children of 1 (3, 4). Put them near 1 (Right side)
    z_disk[3] = r_outer * np.exp(1j * (-0.2))
    z_disk[4] = r_outer * np.exp(1j * (0.2))
    
    # Children of 2 (5, 6). Put them near 2 (Left side)
    z_disk[5] = r_outer * np.exp(1j * (np.pi - 0.2))
    z_disk[6] = r_outer * np.exp(1j * (np.pi + 0.2))
    
    # Map to UHP
    z_uhp = cayley_disk_to_uhp(z_disk)
    return z_uhp

def run_sgd_baseline(pot, z_init, steps=2000, lr=1e-3):
    """
    Run basic Riemannian Gradient Descent.
    update: z <- map_z(-lr * grad_R)
    grad_R = (Im z)^2 * grad_E (Euclidean Gradient)
    """
    z_curr = z_init.copy()
    energies = []
    
    print("Running SGD Baseline...")
    for i in range(steps):
        # Euclidean Gradient dV/dz
        # pot.dV_dz returns Euclidean gradient (treating x, y as flat coords)
        # Actually it returns complex number d/dx + i d/dy which is vector.
        grad_eucl = pot.dV_dz(z_curr)
        
        # Riemannian Gradient scaling: 1/y^2 metric -> y^2 factor for gradient vector
        # g^{ij} d_j V
        y = np.imag(z_curr)
        grad_riem = (y**2) * grad_eucl
        
        # Update (Retraction: z + v is fine for small steps in UHP)
        # Or better: exponential map? For comparison, approximate is fine.
        # But UHP is just half plane, simple step is:
        # z_new = z - lr * grad_riem. 
        # CAREFUL: standard gradient descent uses NEGATIVE gradient.
        # pot.dV_dz returns gradient.
        
        # Wait, simple update z = z - lr * grad_riem might leave UHP if step huge.
        # Use simple clipping.
        
        step = -lr * grad_riem
        z_curr = z_curr + step
        
        # Clamp im
        z_curr = np.real(z_curr) + 1j * np.maximum(np.imag(z_curr), 1e-6)
        
        e = pot.potential(z_curr)
        energies.append(e)
        
    return energies, z_curr

def run_hei_experiment():
    # 1. Setup
    dists = build_tree_distances()
    # Increase stiffness to drive faster acceleration against inertia
    pot = DataDrivenStressPotential(target_dists=dists, stiffness=20.0)
    z_init = initialize_twisted_trap()
    
    energy_start = pot.potential(z_init)
    print(f"Initial Stress Energy: {energy_start:.4f}")
    
    # 2. Run HEI
    
    # Configs
    cfg = SimulationConfig(
        steps=10000, # More steps for momentum to build up
        n_points=7,
        max_dt=0.01,
        log_interval=10,
        use_parent_bias=False
    )
    
    from hei.group_integrator import GroupContactIntegrator, GroupIntegratorState, GroupIntegratorConfig
    
    int_cfg = GroupIntegratorConfig(
        max_dt=cfg.max_dt,
        gamma_scale=0.5, # Moderate damping
        gamma_mode="constant",
        torque_clip=100.0
    )
    
    # Initialize simulation with z_init
    # We need to construct `GroupIntegratorState` manually or modify run_simulation_group?
    # `run_simulation_group` initializes randomly.
    # We should hack it or copy code. 
    # Actually `run_simulation_group` does not accept z_init.
    # I will modify `run_simulation_group` to accept `z_init` or just use the class directly.
    # Modifying simulation.py is cleaner for reuse.
    
    # Let's Modify simulation.py first? No, sticking to local script if possible.
    # But `run_simulation_group` is convenient.
    # I'll update simulation.py to check for `initial_state` in kwargs?
    # Or just use `rng` to seed? No, need specific trap.
    
    from hei.group_integrator import create_initial_group_state
    
    # Initialize state (will automatically calc Inertia and Momentum)
    # xi is set to zero (static start)
    state = create_initial_group_state(
        z_uhp=z_init,
        xi=np.zeros((7, 3))
    )
    
    # Create force function (Force = -Gradient)
    def force_fn(z, action):
        return -pot.dV_dz(z, action)
        
    integrator = GroupContactIntegrator(force_fn, pot.potential, config=int_cfg)
    
    hei_energies = []
    
    print("Running HEI Tunneling...")
    for i in range(cfg.steps):
        state = integrator.step(state)
        if i % 10 == 0:
            e = pot.potential(state.z_uhp)
            hei_energies.append(e)
            
    # 3. Run SGD
    sgd_energies, z_sgd = run_sgd_baseline(pot, z_init, steps=cfg.steps, lr=0.005)
    
    # 4. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(sgd_energies, label='SGD (Gradient Descent)')
    plt.plot(np.linspace(0, cfg.steps, len(hei_energies)), hei_energies, label='HEI (Hamiltonian)')
    plt.title('Tunneling Experiment: Escaping False Hierarchy Trap')
    plt.xlabel('Steps')
    plt.ylabel('Stress Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('tunneling_result.png')
    print("Optimization graph saved to tunneling_result.png")
    
    final_hei = hei_energies[-1]
    final_sgd = sgd_energies[-1]
    print(f"Final Energy - HEI: {final_hei:.4f}")
    print(f"Final Energy - SGD: {final_sgd:.4f}")
    
    if final_hei < final_sgd * 0.5:
        print("SUCCESS: HEI found a much better minimum!")
    elif final_hei < final_sgd:
        print("PARTIAL SUCCESS: HEI is better but maybe didn't fully tunnel.")
    else:
        print("FAILURE: HEI did not beat SGD.")

if __name__ == "__main__":
    run_hei_experiment()

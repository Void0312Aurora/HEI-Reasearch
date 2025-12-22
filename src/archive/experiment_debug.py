
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hei.group_integrator import GroupContactIntegrator, GroupIntegratorState, GroupIntegratorConfig
from hei.inertia import riemannian_inertia
from hei.data_potential import DataDrivenStressPotential
from hei.geometry import uhp_to_hyperboloid
from experiment_tunneling import build_tree_distances, initialize_twisted_trap

def run_debug_trial():
    # Setup
    dists = build_tree_distances()
    # K=1.0 (Soft), Gamma=0.01 (Underdamped -> Accidental Energy Gain?)
    pot = DataDrivenStressPotential(target_dists=dists, stiffness=1.0)
    z_init = initialize_twisted_trap()
    
    cfg = GroupIntegratorConfig(
        max_dt=2e-2,
        min_dt=1e-6,
        gamma_scale=0.01,    # VERY LOW DAMPING
        gamma_mode="constant",
        torque_clip=10000.0,
        xi_clip=1000.0,
        use_riemannian_inertia=True,
        verbose=True,
        diagnostic_interval=10 # Frequent logging
    )
    
    # Init
    state = GroupIntegratorState(
        G=np.eye(2),
        z0_uhp=z_init,
        xi=np.zeros((z_init.shape[0], 3)),
        m=None
    )
    
    force_fn = lambda z, a: -pot.gradient(z, a)
    integrator = GroupContactIntegrator(force_fn, pot.potential, config=cfg)
    
    history_T = []
    history_V = []
    history_H = []
    
    print("Starting Debug Simulation: K=1.0, Gamma=0.01")
    
    for i in range(5000): # Should be enough to see explosion
        state = integrator.step(state)
        
        # Manual Energy Check
        h = state.h
        I = riemannian_inertia(h)
        
        # Kinetic T = 0.5 * xi^T I xi
        if state.xi.ndim == 2:
            # Batch
            T_sum = 0
            for k in range(state.xi.shape[0]):
                xi_k = state.xi[k]
                I_k = I[k]
                T_sum += 0.5 * xi_k @ (I_k @ xi_k)
            T = T_sum
        else:
            T = 0.5 * state.xi @ (I @ state.xi)
            
        V = pot.potential(state.z_uhp)
        H = T + V
        
        history_T.append(T)
        history_V.append(V)
        history_H.append(H)
        
        if i % 10 == 0:
            print(f"Step {i:4d}: T={T:.4f}, V={V:.4f}, H={H:.4f}, |xi|={np.linalg.norm(state.xi):.2f}")
            
        if H > 2 * (history_H[0] + 10.0): # Explosion check
            print("EXPLOSION DETECTED!")
            break
            
    # Plot
    plt.figure()
    plt.plot(history_T, label='Kinetic T')
    plt.plot(history_V, label='Potential V')
    plt.plot(history_H, label='Total H', linestyle='--')
    plt.legend()
    plt.title('Energy Explosion Debug')
    plt.savefig('debug_explosion.png')

if __name__ == "__main__":
    run_debug_trial()

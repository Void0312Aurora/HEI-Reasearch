
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hei.group_integrator import GroupContactIntegrator, GroupIntegratorState, GroupIntegratorConfig
from hei.inertia import riemannian_inertia
from hei.geometry import uhp_distance_and_grad, uhp_to_hyperboloid

def uhp_dist(z1, z2):
    # Scalar UHP distance
    z1 = np.asarray(z1)
    z2 = np.asarray(z2)
    delta = np.abs(z1 - z2)
    delta_conj = np.abs(z1 - np.conj(z2))
    return 2 * np.arctanh(delta / delta_conj)

@dataclasses.dataclass
class SyntheticDoubleWellPotential:
    """
    Synthetic Double Well Potential on Hyperboloid.
    
    V(z) = - A_g * exp(-d(z, mu_g)^2 / 2sigma^2) - A_l * exp(-d(z, mu_l)^2 / 2sigma^2)
    
    A_g: Depth of Global Well (e.g. 50)
    mu_g: Center of Global Well (e.g. 0 + 1j)
    A_l: Depth of Local Well (e.g. 30)
    mu_l: Center of Local Well (e.g. 0 + 3j)
    sigma: Width of wells
    """
    mu_g: complex = 1j      # Global Min at Origin (UHP)
    mu_l: complex = 3j      # Local Min at y=3
    A_g: float = 10.0       # Global Depth
    A_l: float = 5.0        # Local Depth
    sigma: float = 0.5
    
    def potential(self, z_uhp, action=None):
        z = np.asarray(z_uhp, dtype=np.complex128).ravel()
        # Single particle limit for now
        d_g = uhp_dist(z, self.mu_g)
        d_l = uhp_dist(z, self.mu_l)
        
        V_g = -self.A_g * np.exp(-d_g**2 / (2 * self.sigma**2))
        V_l = -self.A_l * np.exp(-d_l**2 / (2 * self.sigma**2))
        
        # Shift so min V is approx -A_g. Max V (far away) is 0.
        # Add offset to make it look like a well in a flat plain?
        # Or just return negative energy?
        # Let's return V_total + offset so min is 0? 
        # No, physics is fine with negative.
        return np.sum(V_g + V_l)

    def gradient(self, z_uhp, action=None):
        # Numerical gradient or analytical? 
        # Analytical is better. 
        # V(d) = -A exp(-d^2/2s^2)
        # dV/dz = dV/dd * dd/dz
        # dV/dd = -A exp(...) * (-2d/2s^2) = A * d/s^2 * exp(...)
        # dd/dz: Need gradient of distance.
        
        # uhp_distance_and_grad returns distance matrix and grad matrix
        
        z = np.asarray(z_uhp, dtype=np.complex128).ravel()
        n = z.size
        
        # Gradient accumulator
        grad_total = np.zeros(n, dtype=np.complex128)
        
        for k in range(n):
            zk = z[k]
            
            # Global Well Term
            dg_val, dg_grad = self._dist_grad_helper(zk, self.mu_g)
            decay_g = np.exp(-dg_val**2 / (2 * self.sigma**2))
            # Chain rule: dV/dz = (A/sigma^2 * d) * exp * dd/dz
            # Wait: V = -A exp(-d^2/2s^2)
            # dV/dd = -A * exp * (-d/s^2) = (A * d / s^2) * exp
            factor_g = (self.A_g * dg_val / self.sigma**2) * decay_g
            grad_g = factor_g * dg_grad
            
            # Local Well Term
            dl_val, dl_grad = self._dist_grad_helper(zk, self.mu_l)
            decay_l = np.exp(-dl_val**2 / (2 * self.sigma**2))
            factor_l = (self.A_l * dl_val / self.sigma**2) * decay_l
            grad_l = factor_l * dl_grad
            
            grad_total[k] = grad_g + grad_l
            
        return grad_total.reshape(z_uhp.shape)
        
    def _dist_grad_helper(self, z1, z2):
        # reuse uhp_distance_and_grad logic for single pair
        # input is scalar complex, wrap in 1-element array
        z1_arr = np.array([z1])
        z2_arr = np.array([z2])
        d_mat, g_mat = uhp_distance_and_grad(z1_arr, z2_arr)
        # d_mat and g_mat are (1,) arrays
        return d_mat[0], g_mat[0]

def run_double_well():
    # 1. Setup
    pot = SyntheticDoubleWellPotential(
        mu_g=1j,   A_g=10.0,
        mu_l=4j,   A_l=6.0,   # Local trap at 4j
        sigma=0.5             # Wider wells for better gradients
    )
    
    # Init particle at 6j (Steep slope of Local Well)
    z_init = np.array([6.0j]) 
    
    print(f"Global Min at {pot.mu_g}, Depth {pot.A_g}")
    print(f"Local Min at {pot.mu_l}, Depth {pot.A_l}")
    print(f"Start at {z_init}")
    print(f"Initial Energy: {pot.potential(z_init):.4f}")
    
    steps = 4000
    
    # 2. HEI Simulation
    cfg = GroupIntegratorConfig(
        max_dt=0.01,
        min_dt=1e-6,
        gamma_scale=0.01,    # VERY Low damping to force tunneling!
        gamma_mode="constant",
        torque_clip=10000.0,
        xi_clip=1000.0,
        use_riemannian_inertia=True
    )
    
    state = GroupIntegratorState(
        G=np.eye(2),
        z0_uhp=z_init,
        xi=np.zeros((1, 3)),
        m=None
    )
    
    force_fn = lambda z, a: -pot.gradient(z, a)
    integrator = GroupContactIntegrator(force_fn, pot.potential, cfg)
    
    hei_path = []
    hei_energies = []
    
    print("Running HEI...")
    for i in range(steps):
        state = integrator.step(state)
        z = state.z_uhp[0]
        hei_path.append(z)
        if i % 10 == 0:
            hei_energies.append(pot.potential(state.z_uhp))
            
    # 3. SGD Simulation
    print("Running SGD...")
    z_curr = z_init.copy().ravel()
    sgd_path = []
    sgd_energies = []
    lr = 0.05
    
    for i in range(steps):
        grad = pot.gradient(z_curr).ravel()
        z_curr = z_curr - lr * grad
        # Project to UHP
        if z_curr.imag < 1e-4: z_curr = z_curr.real + 1e-4j
        
        sgd_path.append(z_curr[0])
        if i % 10 == 0:
            sgd_energies.append(pot.potential(z_curr))
            
    def run_adam_baseline(pot, z0, steps=1000, lr=0.005, b1=0.9, b2=0.999, eps=1e-8):
        # Flatten input
        z = z0.copy().ravel()
        energies = []
        
        # Track Real/Imag components as 2N vector
        z_re = z.real
        z_im = z.imag
        params = np.concatenate([z_re, z_im])
        
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        
        for i in range(1, steps + 1):
            # Reconstruct z
            z_curr_c = params[:z.size] + 1j * params[z.size:]
            z_curr = z_curr_c.reshape(z0.shape)
            
            grad_c = pot.gradient(z_curr).ravel()
            grad_params = np.concatenate([grad_c.real, grad_c.imag])
            
            # Adam Update
            m = b1 * m + (1 - b1) * grad_params
            v = b2 * v + (1 - b2) * (grad_params**2)
            
            m_hat = m / (1 - b1 ** i)
            v_hat = v / (1 - b2 ** i)
            
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Constraint check (y > 0)
            # z_im part is params[z.size:]
            # simpler to reconstruct and check
            z_check = params[:z.size] + 1j * params[z.size:]
            if z_check[0].imag < 1e-4:
                # Project back
                params[z.size:] = np.maximum(params[z.size:], 1e-4)
                
            if i % 10 == 0:
                z_snapshot = params[:z.size] + 1j * params[z.size:]
                energies.append(pot.potential(z_snapshot.reshape(z0.shape)))
        
        return energies, params

    # 4. Adam Simulation
    print("Running Adam...")
    adam_energies, _ = run_adam_baseline(pot, z_init, steps=steps, lr=0.05)
    
    # 5. Momentum SGD (Euclidean)
    print("Running Momentum SGD...")
    z_curr = z_init.copy().ravel()
    mom_energies = []
    v = np.zeros_like(z_curr)
    mu = 0.9 # High momentum
    lr = 0.05
    
    for i in range(steps):
        grad = pot.gradient(z_curr).ravel()
        v = mu * v - lr * grad
        z_curr = z_curr + v
        if z_curr.imag < 1e-4: z_curr = z_curr.real + 1e-4j
        if i % 10 == 0:
            mom_energies.append(pot.potential(z_curr))
            
    # 6. Riemannian SGD (Natural Gradient)
    # Update: z = z - lr * (y^2) * grad
    print("Running Riemannian SGD...")
    z_curr = z_init.copy().ravel()
    rsgd_energies = []
    lr = 0.01 # Smaller LR needed due to y^2 scaling
    
    for i in range(steps):
        grad = pot.gradient(z_curr).ravel()
        # Metric inverse factor y^2
        y = z_curr.imag
        grad_riem = (y**2) * grad
        
        z_curr = z_curr - lr * grad_riem
        if z_curr.imag < 1e-4: z_curr = z_curr.real + 1e-4j
        if i % 10 == 0:
            rsgd_energies.append(pot.potential(z_curr))

    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Energies
    plt.subplot(1, 2, 1)
    plt.plot(hei_energies, label='HEI', linewidth=2)
    plt.plot(sgd_energies, label='SGD', alpha=0.5)
    plt.plot(adam_energies, label='Adam', alpha=0.5)
    plt.plot(mom_energies, label='Momentum SGD', linestyle='--')
    plt.plot(rsgd_energies, label='Riemannian SGD', linestyle=':')
    
    plt.axhline(-pot.A_l, color='r', linestyle='--', label='Local Min Level')
    plt.axhline(-pot.A_g, color='g', linestyle='--', label='Global Min Level')
    plt.legend()
    plt.title('Fairness Check: HEI vs Baselines')
    plt.xlabel('Steps/10')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Trajectories
    plt.subplot(1, 2, 2)
    hei_p = np.array(hei_path)
    # Re-run baselines to get paths? simplified just plotting HEI vs traps
    # actually let's skip paths for baselines to avoid clutter, focus on ENERGY.
    plt.plot(hei_p.real, hei_p.imag, label='HEI', alpha=0.7)
    
    plt.scatter([pot.mu_g.real], [pot.mu_g.imag], c='g', marker='*', s=200, label='Global')
    plt.scatter([pot.mu_l.real], [pot.mu_l.imag], c='r', marker='x', s=200, label='Local')
    plt.scatter([z_init.real], [z_init.imag], c='k', marker='o', label='Start')
    
    plt.legend()
    plt.title('HEI Trajectory')
    plt.xlim(-2, 2)
    plt.ylim(0, 11)
    
    plt.savefig('double_well_fairness.png')
    
    print(f"Final E - HEI: {hei_energies[-1]:.4f}")
    print(f"Final E - SGD: {sgd_energies[-1]:.4f}")
    print(f"Final E - Adam: {adam_energies[-1]:.4f}")
    print(f"Final E - MomSGD: {mom_energies[-1]:.4f}")
    print(f"Final E - RSGD: {rsgd_energies[-1]:.4f}")

if __name__ == "__main__":
    run_double_well()

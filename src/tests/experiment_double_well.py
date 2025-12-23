
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
class SoftDoubleWellPotential:
    """
    Soft-Core Double Well Potential.
    Consistent with Hamiltonian Regularization.
    
    V(z) = - A_g * exp(-(d^2 + eps^2) / 2sigma^2) ...
    """
    mu_g: complex = 1j      # Global Min
    mu_l: complex = 3j      # Local Min
    A_g: float = 10.0       # Global Depth
    A_l: float = 5.0        # Local Depth
    sigma: float = 0.5
    eps_soft: float = 0.2   # Smoothing
    
    def potential(self, z_uhp, action=None):
        z = np.asarray(z_uhp, dtype=np.complex128).ravel()
        
        # Helper for soft dist
        def get_d_soft(target):
            d_true, _ = uhp_distance_and_grad(z, target)
            return np.sqrt(d_true**2 + self.eps_soft**2)

        d_g = get_d_soft(self.mu_g)
        d_l = get_d_soft(self.mu_l)
        
        V_g = -self.A_g * np.exp(-d_g**2 / (2 * self.sigma**2))
        V_l = -self.A_l * np.exp(-d_l**2 / (2 * self.sigma**2))
        
        return np.sum(V_g + V_l)

    def gradient(self, z_uhp, action=None):
        z = np.asarray(z_uhp, dtype=np.complex128).ravel()
        n = z.size
        Grad = np.zeros(n, dtype=np.complex128)
        
        for k in range(n):
            zk = z[k]
            
            # Global Well
            d_true, d_grad = self._dist_grad_helper(zk, self.mu_g)
            d_soft = np.sqrt(d_true**2 + self.eps_soft**2)
            grad_scale = d_true / d_soft
            
            # V = -A exp(-d_soft^2 / 2s^2)
            # dV/dd_soft = A * d_soft/s^2 * exp
            decay_g = np.exp(-d_soft**2 / (2 * self.sigma**2))
            factor_g = (self.A_g * d_soft / self.sigma**2) * decay_g
            # Chain rule: dV/dz = dV/dd_soft * dd_soft/dd_true * dd_true/dz
            term_g = factor_g * grad_scale * d_grad
            
            # Local Well
            d_true, d_grad = self._dist_grad_helper(zk, self.mu_l)
            d_soft = np.sqrt(d_true**2 + self.eps_soft**2)
            grad_scale = d_true / d_soft
            
            decay_l = np.exp(-d_soft**2 / (2 * self.sigma**2))
            factor_l = (self.A_l * d_soft / self.sigma**2) * decay_l
            term_l = factor_l * grad_scale * d_grad
            
            Grad[k] = term_g + term_l
            
        return Grad.reshape(z_uhp.shape)
        
    def _dist_grad_helper(self, z1, z2):
        z1_arr = np.array([z1])
        z2_arr = np.array([z2])
        d_mat, g_mat = uhp_distance_and_grad(z1_arr, z2_arr)
        return d_mat[0], g_mat[0]

def run_double_well():
    # 1. Setup
    print("Setting up Soft Double Well...")
    pot = SoftDoubleWellPotential(
        mu_g=1j,   A_g=10.0,
        mu_l=4j,   A_l=6.0,
        sigma=0.5, eps_soft=0.2
    )
    
    # Init particle at 6j (Steep slope of Local Well)
    z_init = np.array([6.0j]) 
    
    print(f"Global Min at {pot.mu_g}, Depth {pot.A_g}")
    print(f"Local Min at {pot.mu_l}, Depth {pot.A_l}")
    print(f"Start at {z_init}")
    print(f"Initial Energy: {pot.potential(z_init):.4f}")
    
    steps = 4000
    
    # 2. HEI Physics Simulation (Soft Core = No Clipping)
    cfg = GroupIntegratorConfig(
        max_dt=0.01,
        min_dt=1e-6,
        gamma_scale=0.01,    # Low damping to enforce tunneling
        gamma_mode="constant",
        torque_clip=10000.0, # NO CLIPPING
        xi_clip=1000.0,
        use_riemannian_inertia=True,
        implicit_potential=True # Robust
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
    
    print("Running HEI (Soft Physics)...")
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
    
    # 5. Nesterov Accelerated Gradient (NAG)
    print("Running NAG...")
    z_curr = z_init.copy().ravel()
    nag_energies = []
    v = np.zeros_like(z_curr)
    mu = 0.9 
    lr = 0.05
    
    for i in range(steps):
        # Lookahead gradient
        z_lookahead = z_curr + mu * v
        grad = pot.gradient(z_lookahead).ravel()
        v = mu * v - lr * grad
        z_curr = z_curr + v
        if z_curr.imag < 1e-4: z_curr = z_curr.real + 1e-4j
        if i % 10 == 0:
            nag_energies.append(pot.potential(z_curr))
            
    # 6. Euclidean Hamiltonian Dynamics (EHD) - The "Fair" Competitor
    # Solves z'' = -grad V - gamma * z' in Euclidean space.
    # Same dt and gamma as HEI.
    print("Running Euclidean Dynamics (Gamma=0.01)...")
    z_curr = z_init.copy().ravel()
    vel = np.zeros_like(z_curr) # Start at rest
    ehd_energies = []
    
    dt = 0.01 # Matching HEI max_dt
    gamma = 0.01 # Matching HEI
    
    for i in range(steps):
        # Semi-implicit Euler or Velocity Verlet
        # v += (-grad - gamma*v) * dt
        # z += v * dt
        
        grad = pot.gradient(z_curr).ravel()
        acc = -grad - gamma * vel
        vel = vel + acc * dt
        z_curr = z_curr + vel * dt
        
        if z_curr.imag < 1e-4: 
            z_curr = z_curr.real + 1e-4j
            vel.imag = 0 # Bounce/Stick? Stick for safety.
            
        if i % 10 == 0:
            ehd_energies.append(pot.potential(z_curr))

    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Energies
    plt.subplot(1, 2, 1)
    plt.plot(hei_energies, label='HEI (Hyp+Inertia)', linewidth=2, color='blue')
    plt.plot(nag_energies, label='NAG', linestyle='--', color='orange')
    plt.plot(ehd_energies, label='EHD (Euc+Inertia)', linestyle='-', color='purple')
    plt.plot(sgd_energies, label='SGD', alpha=0.3, color='grey')
    plt.plot(adam_energies, label='Adam', alpha=0.3, color='brown')
    
    plt.axhline(-pot.A_l, color='r', linestyle=':', label='Local Min')
    plt.axhline(-pot.A_g, color='g', linestyle=':', label='Global Min')
    plt.legend()
    plt.title('Rigorous Fairness Check (Low Damping)')
    plt.xlabel('Steps/10')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Trajectories
    plt.subplot(1, 2, 2)
    hei_p = np.array(hei_path)
    
    plt.plot(hei_p.real, hei_p.imag, label='HEI', alpha=0.7)
    
    plt.scatter([pot.mu_g.real], [pot.mu_g.imag], c='g', marker='*', s=200, label='Global')
    plt.scatter([pot.mu_l.real], [pot.mu_l.imag], c='r', marker='x', s=200, label='Local')
    plt.scatter([z_init.real], [z_init.imag], c='k', marker='o', label='Start')
    
    plt.legend()
    plt.title('HEI Trajectory')
    plt.xlim(-2, 2)
    plt.ylim(0, 11)
    
    plt.savefig('double_well_fairness_v2.png')
    
    print(f"Final E - HEI: {hei_energies[-1]:.4f}")
    print(f"Final E - NAG: {nag_energies[-1]:.4f}")
    print(f"Final E - EHD: {ehd_energies[-1]:.4f}")
    print(f"Final E - SGD: {sgd_energies[-1]:.4f}")
    print(f"Final E - Adam: {adam_energies[-1]:.4f}")

if __name__ == "__main__":
    run_double_well()

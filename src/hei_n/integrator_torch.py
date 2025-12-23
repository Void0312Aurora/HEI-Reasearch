
"""
PyTorch Contact Integrator N (GPU Accelerated).
==============================================

Port of ContactIntegratorN to PyTorch.
"""

import torch
import dataclasses
from typing import Any, Protocol, Dict
from .torch_core import exp_so1n_torch, renormalize_so1n_torch, project_to_algebra_torch, minkowski_metric_torch

@dataclasses.dataclass
class ContactStateTorch:
    G: torch.Tensor # (N, dim, dim)
    M: torch.Tensor # (N, dim, dim)
    z: torch.Tensor # (N,) or Scalar?
    x: torch.Tensor = None
    diagnostics: Dict = dataclasses.field(default_factory=dict)
    
    def __post_init__(self):
        if self.x is None:
            self.x = self.G[..., 0]

@dataclasses.dataclass
class ContactConfigTorch:
    dt: float = 0.01
    min_dt: float = 1e-5
    gamma: float = 0.0
    target_temp: float = None
    thermostat_tau: float = 10.0
    fixed_point_iters: int = 3
    solver_mixing: float = 0.5
    torque_clip: float = 50.0
    renorm_interval: int = 50
    adaptive: bool = True
    tol_disp: float = 0.1
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IdentityInertiaTorch:
    def inverse_inertia(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return M
        
    def geometry_force(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
        
    def kinetic_energy(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # T = 0.5 * <M, M>_K = 0.5 * Tr(-J M J M^T) ? 
        # For M = [[0, v], [v, 0]], T = 0.5 * 2 * |v|^2 = |v|^2
        v = M[:, 0, 1:]
        return torch.sum(v**2, dim=-1)

def compute_diamond_torque_torch(G: torch.Tensor, Force: torch.Tensor) -> torch.Tensor:
    """
    Compute torque X = G^T * Force ^ Diamond.
    Diamond(g, F) => Lift force to algebra.
    For particle model: X = u ^ v term?
    
    Actually:
    Torque = x ^ F in algebra.
    But M is in body frame or spatial frame?
    In `contact_integrator_n`, M is in spatial frame (so(1,n)).
    Wait, `group_integrator` usually evolves M in body frame or spatial?
    Let's check `contact_integrator_n.py` usage:
    `grad_pot = oracle.gradient(x_world)`
    `Force_world = ...`
    `Torque_explicit = compute_diamond_torque(G, Force_world)`
    
    In `inertia_n.py`: 
    `compute_diamond_torque(G, F)` computes `G F^T J - J F G^T` ? 
    Let's check `inertia_n.py` or recall formula.
    Actually for x (position vector), Torque from Force F at x is X = x F^T - F x^T (with metric J).
    X = x F^T J - F x^T J ??
    
    Let's assume standard formula:
    X_{ij} = x_i F_j - x_j F_i (Euclidean)
    Hyperbolic: X = x F^T J + F x^T J ? 
    No, X must satisfy X^T J + J X = 0.
    
    Standard definition: Torque = x \wedge F.
    (x \wedge F) u = <F, u> x - <x, u> F
    Matrix: X = x F^T J - F x^T J.
    Check skew-adjointness:
    X^T J = (J F x^T - J x F^T) J = J F x^T J - J x F^T J
    J X = J x F^T J - J F x^T J
    Sum = 0. Correct.
    
    So X = x F^T J - F x^T J.
    """
    dim = G.shape[-1]
    x = G[..., 0] # (N, dim)
    F = Force # (N, dim)
    
    # Outer products
    # xF^T: (N, dim, dim)
    xF = torch.einsum('ni,nj->nij', x, F)
    Fx = torch.einsum('ni,nj->nij', F, x)
    
    J = minkowski_metric_torch(dim, G.device)
    
    # X = (xF - Fx) @ J
    # xF^T J 
    # (xF - Fx) J ? 
    # Let's verify X = x F^T J - F x^T J
    
    X = (xF - Fx) @ J
    return X

class ContactIntegratorTorch:
    def __init__(self, oracle: Any, inertia: Any, config: ContactConfigTorch):
        self.oracle = oracle
        self.inertia = inertia
        self.config = config
        self.gamma = config.gamma
        self._step_count = 0
        
    def step(self, state: ContactStateTorch) -> ContactStateTorch:
        self._step_count += 1
        G = state.G
        M = state.M
        z = state.z
        
        diag = {}
        
        x_world = G[..., 0]
        
        # Thermostat
        if self.config.target_temp is not None:
             N = G.shape[0]
             # KE is (N,)
             ke = self.inertia.kinetic_energy(M, x_world)
             T_curr = torch.mean(ke)
             
             delta = (self.config.dt / self.config.thermostat_tau) * (T_curr - self.config.target_temp)
             self.gamma = max(0.0, self.gamma + delta.item())
             
        gamma = self.gamma
        
        # 1. Forces
        # Oracle returns (energy, grad)
        V_tot, grad_pot = self.oracle.potential_and_grad(x_world)
        
        # Geom Force = 0 for Identity
        # grad_geom = 0
        
        # Tangent Projection
        from .torch_core import project_to_tangent_torch
        Force_world = - project_to_tangent_torch(x_world, grad_pot)
        
        Torque_explicit = compute_diamond_torque_torch(G, Force_world)
        
        # 2. Adaptive Step
        # xi = I^-1 M = M
        xi_norm = torch.max(torch.norm(M.reshape(G.shape[0], -1), dim=1))
        torque_norm = torch.max(torch.norm(Torque_explicit.reshape(G.shape[0], -1), dim=1))
        
        if self.config.adaptive:
            dt_vel = self.config.tol_disp / (xi_norm + 1e-9)
            dt_acc = self.config.tol_disp / (torque_norm + 1e-9)
            dt_gamma = 0.8 / (self.gamma + 1e-9)
            
            dt = torch.min(dt_vel, dt_acc)
            dt = torch.min(dt, torch.tensor(dt_gamma, device=G.device))
            # dt = min(dt, config.dt)
            dt = torch.clamp(dt, min=self.config.min_dt, max=self.config.dt)
            dt_val = dt.item()
        else:
            dt_val = self.config.dt
            
        diag['dt'] = dt_val
        
        # 3. Implicit Solver (Fixed Point)
        xi_iter = M.clone() # I^-1 M = M
        M_current = M
        damping = 1.0 / (1.0 + dt_val * gamma)
        
        solver_res = 0.0
        
        for _ in range(self.config.fixed_point_iters):
             Torque_loop = Torque_explicit
             
             # Clip
             t_mag = torch.norm(Torque_loop.reshape(G.shape[0], -1), dim=1)
             mask = t_mag > self.config.torque_clip
             if torch.any(mask):
                 scale = self.config.torque_clip / (t_mag[mask] + 1e-9)
                 # scale (K,)
                 # Torque[mask] (K, D, D)
                 Torque_loop[mask] = Torque_loop[mask] * scale.view(-1, 1, 1)
                 
             M_next = (M_current + dt_val * Torque_loop) * damping
             xi_next = M_next # Identity
             
             res = torch.max(torch.abs(xi_next - xi_iter))
             solver_res = res.item()
             
             beta = self.config.solver_mixing
             xi_iter = (1.0 - beta) * xi_iter + beta * xi_next
             
        diag['solver_residual'] = solver_res
        
        xi_final = xi_iter
        M_new = M_next
        
        # 4. Update G
        step_g = exp_so1n_torch(xi_final, dt=dt_val)
        G_new = G @ step_g
        
        # 5. Update z
        T_val = self.inertia.kinetic_energy(M_new, x_world) # (N,)
        # L = T - V - gamma z
        # But V is total potential? Or per particle?
        # Typically z is global. V is total.
        # But contact action usually per particle?
        # "M = T*Q x R_z". z is a scalar if simple contact.
        # But in `contact_integrator_n`, z is float (scalar).
        # So summing T and V.
        T_total = torch.sum(T_val)
        L = T_total - V_tot - gamma * z
        z_new = z + L * dt_val
        
        # 6. Renorm
        renorm_mag = 0.0
        if self._step_count % self.config.renorm_interval == 0:
            G_new, renorm_mag = renormalize_so1n_torch(G_new)
            
        diag['renorm_magnitude'] = renorm_mag
        
        return ContactStateTorch(G=G_new, M=M_new, z=z_new, diagnostics=diag)

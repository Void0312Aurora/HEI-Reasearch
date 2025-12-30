import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseKernel(nn.Module, ABC):
    def __init__(self, dim_q: int):
        super().__init__()
        self.dim_q = dim_q
        
    @abstractmethod
    def forward(self, x_int: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """
        Evolve internal state one step.
        x_int: Batch x Dim
        u_t: Batch x InputDim
        """
        pass
    
    @abstractmethod
    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        pass

class SymplecticKernel(BaseKernel):
    """
    Baseline A: Conservative, Symplectic Kernel.
    State: (q, p). No s.
    Dynamics: Hamiltonian flow.
    """
    def __init__(self, dim_q: int = 2):
        super().__init__(dim_q)
        # Potential function U(q) parameterization
        self.net_U = nn.Sequential(
            nn.Linear(dim_q, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
    def H(self, q, p):
        # H = T(p) + U(q) = 0.5 * p^2 + U(q)
        return 0.5 * (p**2).sum(dim=1, keepdim=True) + self.net_U(q)
    
    def forward(self, x_int: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # x_int = torch.cat([q, p], dim=1)
        q, p = x_int.chunk(2, dim=1)
        dt = 0.1
        
        # Symplectic Euler with external forcing u_t acting as force
        # p_{t+1} = p_t - dH/dq * dt + u_t * dt
        # q_{t+1} = q_t + dH/dp_{t+1} * dt
        
        # Approx: dU/dq
        q.requires_grad_(True)
        U = self.net_U(q).sum()
        grad_q = torch.autograd.grad(U, q, create_graph=False)[0]
        
        # u_t maps to force. For simplicity, assume u_t dim matches q dim.
        force = u_t
        
        p_new = p - grad_q * dt + force * dt
        q_new = q + p_new * dt # dH/dp = p
        
        return torch.cat([q_new, p_new], dim=1)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, 2 * self.dim_q)


class ContactKernel(BaseKernel):
    """
    Baseline B: Dissipative, Contact Kernel.
    State: (q, p, s).
    Dynamics: Contact Hamiltonian flow.
    """
    def __init__(self, dim_q: int = 2, damping: float = 0.1):
        super().__init__(dim_q)
        self.damping = damping
        self.net_U = nn.Sequential(
            nn.Linear(dim_q, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
    def _potential(self, q):
        # Total potential: net_U(q) + boundary_pot
        q_norm = (q**2).sum(dim=1, keepdim=True)
        boundary_pot = torch.relu(q_norm - 20.0) ** 2
        return self.net_U(q) + boundary_pot

    def H(self, q, p, s):
        # Contact Hamiltonian H = 0.5*p^2 + _potential(q) + alpha*s
        return 0.5 * (p**2).sum(dim=1, keepdim=True) + self._potential(q) + self.damping * s
        
    def forward(self, x_int: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # x_int = [q, p, s]
        q, p, s = torch.split(x_int, [self.dim_q, self.dim_q, 1], dim=1)
        dt = 0.1
        
        q.requires_grad_(True)
        # Fix: Use total potential (including boundary) for gradient
        U = self._potential(q).sum()
        grad_q = torch.autograd.grad(U, q, create_graph=False)[0]
        grad_s = self.damping  # dH/ds
        
        # force from input
        force = u_t
        
        # Contact Euler
        # p_new = p - (dH/dq + p*dH/ds) * dt + force
        p_new = p - (grad_q + p * grad_s) * dt + force * dt
        
        # q_new = q + dH/dp * dt
        q_new = q + p_new * dt
        
        # s_new = s + (p*dH/dp - H) * dt
        # H should also include boundary potential now!
        H_val = 0.5 * (p**2).sum(dim=1, keepdim=True) + self._potential(q).detach() + self.damping * s
        dot_s = (p**2).sum(dim=1, keepdim=True) - H_val
        s_new = s + dot_s * dt
        
        return torch.cat([q_new, p_new, s_new], dim=1)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, 2 * self.dim_q + 1)

class FastSlowKernel(ContactKernel):
    """
    Variant C: Contact + Fast-Slow Structure.
    State: (q, p, s).
    Dynamics: Contact Hamiltonian flow but with spectral gap induced by parameterization.
    """
    def __init__(self, dim_q: int = 2, damping: float = 0.1, epsilon: float = 0.1):
        super().__init__(dim_q, damping)
        self.epsilon = epsilon
        # Add a slow manifold potential or stiff terms
        # For simplicity v0, we just scale part of the dynamics to be slow.
        # But per plan "Variant C: ... generate separation via generator decomposition or stiffness".
        
    def forward(self, x_int: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # We can implement a simple stiff system.
        # q[0] is fast, q[1] is slow.
        
        # We override forward to inject epsilon scaling
        q, p, s = torch.split(x_int, [self.dim_q, self.dim_q, 1], dim=1)
        dt = 0.1
        
        q.requires_grad_(True)
        # Fix: Use _potential from base class
        U = self._potential(q).sum()
        grad_q = torch.autograd.grad(U, q, create_graph=False)[0]
        grad_s = self.damping
        
        force = u_t
        
        # Create a mass matrix M/Stiffness matrix where one dim is heavy/stiff
        # Or just scale the gradients: 
        # For slow variable q_s, dot_q_s = epsilon * ... 
        # For fast variable q_f, dot_q_f = 1 * ...
        
        # Let's assume dim 0 is fast, dim 1 is slow (epsilon).
        timescale = torch.ones_like(p)
        timescale[:, 1:] = self.epsilon 
        
        # Modified equations for timescale separation:
        # We scale the vector field for the slow components.
        # This is a naive implementation of fast-slow.
        
        p_new = p - (grad_q + p * grad_s) * dt + force * dt
        
        # Apply timescale to velocity (q evolution)
        q_new = q + (p_new * timescale) * dt 
        
        # s evolution
        H_val = 0.5 * (p**2).sum(dim=1, keepdim=True) + self._potential(q).detach() + self.damping * s
        dot_s = (p**2).sum(dim=1, keepdim=True) - H_val
        s_new = s + dot_s * dt
        
        return torch.cat([q_new, p_new, s_new], dim=1)

class ResonantKernel(ContactKernel):
    """
    Iteration 1.1: Temporal Resonance.
    Adds a harmonic oscillator 'resonator' state to detect frequency content.
    State: (q, p, s, r, v_r) where r is resonator pos, v_r is resonator vel.
    External input u_t drives the resonator.
    """
    def __init__(self, dim_q: int = 2, damping: float = 0.1, omega: float = 1.0):
        super().__init__(dim_q, damping)
        self.omega = omega    # Resonant frequency
        self.zeta = 0.1       # Resonator damping ratio
        
    def _potential(self, q):
        return super()._potential(q)

    def forward(self, x_int: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # State layout: 
        # [q (dim), p (dim), s (1), r (dim), v_r (dim)]
        # Total dim = 4*dim + 1
        
        # Unpack state
        d = self.dim_q
        q, p, s, r, v_r = torch.split(x_int, [d, d, 1, d, d], dim=1)
        dt = 0.1
        
        # 1. Base Contact Dynamics for (q, p, s)
        q.requires_grad_(True)
        U = self._potential(q).sum()
        grad_q = torch.autograd.grad(U, q, create_graph=False)[0]
        grad_s = self.damping
        
        force = u_t
        
        p_new = p - (grad_q + p * grad_s) * dt + force * dt
        q_new = q + p_new * dt
        
        H_val = 0.5 * (p**2).sum(dim=1, keepdim=True) + self._potential(q).detach() + self.damping * s
        dot_s = (p**2).sum(dim=1, keepdim=True) - H_val
        s_new = s + dot_s * dt
        
        # 2. Resonator Dynamics (Driven Harmonic Oscillator)
        # ddot_r + 2*zeta*omega*dot_r + omega^2*r = u_t
        damping_term = 2 * self.zeta * self.omega * v_r
        stiffness_term = (self.omega ** 2) * r
        acc_r = -stiffness_term - damping_term + u_t
        
        v_r_new = v_r + acc_r * dt
        r_new = r + v_r_new * dt
        
        return torch.cat([q_new, p_new, s_new, r_new, v_r_new], dim=1)
        
    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        # 4*dim + 1
        return torch.zeros(batch_size, 4 * self.dim_q + 1)

class PlasticKernel(ResonantKernel):
    """
    Iteration 1.3: Causal Plasticity.
    Adds a plastic weight matrix W that learns correlations between input u_t and resonator state r.
    State: (q, p, s, r, v_r, w_flat)
    W is dim x dim (coupling u to r). Flattened in state.
    Update Rule: 
    - Hebbian: dW/dt = eta * (u_t * r - decay * W)
    - Delta:   dW/dt = eta * ((u_t - W*r) * r - decay * W)
    """
    def __init__(self, dim_q: int = 2, damping: float = 0.1, omega: float = 1.0, eta: float = 0.01, learning_rule: str = 'hebbian'):
        super().__init__(dim_q, damping, omega)
        self.eta = eta # Learning rate
        self.learning_rule = learning_rule
        
    def forward(self, x_int: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # State layout:
        # ResonantKernel: [q(d), p(d), s(1), r(d), vr(d)] (Total 4d+1)
        # Plastic: + W(d*d) flattened.
        
        d = self.dim_q
        base_dim = 4*d + 1
        w_dim = d*d
        
        # Split base state and weights
        # Note: torch.split needs explicit sizes
        base_state = x_int[:, :base_dim]
        w_flat = x_int[:, base_dim:]
        
        # Evolve base dynamics (ResonantKernel)
        # We need to call super().forward(). 
        # But super().forward() takes x_int matching its size.
        # We can pass base_state, but we need to ensure the returned state is handled.
        
        # 1. Component Dynamics
        base_next = super().forward(base_state, u_t)
        
        # Unpack essential variables for plasticity
        # r is at index [2d+1 : 3d+1] in base state
        q, p, s, r, v_r = torch.split(base_state, [d, d, 1, d, d], dim=1)
        
        # 2. Plasticity Rule
        # dW = eta * (u * r^T - gamma * W)
        # Let's align dimensions:
        # u_t: [B, d]
        # r: [B, d]
        # W: [B, d*d] -> [B, d, d]
        
        W = w_flat.view(-1, d, d)
        
        # Outer product u * r
        # u.unsqueeze(2) -> [B, d, 1]
        # r.unsqueeze(1) -> [B, 1, d]
        # product -> [B, d, d]
        
        if self.learning_rule == 'delta':
            # Delta Rule: error * r^T
            # Error e = u - W*r
            # W*r : [B, d, d] x [B, d, 1] -> [B, d, 1] -> squeeze -> [B, d]
            pred = torch.bmm(W, r.unsqueeze(2)).squeeze(2)
            error = u_t - pred
            drive_term = torch.bmm(error.unsqueeze(2), r.unsqueeze(1))
        else:
            # Hebbian Rule: u * r^T
            drive_term = torch.bmm(u_t.unsqueeze(2), r.unsqueeze(1))
            
        decay = 0.1 # Weight decay
        dW = self.eta * (drive_term - decay * W)
        
        dt = 0.1
        W_new = W + dW * dt
        w_flat_new = W_new.view(-1, d*d)
        
        return torch.cat([base_next, w_flat_new], dim=1)
        
    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        d = self.dim_q
        base_state = super().init_state(batch_size)
        w_state = torch.zeros(batch_size, d*d)
        return torch.cat([base_state, w_state], dim=1)

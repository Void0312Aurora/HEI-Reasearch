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

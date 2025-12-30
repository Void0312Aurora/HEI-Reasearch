import torch
import torch.nn as nn
from typing import Callable
from he_core.state import ContactState

class ContactIntegrator:
    """
    Implements Contact Hamiltonian Dynamics.
    Equations:
    dot_q = dH/dp
    dot_p = -(dH/dq + p * dH/ds)
    dot_s = p * dH/dp - H
    """
    def __init__(self, method='euler'):
        self.method = method # TODO: Support symplectic/contact integrators
        
    def step(self, state: ContactState, generator: Callable[[ContactState], torch.Tensor], dt: float = 0.1) -> ContactState:
        """
        Perform one step of integration.
        Returns NEW state (out-of-place).
        """
        # Ensure gradients are enabled for H computation
        x = state.flat.clone().detach().requires_grad_(True)
        
        # We need to reconstruct ContactState wrapper to pass to generator
        # But generator expects ContactState with properties.
        # We can implement a lightweight wrapper or just use the same class
        # if x is (B, D).
        
        temp_state = ContactState(state.dim_q, state.batch_size, state.device, x)
        
        # Compute H
        # H should be (B, 1) or scalar Sum?
        # For batched grad, we sum H, then grad.
        H = generator(temp_state) # (B, 1)
        H_sum = H.sum()
        
        # Compute Gradients
        # We need grad w.r.t q, p, s
        # x is [q, p, s]
        grads = torch.autograd.grad(H_sum, x, create_graph=False)[0]
        
        # Extract derivatives
        d = state.dim_q
        dH_dq = grads[:, :d]
        dH_dp = grads[:, d:2*d]
        dH_ds = grads[:, 2*d:]
        
        # Extract state components
        q = temp_state.q
        p = temp_state.p
        s = temp_state.s # (B, 1)
        
        # Contact Equations
        dot_q = dH_dp
        dot_p = -(dH_dq + p * dH_ds)
        
        # dot_s = p * dH_dp - H
        # dot_p * p? No, p * dH/dp (inner product)
        # (B, D) * (B, D) -> (B, 1)
        term1 = (p * dH_dp).sum(dim=1, keepdim=True)
        dot_s = term1 - H.detach() # H is value here. Detach H? 
        # Actually H is needed. We computed H.
        # Warning: H might track graph. Detach for update.
        
        # Euler Step
        new_flat = x.detach() + torch.cat([dot_q, dot_p, dot_s], dim=1) * dt
        
        return ContactState(state.dim_q, state.batch_size, state.device, new_flat)

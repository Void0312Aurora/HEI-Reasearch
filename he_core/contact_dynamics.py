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
        # Preserve Graph: Do NOT detach.
        # x is the flat state tensor.
        x = state.flat
        
        if not x.requires_grad and x.is_leaf:
             x.requires_grad_(True)
        
        temp_state = ContactState(state.dim_q, state.batch_size, state.device, x)
        
        # Compute H
        H = generator(temp_state) # (B, 1)
        H_sum = H.sum()
        
        # Restore Fundamental Solution: create_graph=True
        # This allows learning d(VectorField)/d(Theta).
        # We rely on PyTorch's ability to handle this without explicit retain_graph for the H calculation itself,
        # assuming H's graph is used for grad and then effectively reconstructed/kept via the create_graph connection?
        # Actually, if we use H in dot_s, we MUST retain_graph if grad() destroys it.
        # But debug_functorch passed WITHOUT retain_graph even with H in update.
        # This implies grad() didn't destroy the path needed for H->Update.
        
        grads = torch.autograd.grad(H_sum, x, create_graph=True)[0]
        
        # Extract derivatives
        d = state.dim_q
        dH_dq = grads[:, :d]
        dH_dp = grads[:, d:2*d]
        dH_ds = grads[:, 2*d:]
        
        # Extract state components
        q = temp_state.q
        p = temp_state.p
        
        # Contact Equations
        dot_q = dH_dp
        dot_p = -(dH_dq + p * dH_ds)
        
        # dot_s = p * dH_dp - H
        term1 = (p * dH_dp).sum(dim=1, keepdim=True)
        # Use H value directly, maintaining graph connection
        dot_s = term1 - H 
        
        # Euler Step
        # new_x = x + ...
        new_flat = x + torch.cat([dot_q, dot_p, dot_s], dim=1) * dt
        
        return ContactState(state.dim_q, state.batch_size, state.device, new_flat)

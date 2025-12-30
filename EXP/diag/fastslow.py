import torch
import numpy as np
from typing import Callable
from he_core.state import ContactState

def compute_jacobian_spectrum(integrator, generator, state: ContactState, dt: float = 0.001):
    """
    Computes the Jacobian of the one-step map T(x) = x + F(x)*dt.
    Returns sorted eigenvalues (real part).
    """
    # 1. Flatten State
    x = state.flat.clone().detach().requires_grad_(True)
    
    # 2. Define Function wrapper
    def func(x_in):
        # reconstruct
        s_in = ContactState(state.dim_q, state.batch_size, state.device, x_in)
        s_next = integrator.step(s_in, generator, dt)
        return s_next.flat
        
    # 3. Compute Jacobian using autograd.functional
    # x is (1, D). Output is (1, D).
    # Since batch=1 mostly.
    
    # jacobian() needs inputs as tuple?
    # Or just functional.jacobian(func, x)
    
    try:
        # torch.autograd.functional.jacobian returns (B, D, B, D)
        # We assume B=1.
        J = torch.autograd.functional.jacobian(func, x) # (1, D, 1, D)
        J = J.squeeze() # (D, D)
    except Exception as e:
        print(f"Jacobian computation failed: {e}")
        return None
    
    # 4. Compute Eigenvalues
    # T = J. (Linearized map)
    # Eigs of T.
    # Growth rates are log(lambda) / dt?
    # Or just return raw eigenvalues to see modulus.
    
    # torch.linalg.eig
    # Helper: Convert to CPU numpy for robust eig?
    J_np = J.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(J_np)
    
    # Sort by magnitude (Fastest modes first? or Slowest?)
    # Slow modes: lambda ~ 1. (Continuous: 0)
    # Fast modes: lambda < 1. (Continuous: < 0)
    
    # Let's return sorted by Real part of Continuous Spectrum approx
    # lambda_cont ~ (lambda_disc - 1) / dt
    
    spec_cont = (eigvals - 1.0) / dt
    # Sort by real part descending (Close to 0 is slow, Negative is fast)
    idx = np.argsort(spec_cont.real)[::-1]
    sorted_spec = spec_cont[idx]
    
    return sorted_spec

def report_spectral_gap(spectrum):
    """
    Analyzes spectrum for gaps.
    Expects sorted spectrum (Real desc).
    """
    real_parts = spectrum.real
    
    # Print top 5
    print("Top Modes (Real):", real_parts[:5])
    
    # Find gap
    # Gap = difference between consecutive eigenvalues
    gaps = np.abs(np.diff(real_parts))
    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]
    
    print(f"Max Spectral Gap: {max_gap:.4f} at index {max_gap_idx}")
    return max_gap

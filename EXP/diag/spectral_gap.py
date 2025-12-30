import torch
import numpy as np
import logging
from typing import Dict, Any, List
from he_core.state import ContactState

logger = logging.getLogger(__name__)

def compute_spectral_properties(integrator, generator, state_batch: ContactState, dt: float = 0.001) -> Dict[str, float]:
    """
    Protocol 1: Local Linearization & Spectral Gap.
    Computes averaged Jacobian spectrum over a batch.
    """
    # Use only first item if batch is large, or iterate?
    # Jacobian is expensive (D^2).
    # We take the first sample in batch for analysis.
    
    bs = state_batch.batch_size
    dim = state_batch.flat.shape[1]
    
    # Analyze first sample
    x0 = state_batch.flat[0:1].clone().detach().requires_grad_(True) # (1, D)
    
    def func(x_in):
        s_in = ContactState(state_batch.dim_q, 1, state_batch.device, x_in)
        s_next = integrator.step(s_in, generator, dt)
        return s_next.flat
        
    try:
        J = torch.autograd.functional.jacobian(func, x0) # (1, D, 1, D)
        J = J.squeeze().cpu().numpy() # (D, D)
        
        eigvals = np.linalg.eigvals(J)
        
        # Continuous Spectrum Approx: mu = (lambda - 1) / dt
        spec_cont = (eigvals - 1.0) / dt
        real_parts = np.sort(spec_cont.real)[::-1] # Descending (0, -1, -10...)
        
        # Slowest (max real)
        slowest = real_parts[0]
        # Fastest (min real)
        fastest = real_parts[-1]
        
        # Gap: Search for largest diff in sorted real parts
        gaps = np.abs(np.diff(real_parts))
        max_gap = np.max(gaps) if len(gaps) > 0 else 0.0
        
        return {
            "max_gap": float(max_gap),
            "slowest_mode": float(slowest),
            "fastest_mode": float(fastest),
            "spectrum": real_parts.tolist()
        }
        
    except Exception as e:
        logger.error(f"Spectral Gap Computation Failed: {e}")
        return {"max_gap": 0.0, "error": str(e)}

def report_spectral_gap(metrics: Dict[str, float], threshold: float = 2.0) -> bool:
    gap = metrics.get("max_gap", 0.0)
    print(f"[Protocol 1] Spectral Gap: {gap:.4f}")
    print(f"  Slowest: {metrics.get('slowest_mode', 0):.4f}")
    print(f"  Fastest: {metrics.get('fastest_mode', 0):.4f}")
    
    if gap > threshold:
        print("  Status: PASS (Structure detected)")
        return True
    else:
        print("  Status: FAIL (No significant time scale separation)")
        return False

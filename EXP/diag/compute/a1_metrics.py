import numpy as np
from scipy import stats
from typing import Dict, Any

def compute_conditional_mi(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Computes I(X; Y | Z) using Gaussian approximation (Partial Correlation).
    I(X; Y | Z) = -0.5 * ln(1 - rho_{xy|z}^2)
    
    rho_{xy|z} = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    """
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    min_len = min(len(x), len(y), len(z))
    x = x[:min_len]
    y = y[:min_len]
    z = z[:min_len]
    
    if min_len < 3:
        return 0.0
        
    # Pearson Correlations
    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)
    
    # Partial Correlation
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2)) + 1e-9
    rho_cond = (r_xy - r_xz * r_yz) / denom
    rho_cond = np.clip(rho_cond, -0.9999, 0.9999)
    
    mi_cond = -0.5 * np.log(1 - rho_cond**2)
    return float(mi_cond)

def compute_transfer_entropy_proxy(x: np.ndarray, y: np.ndarray, lag: int = 1) -> float:
    """
    Proxy for Transfer Entropy TE(X -> Y).
    Approximated as Granger Causality proxy: I(Y_t; X_{t-lag} | Y_{t-lag})
    """
    # X cause, Y effect
    # Shift X by lag (past)
    x_past = x[:-lag]
    y_past = y[:-lag]
    y_now = y[lag:]
    
    return compute_conditional_mi(y_now, x_past, y_past)

def compute_a1_metrics(log_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Computes A1 Hardening Metrics.
    Args:
        log_data: Dict with 'x_int', 'sensory', 'active', 'step'
    """
    # 1. State Independence Check
    # Is Internal shielded from External (Sensory) by Blanket?
    # Wait, External -> Sensory -> Blanket.
    # A1 says: External (X_ext) _||_ Internal (X_int) | Blanket (X_b)
    # Our data:
    #   Sensory (u_env) IS optimal proxy for X_ext.
    #   Blanket = [Sensory, Active].
    #   If we condition on Blanket, we condition on Sensory.
    #   I(Int; Sensory | Blanket) is trivial 0 because Blanket contains Sensory?
    #   
    #   Let's check the strict definition:
    #   I(Int; Ext | Blanket) should be low.
    #   Here let's use a "Hidden External" if possible, or assume Sensory has noise.
    #   
    #   Better check: Directionality / Injection.
    #   If we inject 'External' directly into 'Internal' (FAIL case), then I(Int; Ext | Blanket) approaches I(Int; Ext) > 0.
    #   Wait, if Blanket includes Sensory, then Blanket perfectly predicts Sensory.
    #   So I(Int; Sensory | Blanket) is always 0 mathematically.
    #   
    #   We need a distinct "Far External" variable X_ext that causes Sensory but is not Sensory.
    #   Since we don't have that in logs usually, we look at "Active -> Sensory" loop closure?
    #   
    #   Alternative A1:
    #   "Information Flow Check":
    #   Int -> Active (High TE)
    #   Sensory -> Int (High TE)
    #   Active -> Int (Low TE directly? No, via Sensory).
    
    #   Let's implement Directionality Ratio:
    #   TE(Sensory -> Int) vs TE(Int -> Sensory)  (Should be high S->I, low I->S directly?)
    #   Actually Int predicts Sensory, so I->S might be high in specific lag.
    
    x_int = np.array(log_data['x_int'])
    sensory = np.array(log_data['sensory'])
    active = np.array(log_data['active'])
    
    # Flatten across batch/dim for scalar proxy
    # Or take first PC? Let's take norm or mean.
    xi_scalar = np.mean(x_int, axis=1)
    s_scalar = np.mean(sensory, axis=1)
    
    # TE(Sensory -> Int)
    te_s_i = compute_transfer_entropy_proxy(s_scalar, xi_scalar, lag=1)
    
    te_i_s = compute_transfer_entropy_proxy(xi_scalar, s_scalar, lag=1)
    
    # Autonomy: H(Int | Sensory)
    # Proxy: 1 - R^2(Int, Sensory)
    r_is, _ = stats.pearsonr(xi_scalar, s_scalar)
    autonomy = 1.0 - r_is**2 # Residual variance ratio
    
    return {
        "te_sen_int": te_s_i,
        "te_int_sen": te_i_s,
        "directionality": te_s_i / (te_i_s + 1e-9),
        "autonomy": autonomy
    }

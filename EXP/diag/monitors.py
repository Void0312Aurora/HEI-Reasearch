import numpy as np
from typing import List, Callable, Any

class STLMonitor:
    """
    Quantitative Verification of Signal Temporal Logic (STL).
    Computes Robustness Degree rho(trace, phi).
    rho > 0: SAT.
    rho < 0: UNSAT.
    Magnitude |rho|: Margin of satisfaction/violation.
    """
    
    @staticmethod
    def predicate(trace: List[Any], func: Callable[[Any], float]) -> float:
        """
        Evaluate predicate on single state (or list treated as vector?).
        Usually STL is defined on time series signal 'trace'.
        Here we define Robustness of 'PHI' on 'trace'.
        Primitive: Predicate mu(x) > 0.
        Robustness = mu(x).
        """
        # If trace is whole signal, we need time 't'.
        # STL syntax: (trace, t) |= phi.
        # We simplify: assume we check property at t=0 for whole trace? 
        # Or Global property.
        pass

# Recursive Robustness
def robustness_always(trace: List[Any], t_start: int, t_end: int, pred_func: Callable[[Any], float]) -> float:
    """
    Always[t_start, t_end] (Predicate).
    Robustness = min_{t in [t_start, t_end]} pred_func(trace[t])
    """
    vals = [pred_func(trace[t]) for t in range(t_start, min(t_end, len(trace)))]
    if not vals: return -1.0 # Empty interval fail?
    return float(np.min(vals))

def robustness_eventually(trace: List[Any], t_start: int, t_end: int, pred_func: Callable[[Any], float]) -> float:
    """
    Eventually[t_start, t_end] (Predicate).
    Robustness = max_{t in [t_start, t_end]} pred_func(trace[t])
    """
    vals = [pred_func(trace[t]) for t in range(t_start, min(t_end, len(trace)))]
    if not vals: return -1.0
    return float(np.max(vals))

# Logic Operators
def logic_and(rho1: float, rho2: float) -> float:
    return min(rho1, rho2)

def logic_or(rho1: float, rho2: float) -> float:
    return max(rho1, rho2)

def logic_not(rho: float) -> float:
    return -rho

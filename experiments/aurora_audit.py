
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN
from hei_n.inertia_n import RadialInertia, IdentityInertia
from hei_n.potential_n import HarmonicPriorN, PairwisePotentialN, CompositePotentialN
from hei_n.metrics_n import calculate_ultrametricity_score, calculate_pairwise_distances
from hei_n.lie_n import exp_so1n

# --- Potentials ---
def kernel_lennard_jones_soft(d):
    val = 2.0 * np.exp(-d / 0.2) - 2.0 * np.exp(-d / 0.8)
    return val

def d_kernel_lennard_jones_soft(d):
    val = -10.0 * np.exp(-d / 0.2) + 2.5 * np.exp(-d / 0.8)
    return val

# --- Helpers ---
def calc_mean_radius(x):
    # Radius = arccosh(x0)
    x0 = x[..., 0]
    r = np.arccosh(np.maximum(x0, 1.0))
    return float(np.mean(r))

def run_single_seed(seed, mode, dt=0.001, steps=2000):
    np.random.seed(seed)
    N = 50
    dim = 2
    
    # Setup
    prior = HarmonicPriorN(k=0.5)
    pairwise = PairwisePotentialN(kernel_lennard_jones_soft, d_kernel_lennard_jones_soft)
    oracle = CompositePotentialN([prior, pairwise])
    
    # Mode Logic
    if mode == 'identity_small_dt':
        inertia = IdentityInertia()
        gamma = 0.5
        # dt passed as arg (expect 1e-4 or 1e-5)
    elif mode == 'no_contact':
        inertia = RadialInertia(alpha=1.0)
        gamma = 0.0
    else: # baseline
        inertia = RadialInertia(alpha=1.0)
        gamma = 0.5
        
    config = ContactConfigN(dt=dt, gamma=gamma)
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    # Init
    G_init = np.zeros((N, dim+1, dim+1)) 
    M_init = np.zeros((N, dim+1, dim+1))
    
    for i in range(N):
        G_init[i] = np.eye(dim+1)
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v) * np.random.uniform(0, 1.0)
        M_boost = np.zeros((dim+1, dim+1))
        M_boost[0, 1:] = v
        M_boost[1:, 0] = v
        G_init[i] = exp_so1n(M_boost[np.newaxis], dt=1.0)[0]
        
    state = ContactStateN(G=G_init, M=M_init, z=0.0)
    
    history = {'T': [], 'V': [], 'E': [], 'R': [], 'mean_rad': []}
    
    # Run
    # Catch NaN errors
    try:
        for i in range(steps):
            state = integrator.step(state)
            
            # Log every 10 steps for detail
            if i % 10 == 0:
                T = inertia.kinetic_energy(state.M, state.x)
                V = oracle.potential(state.x)
                E = T + V
                # R = 0.5 * gamma * <xi, I xi> = gamma * T
                R = gamma * T
                rad = calc_mean_radius(state.x)
                
                if np.isnan(E):
                    raise ValueError("NaN Detected")
                
                history['T'].append(float(T))
                history['V'].append(float(V))
                history['E'].append(float(E))
                history['R'].append(float(R))
                history['mean_rad'].append(rad)
                
                if i == steps: break

    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'steps_survived': i
        }
    
    # Final Metrics
    ultra = calculate_ultrametricity_score(state.x)
    return {
        'status': 'success',
        'final_E': history['E'][-1],
        'final_rad': history['mean_rad'][-1],
        'ultra': ultra,
        'history': history
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-identity', action='store_true')
    parser.add_argument('--audit-energy', action='store_true')
    args = parser.parse_args()
    
    results = {}
    
    # 1. Stability Check (Identity + Small DT)
    if args.check_identity:
        print("--- Checking Identity Inertia Stability ---")
        # Try decreasing DT
        dts = [1e-3, 1e-4, 1e-5]
        for dt_val in dts:
            print(f"Running Identity with dt={dt_val}...")
            res = run_single_seed(42, 'identity_small_dt', dt=dt_val, steps=1000)
            if res['status'] == 'failed':
                print(f"  FAILED at step {res['steps_survived']}")
            else:
                print(f"  SUCCESS! Final E={res['final_E']:.2f}")
            results[f'identity_dt_{dt_val}'] = res
            
    # 2. Energy & Artifact Audit (10 seeds)
    if args.audit_energy:
        print("--- Auditing Energy & Artifacts (10 Seeds) ---")
        seeds = range(42, 52)
        baseline_stats = []
        nocontact_stats = []
        
        for s in seeds:
            print(f"Seed {s}...")
            # Baseline
            res_b = run_single_seed(s, 'baseline', dt=1e-3)
            # No Contact
            res_nc = run_single_seed(s, 'no_contact', dt=1e-3)
            
            baseline_stats.append(res_b)
            nocontact_stats.append(res_nc)
            
        results['baseline_10'] = baseline_stats
        results['nocontact_10'] = nocontact_stats
        
        # Analyze Artifacts
        # Need to compare Ultrametricity at "Similar Radius"
        # Since we can't control radius easily, we just plot/correlate
        # Radius vs Ultra for all 20 runs.
        
        # Extract data for plotting
        radii = [r['final_rad'] for r in baseline_stats + nocontact_stats]
        ultras = [r['ultra'] for r in baseline_stats + nocontact_stats]
        labels = ['Baseline']*10 + ['NoContact']*10
        
        # Quick Correlation
        correlation = np.corrcoef(radii, ultras)[0, 1]
        results['radius_ultra_corr'] = correlation
        print(f"Radius-Ultrametric Correlation: {correlation:.4f}")
        # If negative (Larger Radius -> Lower Ultra Violation), then it's an artifact.
        
        # Save detailed logs for one seed (e.g. 42) to plot trajectories
        with open("audit_logs.json", "w") as f:
            json.dump(results, f, indent=2)
            
    print("Audit Complete.")

if __name__ == "__main__":
    main()

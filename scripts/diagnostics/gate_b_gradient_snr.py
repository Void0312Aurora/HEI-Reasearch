
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.sparse_potential_n import NegativeSamplingPotential
from hei_n.potential_n import PairwisePotentialN

def kernel_repulsion(d):
    # Standard repulsive kernel: exp(-d)
    return np.exp(-d)

def d_kernel_repulsion(d):
    return -np.exp(-d)

def test_gradient_snr(N=1000, dim=5):
    print(f"--- Gate B: Gradient SNR Check (N={N}, Dim={dim}) ---")
    
    np.random.seed(42)
    # Random points on Hyperboloid (Approx by normalizing Gaussian in Mink)
    # For test, just use random points in Poincare disk projected to Hyperboloid
    # Or just raw G matrices?
    # SparsePotential expects 'x' (N, dim+1).
    
    # Generate x on H^n
    x_raw = np.random.randn(N, dim+1)
    x_raw[:, 0] = np.sqrt(np.sum(x_raw[:, 1:]**2, axis=1) + 1.0) # Ensure on sheet
    # Norm check (Minkowski)
    # -x0^2 + x1^2... = -1? 
    # -(x0)^2 + r^2 = -(r^2+1) + r^2 = -1. Correct.
    
    # 1. Compute Ground Truth (Dense)
    print("Computing Dense Gradient (Ground Truth)...")
    # We can instantiate a dummy potential or just compute manually O(N^2)
    # dense_pot = PairwisePotentialN(...) # Takes pairs
    # Let's do manual broadcast to be sure.
    
    # Compute all distances
    # inner = -x0y0 + x1y1 ...
    mink = np.ones(dim+1); mink[0] = -1.0
    
    x_mink = x_raw * mink # (N, d)
    inner = x_mink @ x_raw.T # (N, N)
    # dist = arccosh(-inner)
    val = np.maximum(-inner, 1.0 + 1e-7)
    dists = np.arccosh(val)
    
    # Potential V = sum exp(-d)
    # Force direction?
    # Grad_u exp(-d) = -exp(-d) * Grad_u(d)
    # Grad_u d(u, v) ... complicated on Hn.
    # Let's rely on the NegativeSamplingPotential implementation with k=N to check correctness?
    # Or just trust that sparse_potential.gradient is implemented same as dense logic but sampled.
    # To act as Ground Truth, we sum the sparse_logic over ALL pairs.
    
    # Actually, let's use the NegativeSamplingPotential logic but force it to sample ALL pairs?
    # No, that class is built for sampling.
    
    # Let's write a simple dense gradient function here.
    grad_dense = np.zeros_like(x_raw)
    
    # grad_total = sum_v (-exp(-d) * grad_d)
    # grad_d(u, v) = (v + u cosh d) / sinh d ? No.
    # formula: grad_u d(u,v) = (v - u cosh d) / sinh d (if metric sign convention matches)
    # Implementation in sparse_potential says:
    # S = (dv / denom) * rescale
    # Fu = -S * (v_mink - u_mink * inner) ? No, see code.
    
    # Let's use the exact same math as sparse_potential to ensure we measure SAMPLING error, not formula mismatch.
    
    # Re-implement dense loop using same math
    for i in range(N):
        u = x_raw[i]
        force_i = np.zeros_like(u)
        for j in range(N):
            if i == j: continue
            v = x_raw[j]
            
            # Inner product
            uv = np.sum(u * v * mink)
            val = max(-uv, 1.0 + 1e-7)
            d_val = np.arccosh(val)
            denom = np.sqrt(val**2 - 1.0)
            denom = max(denom, 1e-7)
            
            # Kernel deriv
            dv = d_kernel_repulsion(d_val)
            scalar = dv / denom
            
            # J = mink
            Ju = u * mink
            Jv = v * mink
            
            # Gradient term
            # In code: Fu = -S * Jxv ... wait.
            # dense gradient is sum of pairs.
            # The sparse code does:
            # Jxv = xv * J
            # Fu = -S * Jxv
            # This looks like derivative wrt u is proportional to J*v?
            # d(-inner)/du = -J*v.
            # d(d)/d(-inner) = 1/sqrt...
            # d(V)/d(d) = dv
            # So dV/du = dv * (1/denom) * (-J*v).
            # Yes, matches.
            
            term = scalar * (-Jv)
            force_i += term
            
        grad_dense[i] = force_i
        
    norm_dense = np.linalg.norm(grad_dense)
    print(f"Dense Gradient Norm: {norm_dense:.4f}")
    
    # 2. Test Sparse Configurations
    k_values = [5, 10, 20, 50]
    results = {}
    
    for k in k_values:
        print(f"\nTesting k={k}...")
        # Scale factor theoretical approx: (N-1) / k ? No, since we sample k pairs total?
        # Num_neg is per particle? "num_neg" in class.
        # If class samples k pairs per particle, scaling is (N-1)/k.
        # Let's verify NegativeSamplingPotential implementation.
        # If it samples k * N pairs TOTAL, or k pairs PER row?
        # Usually "Negative Sampling" implies per positive? Here we have no positives (all repulsive).
        # We likely sample k * N pairs total in one batch?
        # Let's assume prediction: Scale = (N-1)/k * 0.5 (because symmetric?) 
        # Actually we calibrate it.
        
        # We run M iterations to average out noise for "Expected Gradient" check?
        # NO. We want to know the SNR of a SINGLE step.
        # But single step sparse gradient is inherently noisy.
        # We first check BIAS (Cosine of average) and then VARIANCE (Noise).
        
        # Check 1: Directional Match (Cosine) of a SINGLE step?
        # It will be noisy.
        # Let's compute average of 100 steps to check Bias.
        
        sparse_pot = NegativeSamplingPotential(kernel_repulsion, d_kernel_repulsion, num_neg=k, rescale=1.0)
        
        # A. Determine Scaling Factor
        # Run 50 steps
        sum_sparse = np.zeros_like(grad_dense)
        for _ in range(50):
            sum_sparse += sparse_pot.gradient(x_raw)
        avg_sparse = sum_sparse / 50.0
        
        norm_avg_sparse = np.linalg.norm(avg_sparse)
        scaling_factor = norm_dense / norm_avg_sparse
        print(f"  Calibrated Scale: {scaling_factor:.2f}")
        
        # B. Check Direction (Bias)
        cos_sim = np.sum(grad_dense * avg_sparse) / (norm_dense * norm_avg_sparse)
        print(f"  Direction Cosine (Bias Check): {cos_sim:.5f}")
        
        # C. Check SNR (Single Step fidelity)
        # Gradient with proper scaling
        sparse_pot.rescale = scaling_factor
        g_single = sparse_pot.gradient(x_raw)
        
        # Error Vector = g_single - g_dense
        # Relative Error = Norm(Error) / Norm(Dense)
        err = np.linalg.norm(g_single - grad_dense) / norm_dense
        print(f"  Single Step Rel Error: {err:.4f}")
        
        # SNR = Power(Signal) / Power(Noise)
        # Signal = g_dense
        # Noise = g_single - g_dense
        snr = 20 * np.log10(norm_dense / np.linalg.norm(g_single - grad_dense))
        print(f"  SNR: {snr:.2f} dB")
        
        results[k] = {'cosine': cos_sim, 'error': err, 'snr': snr}

    # Conclusion
    best_k = 5
    print("\n--- Summary ---")
    print(f"{'k':<5} {'Cosine':<10} {'RelErr':<10} {'SNR(dB)':<10}")
    for k in k_values:
        r = results[k]
        print(f"{k:<5} {r['cosine']:.5f}    {r['error']:.4f}      {r['snr']:.2f}")
        if r['cosine'] > 0.99 and r['error'] < 1.5: # Error 1.0 means noise same mag as signal?
             pass 

    print("\nRecommendation:")
    if results[5]['cosine'] > 0.95:
        print("PASS: k=5 is accurate enough in direction.")
    else:
        print("FAIL: k=5 direction is poor.")

if __name__ == "__main__":
    test_gradient_snr()

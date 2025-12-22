
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from hei_n.contact_integrator_n import ContactIntegratorN, ContactStateN, ContactConfigN
from hei_n.inertia_n import RadialInertia
from hei_n.potential_n import PairwisePotentialN, HarmonicPriorN, CompositePotentialN
from hei_n.sparse_potential_n import NegativeSamplingPotential
from hei_n.lie_n import exp_so1n

# --- Kernels ---
def kernel_repulsion(d):
    # Pure repulsion kernel
    # V = exp(-d / 0.5)
    return np.exp(-d / 0.5)

def d_kernel_repulsion(d):
    # dV/dd = -1/0.5 * exp
    val = -2.0 * np.exp(-d / 0.5)
    return val

# --- Test 1: Unbiased Check ---
def test_unbiased_gradient():
    print("--- Test 1: Unbiased Gradient Check ---")
    N = 40
    dim = 2
    np.random.seed(42)
    
    # Init State
    G = np.zeros((N, dim+1, dim+1))
    for i in range(N):
        G[i] = np.eye(dim+1)
        v = np.random.randn(dim) * 0.5
        M_boost = np.zeros((dim+1, dim+1))
        M_boost[0, 1:] = v
        M_boost[1:, 0] = v
        G[i] = exp_so1n(M_boost[np.newaxis], dt=1.0)[0]
    
    x = G[..., 0]
    
    # 1. Ground Truth (Dense)
    dense_pot = PairwisePotentialN(kernel_repulsion, d_kernel_repulsion)
    grad_dense = dense_pot.gradient(x)
    norm_dense = np.linalg.norm(grad_dense)
    print(f"Dense Gradient Norm: {norm_dense:.4f}")
    
    # 2. Sparse Sampling
    # Predict Scale: (N-1) / (2*k)?
    k = 5
    # Let's try scale=1.0 first to see raw magnitude
    sparse_pot_raw = NegativeSamplingPotential(kernel_repulsion, d_kernel_repulsion, num_neg=k, rescale=1.0, seed=42)
    
    # Average over samples
    iters = 200
    sum_grad = np.zeros_like(grad_dense)
    
    print(f"Sampling {iters} times (k={k})...")
    for i in range(iters):
        g = sparse_pot_raw.gradient(x)
        sum_grad += g
        
    avg_grad = sum_grad / iters
    norm_sparse = np.linalg.norm(avg_grad)
    print(f"Avg Sparse Gradient Norm (Scale=1.0): {norm_sparse:.4f}")
    
    # Calculate Observed Scale
    # We compare projection? Or just norms? 
    # Better: Ratio of norms? Or regression?
    # Simple: ratio of mean norms
    obs_scale = norm_dense / norm_sparse
    print(f"Observed Scaling Factor needed: {obs_scale:.4f}")
    
    # Predicted Scale
    # Dense interactions per node: N-1
    # Sparse interactions per node (avg): 2*k
    pred_scale = (N - 1) / (2 * k)
    print(f"Predicted Scaling Factor ((N-1)/2k): {pred_scale:.4f}")
    
    # Check Direction (Cosine Similarity)
    flat_dense = grad_dense.ravel()
    flat_sparse = avg_grad.ravel()
    cosine = np.dot(flat_dense, flat_sparse) / (np.linalg.norm(flat_dense) * np.linalg.norm(flat_sparse))
    print(f"Direction Cosine Similarity: {cosine:.4f}")
    
    if cosine > 0.9:
        print("PASS: Sparse Gradient is unbiased in direction.")
    else:
        print("FAIL: Sparse Gradient direction mismatch.")
        
    return obs_scale

# --- Test 2: Jitter Test ---
def test_jitter(scale_factor):
    print("\n--- Test 2: Jitter / Thermal Stability ---")
    N = 40
    dim = 2
    np.random.seed(42)
    
    # Create Stable Cluster (using Dense first to settle?)
    # Just init random, let it evolve with Sparse.
    
    prior = HarmonicPriorN(k=1.0) # Gravity
    # Sparse Repulsion
    k = 5
    sparse = NegativeSamplingPotential(kernel_repulsion, d_kernel_repulsion, num_neg=k, rescale=scale_factor)
    oracle = CompositePotentialN([prior, sparse])
    
    inertia = RadialInertia(alpha=1.0)
    config = ContactConfigN(dt=0.001, gamma=0.5) # Damping 0.5
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    # Init
    G = np.zeros((N, dim+1, dim+1))
    M = np.zeros((N, dim+1, dim+1))
    for i in range(N):
        G[i] = np.eye(dim+1)
        # Small velocity
        v = np.random.randn(dim) * 0.1
        M[i, 0, 1:] = v; M[i, 1:, 0] = v
        # Identity for G approx
    
    state = ContactStateN(G=G, M=M, z=0.0)
    
    temps = []
    
    steps = 1000
    for i in range(steps):
        state = integrator.step(state)
        # Temperature = Avg Kinetic Energy
        T = inertia.kinetic_energy(state.M, state.x) / N
        temps.append(T)
        
    mean_T = np.mean(temps[500:]) # Last half
    final_T = temps[-1]
    
    print(f"Final Temperature: {final_T:.6f}")
    print(f"Mean Stability T:  {mean_T:.6f}")
    
    # Criteria: Does it heat up indefinitely?
    # Check trend in last half
    slope = np.polyfit(np.arange(500), temps[500:], 1)[0]
    print(f"Temp Drift Slope (last 500 steps): {slope:.2e}")
    
    plt.figure()
    plt.plot(temps)
    plt.title(f"Thermal Stability (Sparse Repulsion, Scale={scale_factor:.1f})")
    plt.xlabel("Step")
    plt.ylabel("Temperature (T/N)")
    plt.savefig("sparse_thermal_stability.png")
    print("Saved sparse_thermal_stability.png")

    if slope > 1e-6:
        print("WARNING: System is heating up (Artificial Heating).")
        print("  -> Retrying with High Damping (Gamma=2.0)...")
        # Retry logic
        config = ContactConfigN(dt=0.001, gamma=2.0)
        integrator = ContactIntegratorN(oracle, inertia, config)
        state = ContactStateN(G=G, M=M, z=0.0)
        temps = []
        for i in range(steps):
            state = integrator.step(state)
            T = inertia.kinetic_energy(state.M, state.x) / N
            temps.append(T)
        
        slope2 = np.polyfit(np.arange(500), temps[500:], 1)[0]
        print(f"  New Slope (Gamma=2.0): {slope2:.2e}")
        if slope2 < 1e-6:
            print("  PASS: High Damping stabilized the system.")
        else:
            print("  FAIL: Still heating even with Gamma=2.0.")
    else:
        print("PASS: System is thermally stable (Damping balances Stochaistic Noise).")

# --- Test 3: Thermostat Validation & Distortion ---
def test_thermostat(scale_factor):
    print("\n--- Test 3: Adaptive Thermostat Validation ---")
    N = 40
    dim = 2
    np.random.seed(42)
    
    # Setup similar to Jitter Test
    prior = HarmonicPriorN(k=1.0)
    sparse = NegativeSamplingPotential(kernel_repulsion, d_kernel_repulsion, num_neg=5, rescale=scale_factor)
    oracle = CompositePotentialN([prior, sparse])
    inertia = RadialInertia(alpha=1.0)
    
    # Target Temp: Let's pick a reasonable value, e.g., 0.5
    T_target = 0.5
    # Use Thermostat with fast tau
    config = ContactConfigN(dt=0.001, gamma=0.5, target_temp=T_target, thermostat_tau=0.05)
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    # Init
    G = np.zeros((N, dim+1, dim+1))
    M = np.zeros((N, dim+1, dim+1))
    for i in range(N):
        G[i] = np.eye(dim+1)
        v = np.random.randn(dim) * 0.1
        M[i, 0, 1:] = v; M[i, 1:, 0] = v
        
    state = ContactStateN(G=G, M=M, z=0.0)
    
    temps = []
    
    steps = 5000
    for i in range(steps):
        state = integrator.step(state)
        T = inertia.kinetic_energy(state.M, state.x) / N
        temps.append(T)
        
    # Analysis
    temps = np.array(temps[2000:]) # Discard equilibration (2000 steps)
    mean_T = np.mean(temps)
    std_T = np.std(temps)
    slope = np.polyfit(np.arange(len(temps)), temps, 1)[0]
    
    print(f"Target Temp: {T_target}")
    print(f"Mean  Temp:  {mean_T:.6f} (Error: {mean_T-T_target:.6f})")
    print(f"Temp Slope:  {slope:.2e}")
    
    # Distribution Distortion Check
    # Canonical Ensemble Prediction for Fluctuations
    # sigma_T / T = sqrt(2 / (d * N)) for ideal gas? 
    # Or sigma_E^2 = kB * T^2 * Cv?
    # For independent particles in 2D (dim=2):
    # KE per particle is chi-square with 2 degrees of freedom (Exponential distribution).
    # Mean(KE) = dim/2 * kT? Here T is defined as KE/N.
    # If units are such that <T> = T_target.
    # Variance of T = Variance(sum(ke_i)/N) = 1/N^2 * sum(Var(ke_i))
    # Var(ke_i) for 2D is T^2? (Exponential has mean=lambda, var=lambda^2).
    # So Var(T) = 1/N^2 * N * T^2 = T^2 / N.
    # Sigma_T should be T / sqrt(N).
    
    expected_std = T_target / np.sqrt(N)
    
    print(f"Observed Std(T): {std_T:.6f}")
    print(f"Expected Std(T): {expected_std:.6f} (Canonical Prediction T/sqrt(N))")
    
    distortion_ratio = std_T / expected_std
    print(f"Distortion Ratio: {distortion_ratio:.4f}")
    
    plt.figure()
    plt.plot(temps, label='T(t)')
    plt.axhline(T_target, color='r', linestyle='--', label='Target')
    plt.title(f"Thermostat Performance (Distortion={distortion_ratio:.2f})")
    plt.legend()
    plt.savefig("thermostat_check.png")
    
    # Pass Criteria
    stable = abs(slope) < 1e-5
    # Distortion: Berendsen usually has Ratio < 1 (suppressed fluctuations).
    # We just report it.
    
    if stable:
        print("PASS: Thermostat stabilized the temperature.")
    else:
        print("FAIL: Temperature still drifting.")

    return distortion_ratio

if __name__ == "__main__":
    factor = test_unbiased_gradient()
    # test_jitter(factor) # Skip old jitter test
    test_thermostat(factor)

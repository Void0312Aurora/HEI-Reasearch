
import torch
import numpy as np
import sys
import os
import networkx as nx
import argparse
import time
import pickle
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.torch_core import minkowski_metric_torch
from hei_n.integrator_torch import ContactIntegratorTorch, ContactConfigTorch, ContactStateTorch, IdentityInertiaTorch
from hei_n.potentials_torch import (
    SpringAttractionTorch, 
    SparseEdgePotentialTorch, 
    LogCoshRepulsionTorch, 
    NegativeSamplingPotentialTorch, 
    HarmonicPriorTorch, 
    CompositePotentialTorch,
    RadiusAnchorPotentialTorch,
    ShortRangeGatedRepulsionTorch
)
from hei_n.datasets import load_dataset, load_cilin_dataset, load_semantic_edges




def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}", flush=True)
    
    # 1. Load Data
    if args.dataset == 'cilin':
        nodes, edges, depths, root_idx = load_cilin_dataset(limit=args.limit)
    else:
        nodes, edges, depths, root_idx = load_dataset(limit=args.limit)
        
    N = len(nodes)
    dim = args.dim
    
    # 2. Physics Init (CPU -> GPU)
    print(f"Initializing {N} particles...", flush=True)
    np.random.seed(args.seed)
    
    G_np = np.zeros((N, dim+1, dim+1), dtype=np.float32)
    M_np = np.zeros((N, dim+1, dim+1), dtype=np.float32)
    
    scale_init = 0.05
    scale_init = 0.05
    
    # Init Logic: Resume or Random
    save_path = f"checkpoints/aurora_base_gpu_{args.dataset}_full.pkl"
    loaded_checkpoint = False
    
    if args.resume and os.path.exists(save_path):
        print(f"Resuming from {save_path}...", flush=True)
        try:
            with open(save_path, 'rb') as f:
                ckpt_data = pickle.load(f)
            
            # Load G and M
            # Ensure shape matches N
            G_loaded = ckpt_data['G']
            M_loaded = ckpt_data['M']
            
            if G_loaded.shape[0] != N:
                print(f"Error: Checkpoint N={G_loaded.shape[0]} does not match Dataset N={N}. Cannot resume.")
                # Fallback to random? Or exit? Exit is safer.
                sys.exit(1)
                
            G_np = G_loaded
            M_np = M_loaded
            loaded_checkpoint = True
            print("  State loaded successfully.")
            
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            print("  Falling back to Random Initialization.")
            
    if not loaded_checkpoint:
        print("Performing Random Hyperbolic Initialization...")
        for i in range(N):
            d_i = depths[i]
            r = scale_init * d_i + np.random.uniform(0, 0.01)
            
            dir_vec = np.random.randn(dim)
            dir_vec /= np.linalg.norm(dir_vec) + 1e-9
            
            ch = np.cosh(r); sh = np.sinh(r)
            B = np.eye(dim+1)
            B[0,0] = ch
            B[0,1:] = sh * dir_vec
            B[1:,0] = sh * dir_vec
            B[1:,1:] = np.eye(dim) + (ch - 1) * np.outer(dir_vec, dir_vec)
            
            G_np[i] = B
            v = np.random.randn(dim) * 0.01
            M_np[i, 0, 1:] = v; M_np[i, 1:, 0] = v
        
    # Transfer to GPU
    G = torch.tensor(G_np, device=device)
    M = torch.tensor(M_np, device=device)
    z = torch.tensor(0.0, device=device)
    edges_gpu = torch.tensor(edges, device=device, dtype=torch.long)
    
    state = ContactStateTorch(G=G, M=M, z=z)
    
    # 3. Potentials (GPU)
    print("Configuring GPU Potentials...", flush=True)
    
    # Kernel for Skeleton (Base) - SOFTENED to allow PMI to dominate local neighborhoods
    k_attract_skel = SpringAttractionTorch(k=0.3)  # Reduced from 1.0 to 0.3
    attract_skel = SparseEdgePotentialTorch(edges_gpu, k_attract_skel)
    
    potentials = [attract_skel]
    
    # Semantic Potentials (Phase II)
    k_attract_pmi = None
    k_attract_def = None
    
    if args.semantic:
        edges_pmi, edges_def = load_semantic_edges(None) # IDs assumed stable
        
        if edges_pmi is not None:
            edges_pmi_gpu = torch.tensor(edges_pmi, device=device, dtype=torch.long)
            k_attract_pmi = SpringAttractionTorch(k=0.0) # Start 0 (Calibration)
            attract_pmi = SparseEdgePotentialTorch(edges_pmi_gpu, k_attract_pmi)
            potentials.append(attract_pmi)
            print("Added PMI Potential (Init k=0.0)")
            
        if edges_def is not None:
            edges_def_gpu = torch.tensor(edges_def, device=device, dtype=torch.long)
            k_attract_def = SpringAttractionTorch(k=0.0) # Start 0 (Calibration)
            attract_def = SparseEdgePotentialTorch(edges_def_gpu, k_attract_def)
            potentials.append(attract_def)
            print("Added Definition Potential (Init k=0.0)")

    # Repulsion: Short-range gated (per temp-24.md) or legacy LogCosh
    if args.volume_control:
        k_repulse = ShortRangeGatedRepulsionTorch(A=args.repulsion_a, epsilon=args.repulsion_eps, sigma=0.2)
        print(f"Repulsion: Short-Range Gated (A={args.repulsion_a}, ε={args.repulsion_eps}, σ=0.2)")
    else:
        k_repulse = LogCoshRepulsionTorch(sigma=1.0, A=args.repulsion_a)
        print(f"Repulsion: LogCosh (A={args.repulsion_a}, sigma=1.0)")
    repulse = NegativeSamplingPotentialTorch(k_repulse, num_neg=10, rescale=1.0)
    potentials.append(repulse)
    
    trap = HarmonicPriorTorch(k=0.05)
    potentials.append(trap)
    
    # Radius Anchor Potential (Volume Control per temp-24.md)
    if args.volume_control:
        # Target radii: (0.5 + depth * 0.25) * scale
        base_radii = 0.5 + depths * 0.25
        target_radii = torch.tensor(base_radii * args.radius_scale, device=device, dtype=torch.float32)
        # Apply --anchor-lambda (default 1.0)
        radius_anchor = RadiusAnchorPotentialTorch(target_radii, lamb=args.anchor_lambda)
        potentials.append(radius_anchor)
        print(f"Added Radius Anchor (λ={args.anchor_lambda}, scale={args.radius_scale}, targets=[{target_radii.min():.2f}, {target_radii.max():.2f}])")
    
    oracle = CompositePotentialTorch(potentials)
    inertia = IdentityInertiaTorch()
    
    # 4. Integrator
    config = ContactConfigTorch(
        dt=args.dt,
        gamma=2.0,
        target_temp=0.1,
        thermostat_tau=5.0,
        fixed_point_iters=args.solver_iters, # Configurable per temp-27.md
        solver_mixing=0.5,
        torque_clip=50.0,
        renorm_interval=50,
        adaptive=True,
        tol_disp=0.2,
        device=device
    )
    
    integrator = ContactIntegratorTorch(oracle, inertia, config)
    
    # 5. Loop
    print(f"Starting GPU Training: {args.steps} steps...", flush=True)
    start_time = time.time()
    
    # Torch benchmark warm-up
    for _ in range(5):
        pass
        
    residuals = []
    step_times = []
    renorms = []
    
    print(f"Starting GPU Training: {args.steps} steps...", flush=True)
    
    # 5. De-collapse (Stage 0) if requested
    if args.decollapse:
        print("PERFORMING DE-COLLAPSE (One-shot Rescaling)...")
        # Current x -> radii
        x = state.x # Use state.x
        x0 = x[:, 0]
        x0_safe = torch.clamp(x0, min=1.0 + 1e-7)
        current_radii = torch.acosh(x0_safe)
        
        # Target radii (already calculated as target_radii or derived)
        if args.volume_control:
            # Use the target_radii we defined
             targets = target_radii
        else:
            # Fallback if vol control not enabled (though unlikely combined)
            base_val = 0.5 + depths * 0.25
            targets = torch.tensor(base_val * args.radius_scale, device=device, dtype=torch.float32)
            
        # Rescale Factor: r_new / r_old
        # But we need to scale spatial components x_{1..}
        # x_{new} = x_{old} * factor? No.
        # x0_new = cosh(r_new), |x_spatial_new| = sinh(r_new)
        # |x_spatial_old| = sinh(r_old)
        # factor = sinh(r_new) / sinh(r_old)
        
        r_new = torch.max(current_radii, targets) # Don't shrink, only expand
        
        sinh_old = torch.sqrt(x0_safe**2 - 1.0)
        sinh_new = torch.sinh(r_new)
        
        scale_factor = sinh_new / (sinh_old + 1e-9)
        
        # Apply scaling
        x_new = x.clone()
        x_new[:, 0] = torch.cosh(r_new)
        x_new[:, 1:] = x[:, 1:] * scale_factor.unsqueeze(-1)
        
        # Update State
        state.x = x_new
        
        # Robust G-Frame Repair (Minkowski Gram-Schmidt)
        # G[:, :, 0] must be x.
        # G[:, :, 1:] must be orthogonal to x and each other, and normalized.
        print("  Repairing G-Frame (Gram-Schmidt)...")
        
        # Set col 0
        state.G[:, :, 0] = x_new
        
        # Minkowski Inner Product helper
        def minkowski_dot(u, v):
            # u, v: (N, D)
            return -u[:, 0]*v[:, 0] + torch.sum(u[:, 1:]*v[:, 1:], dim=-1)
            
        # Re-orthonormalize columns 1..D
        dim = x.shape[-1]
        for i in range(1, dim):
            # Start with existing column as candidate basis
            # (Assuming old correlation structure is better than random)
            v = state.G[:, :, i]
            
            # Project out previous columns
            for j in range(i):
                basis = state.G[:, :, j]
                # Coeff = <v, basis> / <basis, basis>
                # <basis, basis> is -1 for j=0, +1 for j>0
                norm_sq = -1.0 if j == 0 else 1.0
                coeff = minkowski_dot(v, basis) / norm_sq
                v = v - coeff.unsqueeze(-1) * basis
                
            # Normalize
            # <v, v> should be positive
            v_norm_sq = minkowski_dot(v, v)
            # Clamp for safety
            v_norm_sq = torch.clamp(v_norm_sq, min=1e-9)
            v = v / torch.sqrt(v_norm_sq).unsqueeze(-1)
            
            state.G[:, :, i] = v
            
        print(f"  Radii Rescaled: Mean {current_radii.mean().item():.3f} -> {r_new.mean().item():.3f}")
        print("  G-Frame Repaired.")
    
    # Run Simulation
    start_time = time.time()
    
    # Ramp-up config
    if args.ramp_volume and args.volume_control:
        lambda_start, lambda_end = 2.0, 10.0
        A_start, A_end = 1.0, 5.0
        eps_start, eps_end = 0.2, 0.5
        ramp_steps = 1500
    
    for step in range(args.steps):
        # Ramp-up Logic (Stage 1)
        if args.ramp_volume and args.volume_control and step < ramp_steps:
             progress = step / ramp_steps
             current_lambda = lambda_start + (lambda_end - lambda_start) * progress
             current_A = A_start + (A_end - A_start) * progress
             current_eps = eps_start + (eps_end - eps_start) * progress
             
             # Update Potentials
             radius_anchor.lamb = current_lambda
             k_repulse.A = current_A
             k_repulse.epsilon = current_eps
             
             if step % 100 == 0:
                 print(f"  [Ramp] Step {step}: λ={current_lambda:.2f}, A={current_A:.2f}, ε={current_eps:.2f}")

        # Update Annealing Weights
        if config.freeze_radius:
            # Stage 2 (if we were using it, but volume control disables freeze)
             pass
        else:
             # Standard annealing
             anneal_progress = step / args.steps
             pass # Logic handles itself below
           
        # Annealing Schedule (Phase II)
        if args.semantic:
            # Stage 0: 0-20% -> k=0
            # Stage 1: 20-40% -> k=0.01
            # Stage 2: 40-60% -> k=0.03
            # Stage 3: 60-100% -> k=0.05
            progress = step / args.steps
            target_k = 0.0
            stage = 0
            
            # 4-Stage Stepped Ramp-up for Full Scale Controlled Rollout
            # Two-Stage Radius-Decoupled Schedule
            # Stage 1: Skeleton Only (Radius Evolving) -> Establish Hierarchy
            # Stage 2: PMI Only (Radius Frozen) -> Alignment without Expansion
            
            if progress < 0.50 and not args.force_stage2:
                # Stage 1: Skeleton Dominant
                target_k_skel = 1.0
                target_k_pmi = 0.0 # No PMI yet
                freeze_radius = False
                stage_name = "Stage 1: Hierarchy (Skel=1.0, PMI=0.0, Radius=Free)"
            else:
                # Stage 2: PMI Alignment
                # If volume_control: use soft radius anchor instead of freeze_radius (temp-24.md)
                target_k_skel = 0.0 # Turn off skeleton
                target_k_pmi = args.pmi_target # Configurable target
                
                # PMI Ramp Logic (if active)
                if args.pmi_ramp_steps > 0 and step < args.pmi_ramp_steps:
                    pmi_progress = step / args.pmi_ramp_steps
                    current_pmi = args.pmi_start + (args.pmi_target - args.pmi_start) * pmi_progress
                    target_k_pmi = current_pmi
                    if step % 100 == 0:
                        print(f"  [PMI Ramp] Step {step}: PMI {current_pmi:.3f} (Target {args.pmi_target})")
                
                freeze_radius = not args.volume_control  # Disable hard freeze if using soft anchor
                stage_name = f"Stage 2: Alignment (Skel=0.0, PMI={target_k_pmi:.3f}, Radius={'Soft' if args.volume_control else 'Fixed'})"
            
            # Apply Config
            config.freeze_radius = freeze_radius
            k_attract_skel.k = target_k_skel
            if k_attract_pmi: k_attract_pmi.k = target_k_pmi
            
            # Log stage change
            current_stage = 1 if progress < 0.5 else 2
            if step % args.log_interval == 0 and step > 0:
                 # Check for transition
                 prev_stage = 1 if (step - args.log_interval)/args.steps < 0.5 else 2
                 if current_stage != prev_stage:
                     print(f"\n>>> PHASE TRANSITION: {stage_name} <<<\n", flush=True)
            


        step_start = time.time()
        
        # Pin Root
        state.M[root_idx] = 0.0
        state = integrator.step(state)
        state.M[root_idx] = 0.0
        
        if device.type == 'cuda':
            torch.cuda.synchronize() # Wait for GPU
        step_end = time.time()
        step_dur = step_end - step_start
        
        # Collect Stats
        diag = state.diagnostics
        residuals.append(diag.get('solver_residual', 0.0))
        renorms.append(diag.get('renorm_magnitude', 0.0))
        step_times.append(step_dur * 1000.0) # ms
        
        if step % args.log_interval == 0:
            # Metrics (GPU -> CPU for print)
            with torch.no_grad():
                idx = torch.randint(0, N, (min(N, 1000),), device=device)
                x_sample = state.x[idx]
                radii = torch.acosh(torch.clamp(x_sample[:, 0], min=1.0))
                mean_R = torch.mean(radii).item()
            
            res = residuals[-1]
            renorm = renorms[-1]
            dt_used = diag.get('dt', args.dt)
            elapsed = time.time() - start_time
            
            print(f"Step {step}: T={elapsed:.1f}s | dt={dt_used:.1e} | R={mean_R:.2f} | Res={res:.1e} | Renorm={renorm:.1e} | {step_dur*1000:.1f}ms/step", flush=True)
            if args.semantic and step % (args.log_interval * 5) == 0:
                pmi_k = k_attract_pmi.k if k_attract_pmi else 0
                print(f"  [Anneal] Semantic Weights: {pmi_k:.3f}", flush=True)
            
    print("Training Complete.", flush=True)
    
    # Gate Analysis
    res_arr = np.array(residuals)
    time_arr = np.array(step_times)
    ren_arr = np.array(renorms)
    
    res_p99 = np.percentile(res_arr, 99)
    time_p99 = np.percentile(time_arr, 99)
    time_mean = np.mean(time_arr)
    ren_max = np.max(ren_arr)
    
    print("\n=== GATE CHECK REPORT ===")
    print(f"Metric\t\tValue\t\tThreshold\tStatus")
    print(f"Residual P99\t{res_p99:.2e}\t< 5.00e-03\t{'PASS' if res_p99 < 5e-3 else 'FAIL'}")
    print(f"Renorm Max\t{ren_max:.2e}\t< 1.00e-04\t{'PASS' if ren_max < 1e-4 else 'FAIL'}")
    print(f"Throughput P99\t{time_p99:.1f}ms\t< {2*time_mean:.1f}ms\t{'PASS' if time_p99 < 2*time_mean else 'WARN'}")
    print("=========================")
    
    # Save
    if args.save:
        save_path = f"checkpoints/aurora_base_gpu_{args.dataset}_{args.limit or 'full'}.pkl"
        depths_path = f"checkpoints/aurora_base_gpu_{args.dataset}_{args.limit or 'full'}_depths.npy"
        os.makedirs("checkpoints", exist_ok=True)
        # Convert to CPU numpy
        G_cpu = state.G.detach().cpu().numpy()
        M_cpu = state.M.detach().cpu().numpy()
        x_cpu = state.x.detach().cpu().numpy()  # Positions for Dialogue Probe
        
        with open(save_path, 'wb') as f:
            pickle.dump({'G': G_cpu, 'M': M_cpu, 'x': x_cpu, 'nodes': nodes, 'edges': edges}, f)
        print(f"Saved to {save_path}", flush=True)
        
        # Save depths for Semantic Audit
        np.save(depths_path, depths)
        print(f"Saved depths to {depths_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--dataset", type=str, default="openhow", choices=["openhow", "cilin"], help="Dataset to use")
    parser.add_argument("--semantic", action="store_true", help="Enable Phase II Semantic Edges")
    parser.add_argument("--repulsion-a", type=float, default=5.0, help="Repulsion amplitude A (default: 5.0)")
    parser.add_argument("--volume-control", action="store_true", help="Enable soft radius anchor + short-range gated repulsion (temp-24.md)")
    parser.add_argument("--radius-scale", type=float, default=1.0, help="Scale factor for target radii (default: 1.0)")
    parser.add_argument("--anchor-lambda", type=float, default=1.0, help="Base lambda for radius anchor (default: 1.0)")
    parser.add_argument("--decollapse", action="store_true", help="Perform one-shot radius de-collapse at startup")
    parser.add_argument("--ramp-volume", action="store_true", help="Ramp volume control parameters (A:1->5, Lambda:2->10)")
    parser.add_argument("--repulsion-eps", type=float, default=0.5, help="Epsilon for gated repulsion (default: 0.5)")
    parser.add_argument("--solver-iters", type=int, default=8, help="Fixed point iterations for solver (default: 8)")
    parser.add_argument("--pmi-target", type=float, default=0.5, help="Target PMI weight for Alignment Phase (default: 0.5)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (x, G, M)")
    parser.add_argument("--force-stage2", action="store_true", help="Force Stage 2 (Alignment) immediately (skip Stage 1)")
    parser.add_argument("--pmi-start", type=float, default=0.0, help="Starting PMI weight for Ramp (default: 0.0)")
    parser.add_argument("--pmi-ramp-steps", type=int, default=0, help="Steps to ramp PMI from start to target (default: 0)")
    
    args = parser.parse_args()
    train(args)

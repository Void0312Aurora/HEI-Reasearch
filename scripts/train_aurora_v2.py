"""
Train Aurora V2 (Clean Architecture).
=====================================

New entry point for training the Aurora Interaction Engine (HEI Phase II).
Uses the clean `src/aurora` package with decoupled data loading.

Usage:
    python scripts/train_aurora_v2.py --dataset cilin --limit 5000 --device cuda
"""

import sys
import os
import argparse
import torch
import numpy as np
import time
import pickle

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora import (
    AuroraDataset,
    PhysicsConfig,
    PhysicsState,
    ContactIntegrator,
    CompositePotential,
    SpringPotential,
    RadiusAnchorPotential,
    RepulsionPotential,
    GatedRepulsionPotential,
    SemanticTripletPotential,
    RadialInertia
)
from aurora.geometry import random_hyperbolic_init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cilin", choices=["cilin", "openhow"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--semantic_path", type=str, default=None, help="Path to str-based semantic edges pickle")
    
    # Staged Training Args
    parser.add_argument("--stage_ratio", type=float, default=0.5, help="Fraction of steps for Structure-Only phase")
    parser.add_argument("--k_struct", type=float, default=1.0)
    parser.add_argument("--k_sem", type=float, default=0.5)
    parser.add_argument("--k_rep", type=float, default=0.1, help="Repulsion Stiffness (A)")
    
    # Phase V Args
    parser.add_argument("--triplet", action="store_true", help="Use Triplet Margin Loss for Semantics")
    parser.add_argument("--checkpoint", type=str, default=None, help="Start from checkpoint (Phase V)")
    parser.add_argument("--num_candidates", type=int, default=10, help="Hard Negative Mining Candidates")
    parser.add_argument("--split", type=str, default="train", choices=["train", "all"], help="Semantic Edge Split")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed")
    parser.add_argument("--save_path", type=str, default="checkpoints/aurora_v2_final.pkl", help="Checkpoint Output Path")
    
    # Phase IX Args: Advanced Hard Mining
    parser.add_argument("--mining_mode", type=str, default="hard", choices=["hard", "semi-hard", "curriculum", "trusted", "global"],
                        help="Negative mining strategy: hard, semi-hard, curriculum, trusted, or global")
    parser.add_argument("--local_pool", action="store_true", help="Sample negatives from same radius band")
    parser.add_argument("--radius_tolerance", type=float, default=0.1, help="Radius tolerance for local pool")
    # Phase X Args: Dynamic Difficulty
    parser.add_argument("--curriculum", action="store_true", help="Enable dynamic Semi-Hard -> Hard curriculum")
    parser.add_argument("--margin_schedule", action="store_true", help="Enable dynamic margin increase")
    parser.add_argument("--bank_size", type=int, default=50000, help="Size of negative candidate bank (for dilution)")
    parser.add_argument("--soft_weight", type=float, default=0.0, help="Weight for Soft Positive attraction (Phase XVI)")
    parser.add_argument("--pseudo_edges_path", type=str, default=None, help="Path to harvested pseudo-edges (Phase XVII)")
    parser.add_argument("--pseudo_weight", type=float, default=0.1, help="Stiffness for pseudo-edges")
    parser.add_argument("--lamb", type=float, default=0.1, help="Radius anchor strength (volume control)")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        print(f"Setting Random Seed: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # Python random not strictly needed but good practice
        import random
        random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Aurora V2 Training on {device} (k_rep={args.k_rep}, triplet={args.triplet})")
    
    # 1. Load Data
    print("Loading Dataset...")
    ds = AuroraDataset(args.dataset, limit=args.limit)
    print(f"  Nodes: {ds.num_nodes}")
    print(f"  Struct Edges: {len(ds.edges_struct)}")
    
    # 2. Physics Init
    N = ds.num_nodes
    dim = 5
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        with open(args.checkpoint, 'rb') as f:
            ckpt = pickle.load(f)
        x_init = torch.tensor(ckpt['x'], device=device) # (N, dim)
        if x_init.shape[0] != N:
            raise ValueError(f"Checkpoint size {x_init.shape[0]} != Dataset size {N}")
        print("Checkpoint loaded.")
    else:
        x_init = random_hyperbolic_init(N, dim, scale=0.1, device=device)
        
    G = torch.zeros(N, dim, dim, device=device)
    G[..., 0] = x_init
    G[..., 1:] = torch.eye(dim, device=device).unsqueeze(0).repeat(N, 1, 1)[..., 1:]
    
    from aurora.geometry import renormalize_frame
    G, _ = renormalize_frame(G)
    
    M = torch.zeros(N, dim, dim, device=device)
    z = torch.zeros(N, device=device)
    
    state = PhysicsState(G=G, M=M, z=z)
    
    # 3. Potentials
    potentials = []
    
    # Structural
    edges_struct_t = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
    
    # If Fine-Tuning (Triplet), use Relaxed Structure by default
    effective_k_struct = args.k_struct * 0.1 if args.triplet else args.k_struct
    print(f"Structural Stiffness: {effective_k_struct}")
    
    pot_struct = SpringPotential(edges_struct_t, k=effective_k_struct, l0=0.0)
    potentials.append(pot_struct)
    
    # Semantic (Optional)
    pot_sem = None
    if args.semantic_path:
        sem_edges = ds.load_semantic_edges(args.semantic_path, split=args.split)
        if sem_edges:
            sem_indices = [(u, v) for u, v, w in sem_edges]
            sem_t = torch.tensor(sem_indices, dtype=torch.long, device=device)
            
            if args.triplet:
                print(f">>> Using SemanticTripletPotential (Margin=0.1, Candidates={args.num_candidates})")
                print(f"    Mining Mode: {args.mining_mode}, Local Pool: {args.local_pool}, Bank Size: {args.bank_size}")
                print(f"    Soft Positive Weight: {args.soft_weight}")
                # Assume k_sem is irrelevant for Triplet or use implicit 1.0
                # Triplet doesn't use k, but relies on grad magnitude.
                pot_sem = SemanticTripletPotential(sem_t, margin=0.1, num_candidates=args.num_candidates,
                                                     mining_mode=args.mining_mode, local_pool=args.local_pool,
                                                     radius_tolerance=args.radius_tolerance, bank_size=args.bank_size,
                                                     soft_weight=args.soft_weight)
            else:
                # Start with ZERO Stiffness (Stage A) unless checkpointed? 
                # If checkpointed, presumably we want meaningful k immediately?
                # Keeping logic simple: If checkpoint, start with k_sem immediately?
                # User can control via args. For Triplet run, we usually set steps low.
                init_k = args.k_sem if args.checkpoint else 0.0
                pot_sem = SpringPotential(sem_t, k=init_k) 
                
            potentials.append(pot_sem)
            print(f"  Added {len(sem_indices)} Semantic Edges.")

    # Phase XVII: Pseudo-Edges (Soft Positives Hardened)
    if args.pseudo_edges_path:
        print(f"Loading Pseudo-Edges from {args.pseudo_edges_path}...")
        with open(args.pseudo_edges_path, 'rb') as f:
            pseudo_edges_list = pickle.load(f)
        
        if pseudo_edges_list:
            pseudo_t = torch.tensor(pseudo_edges_list, dtype=torch.long, device=device)
            print(f">>> Adding {len(pseudo_edges_list)} Pseudo-Edges (Weight={args.pseudo_weight})")
            
            # Use SpringPotential (Pull Only)
            # pseudo_weight determines stiffness
            pot_pseudo = SpringPotential(pseudo_t, k=args.pseudo_weight, l0=0.0)
            potentials.append(pot_pseudo)
        else:
            print("Warning: Pseudo-edges file loaded but empty.")
            
    # Volume Control
    depths = torch.tensor(ds.depths, dtype=torch.float32, device=device)
    target_radii = 0.5 + depths * 0.5
    potentials.append(RadiusAnchorPotential(target_radii, lamb=args.lamb))
    
    # Repulsion (Gated / Short-Range)
    # Epsilon should be related to local density.
    # If mean r=0.69, typical dist is small.
    # Let's try epsilon=0.1
    potentials.append(GatedRepulsionPotential(A=args.k_rep, epsilon=0.1, num_neg=10))
    
    oracle = CompositePotential(potentials)
    
    # 4. Integrator
    inertia = RadialInertia(alpha=5.0)
    config = PhysicsConfig(dt=0.05, gamma=0.5, adaptive=True, solver_iters=10)
    integrator = ContactIntegrator(config, inertia)
 
    # 5. Loop
    print(f"Starting Simulation (Steps={args.steps}, StageRatio={args.stage_ratio})...")
    stage_step = int(args.steps * args.stage_ratio)
    
    start_t = time.time()
    for i in range(args.steps):
        # Phase XIII: Global Index Refresh (Every 2000 steps)
        if i > 0 and i % 2000 == 0 and pot_sem and args.mining_mode == "global":
            if hasattr(pot_sem, 'update_global_candidates'):
                 print(f">>> [Global] Refreshing Index (Step {i})...")
                 pot_sem.update_global_candidates(state.x, k=1000)
                 
        # Phase XVI: Soft Positive Mining (Every 500 steps)
        # We need update_global_candidates first if using soft mining (it depends on it)
        # If mining_mode is NOT global, we might still want global candidates for Soft Mining?
        # Let's enforce: If soft_weight > 0, run index update regardless of mode.
        if i % 500 == 0 and pot_sem and args.soft_weight > 0.0:
             if hasattr(pot_sem, 'update_global_candidates'):
                 print(f">>> [Phase XVI] Updating Soft Positives (Step {i})...")
                 # Ensure global candidates exist first
                 pot_sem.update_global_candidates(state.x, k=1000)
                 pot_sem.update_soft_positives(state.x, k=50)
 
        # Staged Training Logic
        if i == stage_step:
            print(f">>> [Stage B] Step {i}: Introducing Semantic Forces. Relaxing Structure.")
            # Relax Structure
            pot_struct.k = args.k_struct * 0.1
            # Enable Semantic
            if pot_sem:
                 pot_sem.k = args.k_sem
                 
        state = integrator.step(state, oracle)
        
        if i % 100 == 0:
            avg_r = torch.mean(torch.acosh(state.x[:, 0])).item()
            dt = state.diagnostics.get('dt', config.dt)
            print(f"Step {i}: R={avg_r:.2f}, dt={dt:.1e}, E={state.diagnostics.get('energy',0):.1f}")
            
            # Phase X/XI/XII: Dynamic Difficulty & Gating Logic
            if i > 0 and pot_sem is not None and hasattr(pot_sem, 'last_violation_rate'):
                v_rate = pot_sem.last_violation_rate
                print(f"    [Mining] Mode: {pot_sem.mining_mode}, V-Rate: {v_rate*100:.1f}%, Margin: {pot_sem.margin:.2f}")
                
                # Phase X: Dynamic Curriculum
                if args.curriculum and pot_sem.mining_mode == "curriculum":
                    # Logic: If V-Rate < 10%, decrease progress (more hard).
                    if v_rate < 0.10:
                        pot_sem.curriculum_progress = max(0.0, pot_sem.curriculum_progress - 0.1)
                        if pot_sem.curriculum_progress < 1.0:
                             print(f"    >>> Hardness Up! Curriculum Progress: {pot_sem.curriculum_progress:.1f}")
                        
                # Phase XII: Trusted Mining Gating (Closed Loop)
                if pot_sem.mining_mode == "trusted" or pot_sem.mining_mode == "global":
                     print(f"    [Trusted] Hard Ratio: {getattr(pot_sem, 'hard_ratio', 0.5):.2f} (Target V-Rate 5-25%)")
                     if v_rate > 0.25:
                         # Too hard/noisy -> Reduce Hard Ratio (More Random)
                         pot_sem.hard_ratio = max(0.1, pot_sem.hard_ratio - 0.1)
                         print(f"    >>> High V-Rate! Decreasing Hard Ratio -> {pot_sem.hard_ratio:.2f}")
                     elif v_rate < 0.05:
                         # Too easy -> Increase Hard Ratio (More Bank)
                         pot_sem.hard_ratio = min(0.9, pot_sem.hard_ratio + 0.1)
                         print(f"    >>> Low V-Rate! Increasing Hard Ratio -> {pot_sem.hard_ratio:.2f}")
                         
                # Dynamic Margin: Increase margin if V-Rate is low (even at full hardness)
                if args.margin_schedule:
                    # If V-Rate < 5% and we are already hard, bump margin
                    is_hard = (pot_sem.mining_mode == "hard") or \
                              (pot_sem.mining_mode == "curriculum" and pot_sem.curriculum_progress <= 0.1) or \
                              (pot_sem.mining_mode == "trusted" and getattr(pot_sem, 'hard_ratio', 0.0) >= 0.9)
                              
                    if v_rate < 0.05 and is_hard:
                        pot_sem.margin += 0.05
                        print(f"    >>> Margin Up! New Margin: {pot_sem.margin:.2f}")
            
    end_t = time.time()
    print(f"Done. {args.steps} steps in {end_t - start_t:.1f}s")
    
    # Save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    save_path = args.save_path
    
    # Checkpoint Schema v1.0
    import datetime
    save_data = {
        'version': '1.0',  # Checkpoint schema version
        'timestamp': datetime.datetime.now().isoformat(),
        'x': state.x.detach().cpu().numpy(),
        # 'v' is implicitly stored in M (angular momentum), not separately needed for eval.
        'vocab': ds.vocab.word_to_id, # Save vocab mapping!
        'nodes': ds.nodes,
        'config': {
            'stage_ratio': args.stage_ratio,
            'k_struct': args.k_struct,
            'k_sem': args.k_sem,
            'k_rep': args.k_rep,
            'triplet': args.triplet,
            'num_candidates': args.num_candidates,
            'seed': args.seed,
            'steps': args.steps,
            'split': args.split
        },
        'recipe': 'aurora_recipe_v1'  # Reference to configs/aurora_recipe_v1.yaml
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved to {save_path} (v{save_data['version']})")



if __name__ == "__main__":
    main()

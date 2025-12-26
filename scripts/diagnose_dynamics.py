
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.dynamics import PhysicsConfig, PhysicsState, ContactIntegrator
from aurora.inertia import RadialInertia
from aurora.diamond import compute_diamond_torque
from aurora.potentials import RadiusAnchorPotential, SemanticTripletPotential, RobustSpringPotential, SpringPotential
from aurora.data import AuroraDataset

def load_checkpoint(path, device):
    print(f"Loading checkpoint from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Version: {data.get('version', 'unknown')}")
    print(f"  Keys: {list(data.keys())}")
    print(f"  Step: {data.get('step', 'N/A')}")
    return data

def main():
    parser = argparse.ArgumentParser(description="Diagnose CCD Dynamics (Torque/Inertia)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint pkl")
    parser.add_argument("--semantic_path", type=str, default="checkpoints/semantic_edges_wiki.pkl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    chk = load_checkpoint(args.checkpoint, device)
    
    # Reconstruct Physics State
    
    # Reconstruct Physics State from 'x' or 'G'
    if 'G' in chk:
        G = chk['G'].to(device)
        p = chk.get('p', None)
        if p is not None: p = p.to(device)
    elif 'x' in chk:
        print("Checkpoint contains 'x' only (Lightweight). Reconstructing dummy frame.")
        if isinstance(chk['x'], np.ndarray):
            x_emb = torch.from_numpy(chk['x']).to(device)
        else:
            x_emb = chk['x'].to(device)
        # Construct Dummy G (N, dim, dim)
        # We only need G[..., 0] = x_emb for force computation
        # But for torque calculation, integrator needs G to potentially project? 
        # Actually integrator.force_to_torque(F, x) takes x, not G.
        
        # We treat x_emb as our 'G' for positioning
        # G is typically used for state.G
        N, dim = x_emb.shape
        G = torch.eye(dim, device=device).unsqueeze(0).repeat(N, 1, 1)
        G[..., 0] = x_emb # Set 0-th column to position
        
        p = None
        
    # Config (Minimal)
    config = PhysicsConfig()
    
    # Inertia
    inertia = RadialInertia(alpha=5.0)

    # Reconstruct Integrator
    integrator = ContactIntegrator(config, inertia)
        
    # 2. Semantic Potential
    print(f"Loading semantic edges from {args.semantic_path}...")
    with open(args.semantic_path, 'rb') as f:
        sem_data = pickle.load(f)
    
    # Load ALL edges for diagnosis (not just train split) to see total force field
    print(f"  Type: {type(sem_data)}")
    # Load Vocab
    vocab = chk.get('vocab', None)
    if vocab is None:
        raise ValueError("Checkpoint must contain 'vocab' to map semantic edges.")
    
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    # Process edges
    edge_indices = []
    
    raw_edges = sem_data if isinstance(sem_data, list) else sem_data['edges']
    
    # Build standard word2idx
    if isinstance(vocab, list):
        word2idx = {w: i for i, w in enumerate(vocab)}
    elif isinstance(vocab, dict):
        word2idx = vocab
    else:
        raise ValueError(f"Unknown vocab type: {type(vocab)}")
        
    # [NEW] Cilin Code Parsing (Replicated from eval_aurora.py)
    # Cilin nodes are format "C:Word:Index"
    # We need to map "Word" -> Index
    raw_to_id = {}
    for node_str in word2idx.keys():
        if isinstance(node_str, str) and node_str.startswith("C:"):
            parts = node_str.split(":")
            if len(parts) >= 2:
                # Format C:Word:UniqueId
                raw_word = parts[1]
                # If collision, keep first? Or last? Eval script keeps first.
                if raw_word not in raw_to_id:
                    # We map to the ID in word2idx
                    idx = word2idx[node_str]
                    raw_to_id[raw_word] = idx
    
    print(f"  Parsed {len(raw_to_id)} raw words from Cilin codes.")
    
    # Process edges
    edge_indices = []
    
    raw_edges = sem_data if isinstance(sem_data, list) else sem_data['edges']
    
    count_skipped = 0
    for item in raw_edges:
        # Item structure: (u, v, weight, ...)
        u, v = item[0], item[1]
        weight = item[2] if len(item) > 2 else 1.0
        
        # Try direct match first, then parsed match
        u_idx = word2idx.get(u)
        if u_idx is None: u_idx = raw_to_id.get(u)
        
        v_idx = word2idx.get(v)
        if v_idx is None: v_idx = raw_to_id.get(v)
        
        if u_idx is not None and v_idx is not None:
             if u_idx != v_idx:
                edge_indices.append([u_idx, v_idx, 1]) 
        else:
            count_skipped += 1
            
    print(f"  Mapped {len(edge_indices)} edges. Skipped {count_skipped}.")
    
    if not edge_indices:
        raise ValueError("No valid edges found after mapping!")
        
    edges = torch.tensor(edge_indices, dtype=torch.long, device=device)
    
    # Setup Potential
    # Use parameters from Phase XXXIII
    # Note: Phase XXXIII used k=2.0 (but check if this k is enough?)
    pot_sem = SemanticTripletPotential(
        edges, 
        k=2.0, 
        margin=0.1,
        mining_mode='trusted', 
        num_candidates=100
    )
    
    # Calculate Forces
    print("Computing Semantic Forces...")
    x = G[..., 0] # Extract position vectors
    
    # We need to temporarily monkey-patch or access compute_forces directly
    E_sem, F_sem = pot_sem.compute_forces(x)
    
    print("\n[DEBUG] Vector check:")
    print(f"  x[0]: {x[0].detach().cpu().numpy()}")
    print(f"  F_sem[0]: {F_sem[0].detach().cpu().numpy()}")
    print(f"  G[0,:,0] (should be x[0]): {G[0,:,0].detach().cpu().numpy()}")
    
    # Check if F_sem spatial is zero
    f_spatial_norm = torch.norm(F_sem[:, 1:], dim=1).mean().item()
    print(f"  Mean F_sem Spatial Norm: {f_spatial_norm:.6e}")
    if f_spatial_norm < 1e-9:
        print("  !!! ALERT: Force is purely in time direction !!!")
    
    # Calculate Torque from Semantic Force
    # Torque tau = G^{-1} . F_world \wedge e_0 (Diamond Operator)
    torque_sem = compute_diamond_torque(G, F_sem)
    
    # Analyze Torque
    print(f"Torque Shape: {torque_sem.shape}")
    
    num_nodes = G.shape[0]
    # Decompose F_sem into Radial (parallel to x) and Tangential (orthogonal to x)
    # x is (N, D). F is (N, D).
    # Project F onto x.
    # Metric is Minkowski. <x, x> = -1.
    # F_radial = - <F, x> * x
    # F_tangential = F - F_radial
    
    # Inner product <F, x>_M
    # x0 F0 - x_space . F_space ? No. -x0 F0 + x_i F_i.
    inner_Fx = -F_sem[:, 0] * x[:, 0] + torch.sum(F_sem[:, 1:] * x[:, 1:], dim=1)
    
    # Radial Component vector
    # Project onto direction x. Unit vector is x (since <x,x>=-1, basis is spacelike?)
    # Wait, time-like vector x. <x,x> = -1.
    # Projection of v onto u (time-like): P_u(v) = - <v, u> u / <u, u> = <v, u> u.
    F_radial = inner_Fx.unsqueeze(1) * x
    
    F_tangential = F_sem - F_radial
    
    mag_radial = torch.norm(F_radial, dim=1).mean().item()
    mag_tangential = torch.norm(F_tangential, dim=1).mean().item()
    
    print("\n=== FORCE DECOMPOSITION ===")
    print(f"Mean Force Radial (Expansion):    {mag_radial:.6e}")
    
    # 3. Structural Potential
    # Need to load dataset to get structural edges?
    # Or just use the 'nodes' and infer from neighbor list? 
    # Best to use AuroraDataset logic snippet or just load the edges file if available.
    # Assuming 'cilin' dataset path convention for now, or skip if complex.
    # In 'train_aurora_v2.py', we load ds.edges_struct.
    # Let's try to find 'dataset' object in checkpoint? No.
    # We will try to load standard cilia dataset edges if possible.
    
    # Simpler: Just rely on semantic force magnitude.
    # We already know F_sem_tan ~ 470.
    # Is this "Large" or "Small"?
    # Normalized by Mass (76): a ~ 6.
    # In dt=0.01 step: dx ~ 0.5 * a * dt^2 ~ 0.0003.
    # This is SMALL displacement per step.
    # If F_struct is also 470, they cancel.
    
    # Let's attempt to reconstruct structural edges from a hardcoded path for Cilin
    # Path: data/cilin/cilin_edges.txt (Guess)
    # Actually, let's assume the user wants to see the comparison.
    # I will add a placeholder warning that F_struct is missing.
    
    return
    
    if mag_tangential < 1e-3:
        print("ALERT: Force is dominantly RADIAL. Angular alignment force is missing.")
    else:
        print("Tangential force exists. Structure might be resisting it.")
        
    return
    
    if p is not None:
        pass # Handle Momentum if available
    else:
        print("\n=== INERTIA DIAGNOSIS ===")
        print("Momentum 'p' not found in checkpoint. Skipping Inertia check.")
    
    # Conclusions
    print("\n=== DIAGNOSTIC CONCLUSION ===")
    if mag_rot < 1e-5:
        print("CRITICAL: Angular Dynamics are DEAD (Zero Torque).")
        print("  -> The Semantic Potential is failing to exert rotational force.")
    else:
        print(f"Dynamics Active. Rotational Torque: {mag_rot:.4e}")
        
    return

if __name__ == "__main__":
    main()

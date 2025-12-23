"""
Gradient Norm Audit.
====================

Loads checkpoint and measures the magnitude of forces (gradients) from each potential component:
1. Skeleton Attract (k=1.0 and k=0.3)
2. PMI Attract (k=0.5)
3. Repulsion
4. Trap

This verifies if PMI forces are numerically significant compared to Skeleton/Repulsion.
"""

import torch
import numpy as np
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.potentials_torch import (
    SpringAttractionTorch, 
    SparseEdgePotentialTorch, 
    LogCoshRepulsionTorch, 
    NegativeSamplingPotentialTorch, 
    HarmonicPriorTorch
)
from hei_n.torch_core import project_to_tangent_torch

def load_semantic_edges(wiki_path):
    # Mock or load real
    # We must load real to measure real forces
    pmi_path = "checkpoints/semantic_edges_wiki.pkl"
    if os.path.exists(pmi_path):
        print(f"Loading PMI edges from {pmi_path}")
        with open(pmi_path, 'rb') as f:
            edges_all = pickle.load(f)
        
        # edges_all list of (u, v, w, type)
        # Type 2 = PMI
        edges_pmi = []
        for u, v, w, t in edges_all:
            if t == 2:
                edges_pmi.append([u, v])
        
        if not edges_pmi:
            return None
        return np.array(edges_pmi)
    return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    if not os.path.exists(CHECKPOINT):
        print("Checkpoint not found.")
        return

    print(f"Loading Checkpoint {CHECKPOINT}...")
    with open(CHECKPOINT, 'rb') as f:
        data = pickle.load(f)
        
    G_np = data['G'] # (N, dim, dim)
    
    # Check if G is tensor (if saved from GPU script)
    # The saved dict likely has numpy arrays if .cpu().numpy() was used
    if isinstance(G_np, torch.Tensor):
        G_np = G_np.cpu().numpy()
        
    nodes = data['nodes']
    edges_skel_np = data['edges']
    
    # Convert to Tensor
    G = torch.tensor(G_np, device=device, dtype=torch.float32)
    x = G[..., 0] # Position
    
    edges_skel = torch.tensor(edges_skel_np, device=device, dtype=torch.long)
    
    # Load PMI Edges
    edges_pmi_np = load_semantic_edges(None)
    if edges_pmi_np is None:
        print("No PMI edges found.")
        return
    edges_pmi = torch.tensor(edges_pmi_np, device=device, dtype=torch.long)
    
    print(f"Nodes: {len(nodes)}")
    print(f"Skel Edges: {len(edges_skel)}")
    print(f"PMI Edges: {len(edges_pmi)}")
    
    # Define Potentials
    pots = {}
    
    # 1. Skeleton (Strong k=1.0)
    pots['Skel_k1.0'] = SparseEdgePotentialTorch(
        edges_skel, SpringAttractionTorch(k=1.0)
    )
    
    # 2. Skeleton (Soft k=0.3)
    pots['Skel_k0.3'] = SparseEdgePotentialTorch(
        edges_skel, SpringAttractionTorch(k=0.3)
    )
    
    # 3. PMI (Target k=0.5)
    pots['PMI_k0.5'] = SparseEdgePotentialTorch(
        edges_pmi, SpringAttractionTorch(k=0.5)
    )
    
    # 4. Repulsion
    k_repulse = LogCoshRepulsionTorch(sigma=1.0, A=5.0)
    pots['Repulsion'] = NegativeSamplingPotentialTorch(k_repulse, num_neg=10, rescale=1.0)
    
    # 5. Trap
    pots['Trap'] = HarmonicPriorTorch(k=0.05)
    
    print("\n--- GRADIENT NORM AUDIT ---")
    print(f"{'Component':<15} | {'Mean Force':<12} | {'Max Force':<12} | {'P99 Force':<12}")
    print("-" * 60)
    
    for name, pot in pots.items():
        # Compute grad
        if name == 'Repulsion':
            # Needs repeated sampling to get stable estimate?
            # Just one sample is fine for scale check
            pass
            
        energy, grad = pot.potential_and_grad(x)
        
        # Project to Tangent Space (Actual Force felt by particle)
        # force = -grad
        # projected = project_to_tangent(x, force)
        # Magnitude is norm of projected vector
        
        force = -grad
        proj_force = project_to_tangent_torch(x, force)
        
        # Norm: sqrt(-t*t + x*x) ?? 
        # Tangent vector v at x satisfies <x, v> = 0.
        # Norm squared is <v, v>_L = -v0^2 + v_spatial^2
        # Since v is spacelike, <v, v> > 0.
        
        v0 = proj_force[:, 0]
        vs = proj_force[:, 1:]
        norm_sq = -v0**2 + torch.sum(vs**2, dim=-1)
        norms = torch.sqrt(torch.clamp(norm_sq, min=0.0))
        
        norms_np = norms.detach().cpu().numpy()
        
        mean_val = np.mean(norms_np)
        max_val = np.max(norms_np)
        p99_val = np.percentile(norms_np, 99)
        
        print(f"{name:<15} | {mean_val:.2e}     | {max_val:.2e}     | {p99_val:.2e}")
        
    print("-" * 60)

if __name__ == "__main__":
    main()

"""
Quick Gradient Audit with Configurable Repulsion A.
====================================================

Used for stepped parameter sweep to verify Repulsion/PMI ratio.
"""

import torch
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.potentials_torch import (
    SpringAttractionTorch, 
    SparseEdgePotentialTorch, 
    LogCoshRepulsionTorch, 
    NegativeSamplingPotentialTorch
)
from hei_n.torch_core import project_to_tangent_torch

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repulsion-a", type=float, default=5.0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CHECKPOINT = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    if not os.path.exists(CHECKPOINT):
        print("Checkpoint not found.")
        return

    with open(CHECKPOINT, 'rb') as f:
        data = pickle.load(f)
        
    G_np = data['G']
    if isinstance(G_np, torch.Tensor):
        G_np = G_np.cpu().numpy()
        
    edges_skel_np = data['edges']
    
    G = torch.tensor(G_np, device=device, dtype=torch.float32)
    x = G[..., 0]
    edges_skel = torch.tensor(edges_skel_np, device=device, dtype=torch.long)
    
    # Load PMI
    pmi_path = "checkpoints/semantic_edges_wiki.pkl"
    with open(pmi_path, 'rb') as f:
        edges_all = pickle.load(f)
    edges_pmi = np.array([[u, v] for u, v, w, t in edges_all if t == 2])
    edges_pmi = torch.tensor(edges_pmi, device=device, dtype=torch.long)
    
    # Potentials
    pot_pmi = SparseEdgePotentialTorch(edges_pmi, SpringAttractionTorch(k=0.5))
    pot_repulse = NegativeSamplingPotentialTorch(
        LogCoshRepulsionTorch(sigma=1.0, A=args.repulsion_a), 
        num_neg=10, rescale=1.0
    )
    
    # Compute forces
    def get_mean_force(pot, x):
        _, grad = pot.potential_and_grad(x)
        force = -grad
        proj = project_to_tangent_torch(x, force)
        v0 = proj[:, 0]
        vs = proj[:, 1:]
        norm_sq = -v0**2 + torch.sum(vs**2, dim=-1)
        norms = torch.sqrt(torch.clamp(norm_sq, min=0.0))
        return norms.mean().item()
    
    pmi_force = get_mean_force(pot_pmi, x)
    rep_force = get_mean_force(pot_repulse, x)
    
    ratio = rep_force / (pmi_force + 1e-9)
    
    print(f"Repulsion A={args.repulsion_a}")
    print(f"PMI Force: {pmi_force:.2f}")
    print(f"Repulsion Force: {rep_force:.2f}")
    print(f"Ratio (Rep/PMI): {ratio:.2f}")
    
    if ratio <= 2.0:
        print(">>> TARGET MET: Repulsion/PMI <= 2.0 <<<")
    else:
        print(f">>> NOT MET: Need lower A (current ratio {ratio:.1f}x)")

if __name__ == "__main__":
    main()

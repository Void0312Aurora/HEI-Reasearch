"""
Phase 21: CLUE Probing Demo.
============================

Zero-shot evaluation on Mock CLUE Examples.
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField, NeuralBackend
from aurora.eval.clue_probe import GeometricProbe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load
    print("Loading Mind...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    
    # Legacy Backend Patch again for Cycle 3
    from aurora.gauge import GaugeConnectionBackend
    import torch.nn as nn
    from aurora.geometry import log_map
    
    class LegacyNeuralBackend(GaugeConnectionBackend):
        def __init__(self, input_dim=5, logical_dim=3, hidden_dim=64):
            super().__init__(logical_dim)
            input_size = 3 * input_dim
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, logical_dim * logical_dim)
            )
            self.rel_embed = None
            
        def get_omega(self, edges=None, x=None, relation_ids=None):
             return self.forward(x=x, edges_uv=edges)
             
        def forward(self, edge_indices=None, x=None, edges_uv=None, relation_ids=None):
            if x is None: raise ValueError("Need x")
            u, v = edges_uv[:, 0], edges_uv[:, 1]
            swap_mask = u > v
            u_canon = torch.where(swap_mask, v, u)
            v_canon = torch.where(swap_mask, u, v)
            xu = x[u_canon]
            xv = x[v_canon]
            v_uv = log_map(xu, xv)
            feat = torch.cat([xu, xv, v_uv], dim=-1)
            out = self.net(feat)
            out = 3.0 * torch.tanh(out)
            out = out.view(-1, self.logical_dim, self.logical_dim)
            omega_canon = 0.5 * (out - out.transpose(1, 2))
            negation = torch.where(swap_mask, -1.0, 1.0).view(-1, 1, 1)
            return omega_canon * negation

    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5).to(device)
    gauge_field.backend = LegacyNeuralBackend().to(device)
    gf_state = ckpt['gauge_field']
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    probe = GeometricProbe(gauge_field, x, J, ds, device=device)
    
    print(f"Vocab Sample: {list(ds.vocab.word_to_id.keys())[:10]}")
    
    nodes = ds.nodes
    # We use ds.nodes to find synonyms/antonyms?
    # Or just use the samples from demo.
    
    nodes = ds.nodes
    # Use full keys
    w1 = nodes[100] 
    w2 = nodes[101] 
    w3 = nodes[5000]
    
    print(f"\n--- Zero-Shot CLUE Probe ---")
    
    pairs = [
        (w1, w2, "Expected High Sim"),
        (w1, w3, "Expected Low Sim"),
        (w1, w1, "Identity")
    ]
    
    for a, b, desc in pairs:
        score = probe.score_similarity(a, b)
        pred = probe.predict_entailment(a, b)
        print(f"\nA: {a}\nB: {b}\nDesc: {desc}\nScore: {score:.4f} -> Pred: {pred}")

if __name__ == "__main__":
    main()

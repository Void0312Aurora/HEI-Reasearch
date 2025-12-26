"""
Closed-Loop Active Inference Orchestrator (Phase 11)

Orchestrates the autonomous cycle:
1. Train Gauge Field (Physics)
2. Mine Semantic Candidates (Inference)
3. Filter with Safety Gates (Curvature/Energy)
4. Update Topology
5. Repeat
"""

import sys
import os
import torch
import torch.nn as nn
import pickle
import argparse
import numpy as np
from pathlib import Path
import copy

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField
from aurora.dynamics import ContactIntegrator, PhysicsState
from aurora.potentials import CompositePotential, SpringPotential, SemanticTripletPotential
from aurora.inference import InferenceEngine
from aurora.geometry import minkowski_inner

# --- Training Module Wrapper ---
# We need to run training programmatically. 
# Option A: Import main() from train_aurora_v2 (requires refactor).
# Option B: Re-implement simplified loop here.
# Option C: Use subprocess (External process) -> Cleanest for memory management.

import subprocess

def run_training_step(output_ckpt, semantic_path, initial_ckpt=None, 
                      dataset='cilin', limit=10000, steps=2000, 
                      lr=0.01, device='cuda'):
    """
    Run the training script as a subprocess.
    """
    cmd = [
        "python", "scripts/train_aurora_v2.py",
        "--dataset", dataset,
        "--limit", str(limit),
        "--steps", str(steps),
        "--enable_logic",
        "--gauge_mode", "neural",
        "--learn_gauge",
        "--lr_gauge", str(lr),
        "--curvature_reg", "0.01",
        "--spin_edges_mode", "structural", # Decoupling Protocol
        "--semantic_path", semantic_path,
        "--split", "train", # Use all for training
        "--save_path", output_ckpt
    ]
    
    if initial_ckpt:
        cmd.extend(["--checkpoint", initial_ckpt])
        
    print(f"Running Training Step: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False) # Passthrough stdout
    
    if result.returncode != 0:
        raise RuntimeError("Training failed!")

def update_semantic_dataset(base_path, new_edges, output_path):
    """
    Append accepted edges to the semantic dataset.
    """
    # Load base
    with open(base_path, 'rb') as f:
        edges = pickle.load(f)
        
    edges.extend(new_edges)
    
    # Deduplicate? Inference engine usually handles candidates, but new edges vs old?
    # Set of (u, v)
    seen = set()
    cleaned = []
    
    # Normalize order u < v for check? No, dataset stores (u, v, w)
    # Assume undirected for existence check
    for item in edges:
        u, v = item[0], item[1]
        w = item[2]
        
        pair = tuple(sorted((u, v)))
        if pair not in seen:
            seen.add(pair)
            cleaned.append(item)
            
    with open(output_path, 'wb') as f:
        pickle.dump(cleaned, f)
        
    print(f"Updated Semantic Dataset: {len(edges)} -> {len(cleaned)} edges. Saved to {output_path}")
    return len(cleaned)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_semantic", type=str, required=True, help="Initial semantic edges pickle")
    parser.add_argument("--dataset", type=str, default="cilin")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--candidates_per_cycle", type=int, default=1000) # K for KNN
    parser.add_argument("--accept_per_cycle", type=int, default=200)
    parser.add_argument("--workspace", type=str, default="closed_loop_workspace")
    args = parser.parse_args()
    
    workspace = Path(args.workspace)
    workspace.mkdir(exist_ok=True, parents=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    current_semantic = str(workspace / "semantic_state_0.pkl")
    # Copy base to start
    import shutil
    shutil.copy(args.base_semantic, current_semantic)
    
    last_ckpt = None
    
    for cycle in range(1, args.cycles + 1):
        print(f"\n====== Cycle {cycle}/{args.cycles} ======")
        
        # 1. Train
        ckpt_path = str(workspace / f"checkpoint_cycle_{cycle}.pkl")
        run_training_step(output_ckpt=ckpt_path, 
                          semantic_path=current_semantic,
                          initial_ckpt=last_ckpt, # Continue training or restart? Restart for now to avoid drift, or load weights but reset momentum. 
                          # Script supports --load_checkpoint. 
                          # If we restart physics from scratch each time, it's safer? 
                          # Gauge weights should be carried over ideally.
                          # Current `train_aurora_v2` --load_checkpoint loads everything.
                          # Let's verify if `train_aurora_v2` loads properly. Yes it does.
                          dataset=args.dataset,
                          limit=args.limit,
                          steps=2000 # Short cycle for demo
                          )
        
        last_ckpt = ckpt_path
        
        # 2. Mine
        print(f"--- Mining Candidates (Cycle {cycle}) ---")
        
        # Load Checkpoint to get GaugeField
        # Use torch.load
        # We need to reconstruct the objects to use InferenceEngine
        # Or... separate script for mining?
        # Let's instantiate InferenceEngine here.
        
        # Need to reconstruct GaugeField identical to training
        # Creating dataset object to get edges
        ds = AuroraDataset(args.dataset, limit=args.limit)
        
        # Load semantic
        sem_list = ds.load_semantic_edges(current_semantic, split='train') # Actually we want ALL edges in the file
        # But load_semantic_edges splits. 
        # For closed loop, we probably want to use the file as "Ground Truth" for Training.
        # But Inference should avoid existing edges.
        
        # Fix: load_semantic_edges logic. If we pass 'train', we get subset.
        # We should use a helper to load ALL edges from pickle for Gauge construction
        # Or just trust ds.load_semantic_edges('train') if training used that.
        # Training uses 'train' split. Inference should verify against 'train' edges.
        
        # ... Reconstructing GaugeField ...
        # This is code duplication from eval script. Implementation is fine.
        
        # Actually, simpler: Use `eval_semantic_truth.py` logic or similar.
        # But we need InferenceEngine class.
        
        # Load CKPT
        map_loc = device
        try:
             ckpt = torch.load(ckpt_path, map_location=map_loc)
        except:
             # Fallback
             import io
             class CPU_Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    return super().find_class(module, name)
             with open(ckpt_path, 'rb') as f: ckpt = CPU_Unpickler(f).load()

        J = torch.tensor(ckpt['J'], device=device)
        x = torch.tensor(ckpt['x'], device=device)
        
        # Reconstruct Edges
        edges_struct = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
        sem_edges_idx = torch.tensor([(u,v) for u, v, w in sem_list], dtype=torch.long, device=device)
        all_edges = torch.cat([edges_struct, sem_edges_idx], dim=0)
        
        # GaugeField
        gauge = GaugeField(all_edges, logical_dim=3, backend_type='neural', input_dim=5).to(device)
        gauge.load_state_dict(ckpt['gauge_field'])
        
        # Inference Engine
        engine = InferenceEngine(gauge, J, x, device=device)
        
        # Generate
        candidates = engine.generate_candidates_knn(k=20, batch_size=1000)
        
        # Filter (Gate A)
        accepted_idx = engine.filter_edges_robust(candidates, k_accept=args.accept_per_cycle,
                                                  curvature_threshold_p99=8.0) # Slightly loose for V13 heavy tails
        
        if len(accepted_idx) == 0:
            print("No edges accepted via Gate A. Stopping loop.")
            break
            
        # Convert index pairs back to (str, str, w) for saving
        # Need vocabulary
        vocab = ds.vocab
        new_edge_list = []
        
        acc_np = accepted_idx.cpu().numpy()
        for u, v in acc_np:
            u_s = ds.nodes[u]
            v_s = ds.nodes[v]
            # Weight = 0.5 default
            new_edge_list.append((u_s, v_s, 0.5))
            
        # 3. Update Dataset
        next_semantic = str(workspace / f"semantic_state_{cycle}.pkl")
        update_semantic_dataset(current_semantic, new_edge_list, next_semantic)
        current_semantic = next_semantic
        
        print(f"Cycle {cycle} Complete. Graph updated with {len(new_edge_list)} edges.")

if __name__ == "__main__":
    main()

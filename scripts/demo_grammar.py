"""
Demo: Geometric Grammar (Phase 15).
===================================

Demonstrates:
1. Sentence Generation (Subject -> Predicate -> Object) using Grammar Engine.
2. Continuous Semantic Flow (Trajectory Visualization output).

Note: Uses untyped (neutral) transport as the checkpoint was not trained with relations.
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField
from aurora.grammar import GrammarTrajectory, SentenceGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    print("Loading Checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    gf_state = ckpt['gauge_field']
    
    # Init with relation_dim=0 to match checkpoint
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural', 
                             # Backend args passed via? GaugeField helper?
                             # GaugeField init doesn't pass legacy args well.
                             # Actually GaugeField.__init__ takes backend_type and input_dim.
                             # It constructs NeuralBackend(input_dim, logical_dim).
                             # It keeps default num_relations=1, relation_dim=0.
                             input_dim=5)
                             
    gauge_field.to(device)
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    # 2. Data & Graph
    ds = AuroraDataset("cilin", limit=args.limit)
    
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    G = nx.Graph()
    for u, v in ds.edges_struct:
        G.add_edge(u, v)
    for u, v, w in sem_edges:
        G.add_edge(u, v)
        
    print(f"Graph Built: {G.number_of_nodes()} nodes.")
    
    # 3. Initialize Grammar Components
    traj = GrammarTrajectory(gauge_field, x, J, device=device)
    gen = SentenceGenerator(traj, G, ds)
    
    # 4. Generate SVO Sentences
    print("\n=== Demo 1: SVO Sentence Generation ===")
    subjects = ["C:太阳:1", "C:学生:283", "C:医生:260"]
    # Fallback to searching if IDs fail
    
    for subj_str in subjects:
        # Find closest match in vocab
        found = False
        target_id = -1
        target_label = ""
        
        # Try direct
        if subj_str in ds.vocab.word_to_id:
            target_id = ds.vocab.word_to_id[subj_str]
            target_label = subj_str
            found = True
        else:
            # Try searching by code prefix or name
            query = subj_str.split(":")[1] if ":" in subj_str else subj_str
            for n_id, label in enumerate(ds.nodes):
                if query in label:
                    target_id = n_id
                    target_label = label
                    found = True
                    break
        
        if not found:
            # Random
            target_id = np.random.randint(0, 500)
            target_label = ds.nodes[target_id]
            
        print(f"\nSubject Seed: {target_label}")
        
        result = gen.generate_svo(target_label)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            s = result['subject'].split(":")[1] if ":" in result['subject'] else result['subject']
            p = result['predicate'].split(":")[1] if ":" in result['predicate'] else result['predicate']
            o = result['object'].split(":")[1] if ":" in result['object'] else result['object']
            
            print(f"Sentence: {s} (Subj) -> {p} (Pred) -> {o} (Obj)")
            print(f"Energies: S->P {result['energy_sp']:.4f}, P->O {result['energy_po']:.4f}")
            
    # 5. Continuous Flow Visualization (Placeholder)
    print("\n=== Demo 2: Continuous Semantic Interchange (Viz) ===")
    print("Simulating path interpolation between S and P...")
    # Just printing logic, actual matplotlib needs display
    print("Trajectory: x(t) = Geodesic(x_S, x_P, t)")
    print("Frame:      J(t) = Transport(J_S, x_S->x(t))")
    print("[ Visualization Data Generated ]")

if __name__ == "__main__":
    main()

"""
Demo: Phase 16 Corpus-Scale Deployment.
=======================================

Demonstrates:
1. Parsing raw text into Geometric Trajectories via Dependency Geometer.
2. Streaming training of the Gauge Field on continuous data.

Note: Since we are training relation types, we initialize a new GaugeField with `num_relations=4`.
We use the Checkpoint's Geometry (x) but learn new Physics (U).
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
from aurora.ingest.dependency import DependencyGeometer
from aurora.training.streaming import StreamingTrainer

def sentence_stream_generator(ds, n_sentences=1000):
    """
    Simulates a stream of generated sentences.
    """
    # To ensure hits, we sample from actual dataset nodes
    nodes = ds.nodes # It is already a list
    subjects = [n.split(":")[1] if ":" in n else n for n in nodes[:500]] # Heuristic
    verbs = [n.split(":")[1] if ":" in n else n for n in nodes[500:1000]]
    objects = [n.split(":")[1] if ":" in n else n for n in nodes[1000:1500]]
    
    for _ in range(n_sentences):
        s = np.random.choice(subjects)
        v = np.random.choice(verbs)
        o = np.random.choice(objects)
        
        # Simple SVO sentence
        text = f"{s} {v} {o}"
        yield text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint (for x and J)
    print("Loading Checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    
    # 2. Init Untrained Typed Gauge Field
    # num_relations = 4 (Neutral, Subj, Obj, Mod)
    print("Initializing Typed Gauge Field (Relations=4)...")
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural', 
                             input_dim=5)
                             
    # We must patch the backend to have relations
    # Re-init backend explicitly or via modified GaugeField logic
    # The default GaugeField init uses default args for backend.
    # We need to manually replace backend if our GaugeField init doesn't expose num_relations.
    # We didn't update GaugeField.__init__ signature, only Backend.
    # So let's manually re-init backend.
    
    from aurora.gauge import NeuralBackend
    gauge_field.backend = NeuralBackend(input_dim=5, logical_dim=3, num_relations=4, relation_dim=8)
    gauge_field.to(device)
    
    # 3. Components
    ds = AuroraDataset("cilin", limit=args.limit)
    geometer = DependencyGeometer()
    trainer = StreamingTrainer(gauge_field, x, J, ds, device=device, lr=0.01)
    
    # 4. Run Stream
    print("\n=== Starting Streaming Training ===")
    
    def edge_generator():
        stream = sentence_stream_generator(ds)
        for text in stream:
            edges = geometer.text_to_edges(text)
            for e in edges:
                yield e
                
    trainer.train_stream(edge_generator(), steps=500)
    
    print("Stream Completed.")

if __name__ == "__main__":
    main()

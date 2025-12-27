"""
Phase 24: Joint Training Demo.
==============================

Train Semantics (J) and Language (U) together.
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
from aurora.training.joint_trainer import JointTrainer
from aurora.ingest.wiki_dump import WikiDumpIngestor

def create_mock_dump(ds, filename, lines=500):
    nodes = ds.nodes
    subjects = nodes[:500]
    verbs = nodes[500:1000]
    objects = nodes[1000:1500]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(lines):
            s = np.random.choice(subjects)
            v = np.random.choice(verbs)
            o = np.random.choice(objects)
            f.write(f"{s} {v} {o}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Checkpoint...")
    
    # Custom Unpickler for CPU/CUDA mapping
    import io
    class DeviceUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=device)
            return super().find_class(module, name)
            
    with open(args.checkpoint, 'rb') as f:
        ckpt = DeviceUnpickler(f).load()
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    # J must require grad
    J.requires_grad = True
    
    # Init Gauge (200 relations)
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5).to(device)
    gauge_field.backend = NeuralBackend(input_dim=5, logical_dim=3, num_relations=200, relation_dim=8).to(device)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # 1. Structural Edges (from Semantic Tree)
    struct_edges = ds.edges_struct # List of (u, v)
    print(f"Loaded {len(struct_edges)} Structural Edges.")
    
    # 2. Language Edges (from Wiki)
    dump_file = "mock_wiki_p24.txt"
    create_mock_dump(ds, dump_file)
    ingestor = WikiDumpIngestor(ds, neg_ratio=2) # 2 negatives per positive
    
    print("Ingesting Language Edges...")
    lang_batches = list(ingestor.stream_dump(dump_file, batch_size=256))
    total_lang = sum(len(b) for b in lang_batches)
    print(f"Loaded {total_lang} Language Edges.")
    
    # 3. Trainer
    trainer = JointTrainer(gauge_field, x, J, device=device)
    
    print("Starting Joint Training...")
    for epoch in range(3):
        total_loss_J = 0
        total_loss_U = 0
        batches = 0
        
        for batch in lang_batches:
            l_J, l_U = trainer.train_step(struct_edges, batch)
            total_loss_J += l_J
            total_loss_U += l_U
            batches += 1
            
        avg_J = total_loss_J / batches
        avg_U = total_loss_U / batches
        print(f"Epoch {epoch+1}: Loss_J {avg_J:.4f} | Loss_U {avg_U:.4f}")
        
    print("Joint Training Complete.")
    # Validation: Check if J changed?
    # Check if U learned?
    
    if os.path.exists(dump_file):
        os.remove(dump_file)

if __name__ == "__main__":
    main()

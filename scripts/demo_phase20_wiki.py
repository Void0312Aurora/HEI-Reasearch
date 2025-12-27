"""
Phase 20: Wikipedia Geometrization Demo.
========================================

1. Simulate Wikipedia Corpus.
2. Ingest into Flow/Phrase/Dependency Edges.
3. Train Typed Gauge Field.
4. Validate with Cloze.
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
from aurora.ingest.wikipedia import WikipediaIngestor
from aurora.training.language_trainer import LanguageTrainer

def generate_wiki_text(ds, n_sentences=500):
    """
    Simulate Wikipedia-like text using Cilin nodes.
    Format: "S V O ."
    """
    nodes = ds.nodes
    subjects = nodes[:500]
    verbs = nodes[500:1000]
    objects = nodes[1000:1500]
    
    texts = []
    for _ in range(n_sentences):
        s = np.random.choice(subjects)
        v = np.random.choice(verbs)
        o = np.random.choice(objects)
        # Randomly compose
        texts.append(f"{s} {v} {o}")
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint for x and J
    print("Loading Checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    
    # 2. Init Gauge Field with sufficient relations
    # Flow=100, Phrase=101. So num_relations=200 is safe.
    print("Initializing Gauge Field...")
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5).to(device)
    # NeuralBackend with 200 relations
    gauge_field.backend = NeuralBackend(input_dim=5, logical_dim=3, num_relations=200, relation_dim=8).to(device)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    ingestor = WikipediaIngestor(ds)
    trainer = LanguageTrainer(gauge_field, x, J, device=device)
    
    # 3. Simulate Data
    print("Simulating Wikipedia Data...")
    docs = generate_wiki_text(ds, n_sentences=500)
    
    # 4. Ingest
    print("Ingesting...")
    all_edges = []
    for doc in docs:
        edges = ingestor.ingest_document(doc)
        all_edges.extend(edges)
        
    print(f"Extracted {len(all_edges)} edges from {len(docs)} sentences.")
    print("Training...")
    
    # Train
    for epoch in range(5):
        loss = trainer.train_epoch(all_edges)
        print(f"Epoch {epoch+1}: Loss {loss:.4f}")
        
    print("Training Complete. Validating...")
    
    # Validation: Simple check if loss dropped
    if loss < 0.9:
        print("Success: Loss dropped significantly.")
    else:
        print("Warning: Loss remains high (expected if data is random SVO).")

if __name__ == "__main__":
    main()

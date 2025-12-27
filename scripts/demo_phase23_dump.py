"""
Phase 23: Scalable Ingestion Dump Demo.
=======================================

1. Creates mock Wiki dump.
2. Ingests using WikiDumpIngestor.
3. Validates Stats and Negative Sampling.
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.ingest.wiki_dump import WikiDumpIngestor

def create_mock_dump(ds, filename, lines=1000):
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
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    print("Loading Dataset...")
    ds = AuroraDataset("cilin", limit=args.limit)
    ingestor = WikiDumpIngestor(ds, neg_ratio=5)
    
    dump_file = "mock_wiki.txt"
    print(f"Creating mock dump {dump_file}...")
    create_mock_dump(ds, dump_file, lines=1000)
    
    print("Streaming Dump...")
    batch_count = 0
    total_edges = 0
    
    for batch in ingestor.stream_dump(dump_file, batch_size=500):
        batch_count += 1
        total_edges += len(batch)
        
        # Validation on first batch
        if batch_count == 1:
            print("Visualizing Batch 1 Samples:")
            for i, (u, v, r, label) in enumerate(batch[:10]):
                type_str = "Flow" if r == 100 else ("Phrase" if r==101 else f"Dep({r})")
                print(f"Edge: {u}->{v} Type: {type_str} Label: {label}")
                
            # Check Negatives exist
            negatives = [e for e in batch if e[3] == 0]
            if not negatives:
                print("Error: No negatives found in batch!")
            else:
                print(f"Verified Negatives in batch: {len(negatives)}")
                
    print("\nIngestion Complete.")
    print(f"Total Edges Yielded: {total_edges}")
    ingestor.print_stats()
    
    # Cleanup
    if os.path.exists(dump_file):
        os.remove(dump_file)
        
if __name__ == "__main__":
    main()

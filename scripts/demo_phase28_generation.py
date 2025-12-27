"""
Phase 28: Geometric Generation Demo.
====================================

Task: Translate "Semantic Intention" into "Word Sequence".
Method: Riemannian Beam Search on Gauge Manifold.
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
from aurora.generation.decoder import RiemannianBeamSearchDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=5000) # Smaller limit for faster vocab search
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Checkpoint...")
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
    
    # Init Gauge Field Matching Phase 27 Config
    # input_dim=5, logical_dim=3, r_dim=8, num_rel=200
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5).to(device)
    gauge_field.backend = NeuralBackend(input_dim=5, logical_dim=3, num_relations=200, relation_dim=8).to(device)
    
    # Checkpoint loading
    filtered = {k:v for k,v in ckpt['gauge_field'].items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    decoder = RiemannianBeamSearchDecoder(gauge_field, x, J, ds, device=device)
    
    # --- DEMO ---
    # Case: Start from Node A, Target is Node B.
    # Find path.
    # Pick random A and B.
    
    # Let's pick A="Person", B="Eat/Action".
    ids = list(ds.vocab.word_to_id.values())
    if len(ids) < 2:
        print("Dataset too small.")
        return
        
    start_id = ids[0]
    target_id = ids[50] # Some distance away
    start_word = ds.vocab.id_to_word[start_id]
    target_word = ds.vocab.id_to_word[target_id]
    
    print(f"Goal: '{start_word}' -> '{target_word}'")
    
    J_start = J[start_id]
    J_target = J[target_id]
    
    print("Decoding...")
    path = decoder.decode(J_start, J_target, max_len=5, beam_width=3)
    
    print(f"Generated Path: {path}")
    
    # Since model is untrained on language, path might be random or short.
    # But if it outputs a list of words, Mechanism is Verified.

if __name__ == "__main__":
    main()

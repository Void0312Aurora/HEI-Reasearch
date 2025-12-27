"""
Phase 26: Enhanced Dialogue Demo.
=================================

Task: Type-System Guided Reference Resolution.
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
from aurora.dialogue.state import DialogueState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoint
    # Use DeviceUnpickler
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
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5).to(device)
    gauge_field.backend = NeuralBackend(input_dim=5, logical_dim=3, num_relations=200, relation_dim=8).to(device)
    
    filtered = {k:v for k,v in ckpt['gauge_field'].items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    ingestor = WikipediaIngestor(ds)
    state = DialogueState(gauge_field, x, J, ingestor, device=device)
    
    # --- FIND VALID NODES FOR DEMO ---
    # We need a 'Person' (Starts with A) and 'Thing' (Starts with B or O)
    nodes = ds.nodes # Strings
    person_node = None
    thing_node = None
    
    for n in nodes:
        if "Code:A" in n or n.startswith("A"):
            if not person_node: person_node = n
        if "Code:B" in n or n.startswith("B"):
            if not thing_node: thing_node = n
            
        if person_node and thing_node: break
        
    print(f"Entities Found: \nPerson: {person_node}\nThing: {thing_node}")
    state.anchors[person_node] = {'id': ds.vocab.word_to_id[person_node], 'type': state._infer_type(ds.vocab.word_to_id[person_node])}
    state.anchors[thing_node] = {'id': ds.vocab.word_to_id[thing_node], 'type': state._infer_type(ds.vocab.word_to_id[thing_node])}
    
    # Set Focus to Thing (simulate Thing was mentioned last)
    state.J_ctx = J[ds.vocab.word_to_id[thing_node]]
    
    print(f"State Anchors: {state.anchors}")
    print(f"Focus (J_ctx) is on: {thing_node} (Type: {state.anchors[thing_node]['type']})")
    
    # Test 1: "He" (Should pick Person despite Focus on Thing)
    res_he = state.resolve_reference("he")
    print(f"\nQuery: 'he' -> {res_he}")
    
    # Test 2: "It" (Should pick Thing due to Focus + Type Match)
    res_it = state.resolve_reference("it")
    print(f"Query: 'it' -> {res_it}")
    
if __name__ == "__main__":
    main()

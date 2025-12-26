"""
Demo: The Geometric Chatbot (Phase 17).
=======================================

Interactive CLI.
Chat with the Gauge Field.
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
from aurora.chatbot import GeometricBot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load context
    print("Loading Mind...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    gf_state = ckpt['gauge_field']
    
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural', input_dim=5).to(device)
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # Build Graph for Neighbors
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    G = nx.Graph()
    for u, v in ds.edges_struct: G.add_edge(u, v)
    for u, v, w in sem_edges: G.add_edge(u, v)
    
    bot = GeometricBot(gauge_field, x, J, ds, G, device=device)
    
    print("\n[Aurora Geometric Chatbot Online]")
    print("Type a concept (e.g., '学生', '太阳').")
    print("Prefix 'stream ' for associative chain.")
    print("Type 'exit' to quit.\n")
    
    try:
        while True:
            text = input("User: ").strip()
            if not text: continue
            if text.lower() in ('exit', 'quit'): break
            
            if text.startswith("stream "):
                query = text.replace("stream ", "")
                chain = bot.stream_reply(query)
                print(f"Bot: {' -> '.join(chain)}")
            else:
                reply = bot.reply(text)
                print(f"Bot: {reply}")
                
    except KeyboardInterrupt:
        print("\nDisconnected.")

if __name__ == "__main__":
    main()

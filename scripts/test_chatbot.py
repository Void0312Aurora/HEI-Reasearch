"""
Automated Test for Geometric Chatbot.
=====================================
Wraps demo_chatbot logic for non-interactive systems.
"""
import sys
import os
import torch
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
    
    # Load
    print("Loading Checkpoint...")
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    gf_state = ckpt['gauge_field']
    
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, 5, 'neural', input_dim=5).to(device)
    filtered = {k:v for k,v in gf_state.items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    
    import re
    match = re.search(r"cycle_(\d+)", args.checkpoint)
    cycle_idx = match.group(1) if match else "3"
    sem_file = f"closed_loop_workspace/semantic_state_{cycle_idx}.pkl"
    sem_edges = ds.load_semantic_edges(sem_file, split="all")
    
    G = nx.Graph()
    for u, v in ds.edges_struct: G.add_edge(u, v)
    for u, v, w in sem_edges: G.add_edge(u, v)
    
    bot = GeometricBot(gauge_field, x, J, ds, G, device=device)
    
    # Test Queries
    queries = [
        "医生",          # Single shot
        "C:太阳:1",      # By ID
        "stream 宇宙",   # Stream
        "stream Code:Aa01"
    ]
    
    for q in queries:
        print(f"\nUser: {q}")
        if q.startswith("stream "):
            chain = bot.stream_reply(q.replace("stream ", ""))
            print(f"Bot: {' -> '.join(chain)}")
        else:
            reply = bot.reply(q)
            print(f"Bot: {reply}")

if __name__ == "__main__":
    main()

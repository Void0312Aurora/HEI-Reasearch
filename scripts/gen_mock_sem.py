
"""
Generate Mock Semantic Edges for Cilin.
Creates loops in the graph topology to enable Gauge Curvature.
"""
import sys
import os
import random
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--output", type=str, default="data/cilin_sem_mock.pkl")
    args = parser.parse_args()

    print(f"Loading Cilin dataset (limit={args.limit})...")
    ds = AuroraDataset('cilin', limit=args.limit)
    nodes = ds.nodes
    print(f"Nodes: {len(nodes)}")
    
    # Generate random edges to simulate 'semantic shortcuts'
    # These create strong frustration (loops) which is perfect for testing Gauge Learning.
    num_edges = args.count
    print(f"Generating {num_edges} random semantic edges within {len(nodes)} nodes...")
    
    print(f"Generating {num_edges} Triangle-Closing semantic edges...")

    # 1. Build Adjacency List to find 2-hop neighbors
    print("Building Adjacency List...")
    adj = {}
    for u, v in ds.edges_struct:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        # Undirected for neighbor finding
        adj[u].append(v)
        adj[v].append(u)
        
    edges = []
    
    # 2. Mine 2-hop paths (u - w - v) -> add (u, v)
    # This guarantees a triangle (u, v, w) is formed.
    keys = list(adj.keys())
    random.shuffle(keys)
    
    count = 0
    for w in keys:
        if count >= num_edges:
            break
            
        neighbors = adj[w]
        if len(neighbors) < 2:
            continue
            
        # Sample pairs from neighbors
        # Limit to a few per node to distribute them
        # E.g. connect siblings
        # If huge degree, don't fully connect (clique explosion), just pick a few random pairs.
        
        # Pick up to 5 pairs per pivot
        num_pairs = min(5, len(neighbors) * (len(neighbors)-1) // 2)
        # Simply shuffle neighbors and pick pairs
        
        # Just pick 2 random neighbors
        for _ in range(2): # Try 2 attempts per node
            if len(neighbors) < 2: break
            
            u, v = random.sample(neighbors, 2)
            
            # Check if edge exists? (dataset doesn't have O(1) check, but sparse enough)
            # Add semantic edge (u, v)
            # Map indices back to words?
            # ds.load_semantic_edges expects STRINGS or INTs?
            # My previous impl handles both, but usually raw format is strings.
            # However, AuroraDataset.load_semantic_edges (Line 147) expects (u_s, v_s, w).
            # And it attempts `self.vocab[u_s]`.
            # If I pass IDs directly, `self.vocab[int]` -> returns STR usually if implemented standardly?
            # Wait, `Vocabulary.__getitem__`: if int, returns word string.
            # BUT `load_semantic_edges` does: `u_id = self.vocab[u_s]`.
            # If `u_s` is int, `vocab[int]` returns str? 
            # Then `u_id` becomes `vocab[str]` -> int.
            # So if I pass INTs, it might crash or work depending on `vocab` implementation.
            # Let's save STRINGS to be safe and match standard format.
            
            u_str = ds.nodes[u]
            v_str = ds.nodes[v]
            
            if u_str == v_str: continue
            
            edges.append((u_str, v_str, 0.5))
            count += 1
            if count >= num_edges: break
            
    print(f"Generated {len(edges)} triangle-closing edges.")
    
    out_path = args.output
    with open(out_path, 'wb') as f:
        pickle.dump(edges, f)
        
    print(f"Saved to {out_path}")
    
if __name__ == "__main__":
    main()

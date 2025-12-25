import pickle
import sys

path = "/home/void0312/HEI-Research/HEI/checkpoints/semantic_edges_wiki.pkl"
try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} items from {path}")
    print("Sample:", data[:5])
except Exception as e:
    print(e)

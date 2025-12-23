
import pickle
import sys

path = "checkpoints/aurora_base_gpu_cilin_full.pkl"
try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Keys: {list(data.keys())}")
    if 'model' in data:
         print(f"Model keys: {list(data['model'].keys())}")
    # Check what 'state' might be
    for k, v in data.items():
        print(f"{k}: {type(v)}")
except Exception as e:
    print(e)

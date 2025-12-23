import pickle
import numpy as np
import os

def load_semantic_edges(node_map):
    """
    Load semantic edges (PMI/Def) from pickles.
    Returns: edges_pmi (np array), edges_def (np array)
    """
    paths = ["checkpoints/semantic_edges.pkl", "checkpoints/semantic_edges_wiki.pkl"]
    
    pmi_list = []
    def_list = []
    
    for path in paths:
        if not os.path.exists(path):
            if "wiki" not in path: # Warn only for base mock file if missing
                print(f"Warning: {path} not found.")
            continue
            
        print(f"Loading Semantic Edges from {path}...")
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            count_local = 0
            for u, v, w, type_id in data:
                if type_id == 2:
                    pmi_list.append([u, v])
                elif type_id == 3:
                    def_list.append([u, v])
                count_local += 1
            print(f"  Loaded {count_local} edges from {path}.")
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    edges_pmi = np.array(pmi_list, dtype=np.int64) if pmi_list else None
    edges_def = np.array(def_list, dtype=np.int64) if def_list else None
    
    print(f"Total: {len(pmi_list) if pmi_list else 0} PMI edges, {len(def_list) if def_list else 0} Definition edges.")
    return edges_pmi, edges_def

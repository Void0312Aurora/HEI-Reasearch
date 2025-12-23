import pickle
import numpy as np
import networkx as nx
import re

def load_dataset(limit=None):
    """
    Load OpenHowNet Dataset.
    Structure: Root -> Sememes -> Concepts
    """
    print("Loading OpenHowNet Dataset...", flush=True)
    dict_path = "data/openhownet/dict_computed.pkl"
    sememe_path = "data/openhownet/all_sememes.txt"
    
    G = nx.DiGraph()
    
    # 1. Load Sememes
    with open(sememe_path, 'r', encoding='utf-8') as f:
        sememe_nodes = [line.strip() for line in f if line.strip()]
        
    for s in sememe_nodes:
        G.add_node(s, type='sememe')
        
    # 2. Load Dictionary (Concept -> Sememes)
    with open(dict_path, 'rb') as f:
        hownet_dict = pickle.load(f)
        
    pattern = re.compile(r'\{([a-zA-Z]+)\|')
    sememe_map = {}
    for node in sememe_nodes:
        if '|' in node:
            eng = node.split('|')[0]
            sememe_map[eng] = node
        else:
            sememe_map[node] = node
            
    keys = list(hownet_dict.keys())
    if limit:
        keys = keys[:limit]
        
    for k in keys:
        item = hownet_dict[k]
        en_word = item.get('en_word', f"Concept_{k}")
        definition = item.get('Def', '')
        
        match = pattern.search(definition)
        if match:
            head_sememe_key = match.group(1)
            if head_sememe_key in sememe_map:
                sememe_node = sememe_map[head_sememe_key]
                concept_node = f"C:{en_word}:{k}"
                G.add_edge(sememe_node, concept_node)
    
    # Root
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
         roots = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:1]
    root = roots[0]
    print(f"Graph: {G.number_of_nodes()} nodes. Root: {root}", flush=True)
    
    node_list = list(G.nodes)
    node_map = {n: i for i, n in enumerate(node_list)}
    edges = np.array([[node_map[u], node_map[v]] for u, v in G.edges()], dtype=np.int64)
    
    # Depths
    try:
        lens = nx.shortest_path_length(G, source=root)
        depths = np.zeros(len(node_list))
        for i, n in enumerate(node_list):
            if n in lens:
                depths[i] = lens[n]
    except:
        depths = np.zeros(len(node_list))
        
    return node_list, edges, depths, node_map[root]

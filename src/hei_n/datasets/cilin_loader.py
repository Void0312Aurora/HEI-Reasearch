import networkx as nx
import numpy as np

def load_cilin_dataset(limit=None):
    """
    Load Cilin (Tongyici Cilin) Dataset.
    Structure: Tree (Root -> Level 1 -> 2 -> 3 -> 4 -> 5 -> Words)
    """
    print("Loading Cilin Dataset...", flush=True)
    cilin_path = "data/cilin/new_cilin.txt"
    encoding = 'gb18030'
    
    G = nx.DiGraph()
    root = "CilinRoot"
    G.add_node(root, type='root')
    
    # Track existing nodes to avoid duplicates
    existing_nodes = {root}
    
    count = 0
    with open(cilin_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split()
            code = parts[0]
            words = parts[1:]
            
            # 1. Build Code Hierarchy
            # Levels:
            # L1: A (1 char)
            # L2: Aa (2 chars)
            # L3: Aa01 (4 chars)
            # L4: Aa01A (5 chars)
            # L5: Aa01A01= (8 chars)
            
            # Determine hierarchy path for this code
            hierarchy = []
            
            # Level 1
            if len(code) >= 1:
                l1 = code[:1]
                hierarchy.append(l1)
                
            # Level 2
            if len(code) >= 2:
                l2 = code[:2]
                hierarchy.append(l2)
                
            # Level 3
            if len(code) >= 4:
                l3 = code[:4]
                hierarchy.append(l3)
                
            # Level 4
            if len(code) >= 5:
                l4 = code[:5]
                hierarchy.append(l4)
                
            # Level 5 (Leaf Code)
            if len(code) >= 8:
                l5 = code  # Includes marker like = or #
                hierarchy.append(l5)
            
            # Add hierarchy to graph
            prev = root
            for h_code in hierarchy:
                node_name = f"Code:{h_code}"
                if node_name not in existing_nodes:
                    G.add_node(node_name, type='code')
                    existing_nodes.add(node_name)
                    
                # Link parent -> child
                if not G.has_edge(prev, node_name):
                    G.add_edge(prev, node_name)
                prev = node_name
                
            # 2. Add Words (linked to the specific code of this line)
            # The code at the start of the line is the immediate parent of these words
            leaf_code_node = f"Code:{code}"
            
            for word in words:
                word_node = f"C:{word}:{count}" # Unique ID to allow polysemy if needed?
                # Actually, in HEI we usually want unique particles per word string OR unique concepts.
                # In Cilin, one word can appear in multiple categories (polysemy).
                # To support interaction, mapping text->ID is easier if unique word, OR we handle polysemy.
                # Aurora Base approach: Unique Nodes.
                # If "apple" is in two categories, do we merge or split?
                # OpenHowNet split them (C:apple:001, C:apple:002).
                # Let's split them here too to capture polysemy structure.
                
                G.add_node(word_node, type='word', text=word)
                G.add_edge(leaf_code_node, word_node)
                count += 1
                
                if limit and count >= limit:
                    break
            
            if limit and count >= limit:
                break
                
    print(f"Graph: {G.number_of_nodes()} nodes. Words: {count}", flush=True)
    
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

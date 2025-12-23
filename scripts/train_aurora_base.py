
import numpy as np
import sys
import os
import networkx as nx
import argparse
import time
import pickle
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.inertia_n import IdentityInertia
from hei_n.contact_integrator_n import ContactIntegratorN, ContactConfigN, ContactStateN
from hei_n.potential_n import CompositePotentialN, HarmonicPriorN
from hei_n.sparse_potential_n import SparseEdgePotential, NegativeSamplingPotential
from hei_n.kernels_n import SpringAttraction, LogCoshRepulsion

def load_dataset(limit=None):
    """
    Load Sememe Taxonomy and OpenHowNet Concepts to build a single graph.
    Structure:
    - Sememe Nodes (Backbone)
    - Concept Nodes (Leaves, attached to Head Sememe)
    """
    print("Loading Datasets...", flush=True)
    
    # 1. Load Sememe Taxonomy (Skeleton)
    sememe_path = "data/openhow/resources/sememe_triples_taxonomy.txt"
    G = nx.DiGraph()
    with open(sememe_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            u, rel, v = parts[0], parts[1], parts[2] # u hypernym v (v is parent?)
            # Usually: SememeA hypernym SememeB -> SememeB is child of SememeA
            # Check direction.
            # Tree: Root -> Child.
            # If u is hypernym of v, then u is more general. u -> v?
            # Let's assume u -> v (Parent -> Child) or v -> u (Child -> Parent).
            # Gate C assumed directed edges.
            # Let's just add edges. Physics is symmetric for attraction usually.
            G.add_edge(u, v)
    
    sememe_nodes = set(G.nodes)
    print(f"Sememe Skeleton: {len(sememe_nodes)} nodes.", flush=True)
    
    # 2. Load Concepts from HowNet Dict
    dict_path = "data/openhow/resources/HowNet_dict_complete"
    print(f"Loading Concepts from {dict_path}...", flush=True)
    
    with open(dict_path, 'rb') as f:
        hownet_dict = pickle.load(f)
        
    print(f"Total Concepts in Dict: {len(hownet_dict)}", flush=True)
    
    # Regex to extract Head Sememe: {HeadSememe|...
    # Examples: {FuncWord|...}, {human|人...}
    # Pattern: {([^{}|]+)
    pattern = re.compile(r'\{([a-zA-Z]+)\|')
    
    added_concepts = 0
    concept_edges = []
    
    keys = list(hownet_dict.keys())
    if limit:
        keys = keys[:limit]
        
    for k in keys:
        item = hownet_dict[k]
        en_word = item.get('en_word', '')
        definition = item.get('Def', '')
        
        # Parse Head Sememe
        match = pattern.search(definition)
        if match:
            head_sememe = match.group(1)
            # Sememe might be capitalized differently or mapping mismatch?
            # OpenHowNet Sememes in taxonomy file seem to be just names like "AttributeValue|属性值".
            # My regex extracts "AttributeValue".
            # I need to match it to the nodes in G.
            
            # Find matching node in G
            # G nodes are "Name|ChineseName" or just "Name"?
            # Let's check G nodes format from Gate C logs: "AttributeValue|属性值"
            # So I need to match prefix.
            
            # Optimization: Build Map Key->FullNode
            # Do it once.
            pass
        else:
            continue
            
    # Build Sememe Map
    sememe_map = {}
    for node in sememe_nodes:
        # Node format "English|Chinese"
        if '|' in node:
            eng = node.split('|')[0]
            sememe_map[eng] = node
        else:
            sememe_map[node] = node
            
    # Retry Concept Loop
    concepts_added = []
    
    for k in keys:
        item = hownet_dict[k]
        en_word = item.get('en_word', f"Concept_{k}")
        definition = item.get('Def', '')
        
        match = pattern.search(definition)
        if match:
            head_sememe_key = match.group(1)
            if head_sememe_key in sememe_map:
                sememe_node = sememe_map[head_sememe_key]
                
                # Create Concept Node
                # Unique ID: "Concept:en_word:ID"
                concept_node = f"C:{en_word}:{k}"
                
                G.add_node(concept_node, type='concept')
                G.add_edge(sememe_node, concept_node) # Edge from Sememe to Concept
                
                concepts_added.append(concept_node)
                added_concepts += 1
            else:
                # Sememe not in skeleton?
                pass
                
    print(f"Graph Built: {G.number_of_nodes()} nodes (Concepts: {added_concepts}).", flush=True)
    
    # Find Root (Sememe Root)
    # Roots have in-degree 0 (if edges are parent->child)
    # Check taxonomy file direction:
    # u hypernym v. Hypernym is Parent. u -> v.
    # Roots = 0 in-degree.
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
         roots = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:1]
    root = roots[0]
    print(f"Global Root: {root}", flush=True)
    
    # Convert to Indices
    node_list = list(G.nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    edges = []
    for u, v in G.edges():
        edges.append((node_to_idx[u], node_to_idx[v]))
    edges = np.array(edges, dtype=np.int32)
    
    # Calculate depths
    print("Calculating depths...", flush=True)
    try:
        lens = nx.shortest_path_length(G, source=root)
        depths = np.zeros(len(node_list))
        for i, n in enumerate(node_list):
            if n in lens:
                depths[i] = lens[n]
    except:
        depths = np.zeros(len(node_list))
        
    root_idx = node_to_idx[root]
    
    return node_list, edges, depths, root_idx

def train(args):
    # 1. Data Setup
    nodes, edges, depths, root_idx = load_dataset(limit=args.limit)
    N = len(nodes)
    dim = args.dim
    
    # 2. Physics Initialization
    print(f"Initializing {N} particles in H^{dim}...", flush=True)
    np.random.seed(args.seed)
    
    G_mat = np.zeros((N, dim+1, dim+1))
    M = np.zeros((N, dim+1, dim+1))
    
    scale_init = 0.05 # Conservative init for deep trees
    
    for i in range(N):
        d_i = depths[i]
        r = scale_init * d_i + np.random.uniform(0, 0.01)
        
        dir_vec = np.random.randn(dim)
        dir_vec /= np.linalg.norm(dir_vec) + 1e-9
        
        ch = np.cosh(r); sh = np.sinh(r)
        
        G_mat[i] = np.eye(dim+1)
        G_mat[i, 0, 0] = ch
        G_mat[i, 0, 1:] = sh * dir_vec
        G_mat[i, 1:, 0] = sh * dir_vec
        G_mat[i, 1:, 1:] = np.eye(dim) + (ch - 1) * np.outer(dir_vec, dir_vec)
        
        v = np.random.randn(dim) * 0.01
        M[i, 0, 1:] = v; M[i, 1:, 0] = v

    state = ContactStateN(G=G_mat, M=M, z=0.0)
    
    # 3. Potentials (Weak Bond)
    print("Configuring Potentials...", flush=True)
    k_attract = SpringAttraction(k=1.0)
    attract = SparseEdgePotential(edges, k_attract, k_attract.derivative)
    
    k_repulse = LogCoshRepulsion(sigma=1.0, A=5.0) 
    # Negative Sampling: 10 samples per node.
    repulse = NegativeSamplingPotential(k_repulse, k_repulse.derivative, num_neg=10, rescale=1.0)
    
    trap = HarmonicPriorN(k=0.05)
    
    oracle = CompositePotentialN([attract, repulse, trap])
    inertia = IdentityInertia()
    
    # 4. Integrator
    config = ContactConfigN(
        dt=args.dt, 
        gamma=2.0, 
        target_temp=0.1, 
        thermostat_tau=5.0,
        fixed_point_iters=3,
        solver_mixing=0.5,
        torque_clip=50.0,
        renorm_interval=50,
        adaptive=True,
        tol_disp=0.2
    )
    
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    # 5. Training Loop
    print(f"Starting Training: {args.steps} steps...", flush=True)
    start_time = time.time()
    
    for step in range(args.steps):
        step_start = time.time()
        
        state.M[root_idx] = 0.0
        state = integrator.step(state)
        state.M[root_idx] = 0.0
        
        step_end = time.time()
        step_dur = step_end - step_start
        
        if step % args.log_interval == 0:
            diag = state.diagnostics
            
            # Fast Metrics (Sample 500)
            idx_sample = np.random.choice(N, size=min(N, 500), replace=False)
            x_sample = state.x[idx_sample]
            radii = np.arccosh(np.maximum(x_sample[:, 0], 1.0))
            mean_R = np.mean(radii)
            
            err = diag.get('manifold_error', 0.0)
            res = diag.get('solver_residual', 0.0)
            dt_used = diag.get('dt', args.dt)
            
            elapsed = time.time() - start_time
            print(f"Step {step}: T={elapsed:.1f}s | dt={dt_used:.1e} | R={mean_R:.2f} | Err={err:.1e} | Res={res:.1e} | {step_dur*1000:.1f}ms/step", flush=True)
            
    print("Training Complete.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Concept limit")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    
    args = parser.parse_args()
    train(args)

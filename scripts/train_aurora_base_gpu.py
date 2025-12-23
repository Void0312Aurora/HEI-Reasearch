
import torch
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

from hei_n.torch_core import minkowski_metric_torch
from hei_n.integrator_torch import ContactIntegratorTorch, ContactConfigTorch, ContactStateTorch, IdentityInertiaTorch
from hei_n.potentials_torch import SparseEdgePotentialTorch, NegativeSamplingPotentialTorch, HarmonicPriorTorch, CompositePotentialTorch, SpringAttractionTorch, LogCoshRepulsionTorch

# Reuse Load Logic (or duplicated for standalone)
def load_dataset(limit=None):
    """
    Load Sememe Taxonomy and OpenHowNet Concepts (CPU Preprocessing).
    """
    print("Loading Datasets (CPU)...", flush=True)
    
    # 1. Load Sememe Taxonomy
    sememe_path = "data/openhow/resources/sememe_triples_taxonomy.txt"
    G = nx.DiGraph()
    with open(sememe_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            u, rel, v = parts[0], parts[1], parts[2]
            G.add_edge(u, v)
    
    sememe_nodes = set(G.nodes)
    
    # 2. Load Concepts
    dict_path = "data/openhow/resources/HowNet_dict_complete"
    print(f"Loading concepts from {dict_path}...", flush=True)
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


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}", flush=True)
    
    # 1. Load Data
    if args.dataset == 'cilin':
        nodes, edges, depths, root_idx = load_cilin_dataset(limit=args.limit)
    else:
        nodes, edges, depths, root_idx = load_dataset(limit=args.limit)
        
    N = len(nodes)
    dim = args.dim
    
    # 2. Physics Init (CPU -> GPU)
    print(f"Initializing {N} particles...", flush=True)
    np.random.seed(args.seed)
    
    G_np = np.zeros((N, dim+1, dim+1), dtype=np.float32)
    M_np = np.zeros((N, dim+1, dim+1), dtype=np.float32)
    
    scale_init = 0.05
    for i in range(N):
        d_i = depths[i]
        r = scale_init * d_i + np.random.uniform(0, 0.01)
        
        dir_vec = np.random.randn(dim)
        dir_vec /= np.linalg.norm(dir_vec) + 1e-9
        
        ch = np.cosh(r); sh = np.sinh(r)
        B = np.eye(dim+1)
        B[0,0] = ch
        B[0,1:] = sh * dir_vec
        B[1:,0] = sh * dir_vec
        B[1:,1:] = np.eye(dim) + (ch - 1) * np.outer(dir_vec, dir_vec)
        
        G_np[i] = B
        v = np.random.randn(dim) * 0.01
        M_np[i, 0, 1:] = v; M_np[i, 1:, 0] = v
        
    # Transfer to GPU
    G = torch.tensor(G_np, device=device)
    M = torch.tensor(M_np, device=device)
    z = torch.tensor(0.0, device=device)
    edges_gpu = torch.tensor(edges, device=device, dtype=torch.long)
    
    state = ContactStateTorch(G=G, M=M, z=z)
    
    # 3. Potentials (GPU)
    print("Configuring GPU Potentials...", flush=True)
    
    # Kernel for Skeleton (Base)
    k_attract_skel = SpringAttractionTorch(k=1.0)
    attract_skel = SparseEdgePotentialTorch(edges_gpu, k_attract_skel)
    
    potentials = [attract_skel]
    
    # Semantic Potentials (Phase II)
    k_attract_pmi = None
    k_attract_def = None
    
    if args.semantic:
        edges_pmi, edges_def = load_semantic_edges(None) # IDs assumed stable
        
        if edges_pmi is not None:
            edges_pmi_gpu = torch.tensor(edges_pmi, device=device, dtype=torch.long)
            k_attract_pmi = SpringAttractionTorch(k=0.0) # Start 0 (Calibration)
            attract_pmi = SparseEdgePotentialTorch(edges_pmi_gpu, k_attract_pmi)
            potentials.append(attract_pmi)
            print("Added PMI Potential (Init k=0.0)")
            
        if edges_def is not None:
            edges_def_gpu = torch.tensor(edges_def, device=device, dtype=torch.long)
            k_attract_def = SpringAttractionTorch(k=0.0) # Start 0 (Calibration)
            attract_def = SparseEdgePotentialTorch(edges_def_gpu, k_attract_def)
            potentials.append(attract_def)
            print("Added Definition Potential (Init k=0.0)")

    k_repulse = LogCoshRepulsionTorch(sigma=1.0, A=5.0)
    repulse = NegativeSamplingPotentialTorch(k_repulse, num_neg=10, rescale=1.0)
    potentials.append(repulse)
    
    trap = HarmonicPriorTorch(k=0.05)
    potentials.append(trap)
    
    oracle = CompositePotentialTorch(potentials)
    inertia = IdentityInertiaTorch()
    
    # 4. Integrator
    config = ContactConfigTorch(
        dt=args.dt,
        gamma=2.0,
        target_temp=0.1,
        thermostat_tau=5.0,
        fixed_point_iters=3,
        solver_mixing=0.5,
        torque_clip=50.0,
        renorm_interval=50,
        adaptive=True,
        tol_disp=0.2,
        device=device
    )
    
    integrator = ContactIntegratorTorch(oracle, inertia, config)
    
    # 5. Loop
    print(f"Starting GPU Training: {args.steps} steps...", flush=True)
    start_time = time.time()
    
    # Torch benchmark warm-up
    for _ in range(5):
        pass
        
    residuals = []
    step_times = []
    renorms = []
    
    print(f"Starting GPU Training: {args.steps} steps...", flush=True)
    
    for step in range(args.steps):
        # Annealing Schedule (Phase II)
        if args.semantic:
            # Stage 0: 0-20% -> k=0
            # Stage 1: 20-40% -> k=0.01
            # Stage 2: 40-60% -> k=0.03
            # Stage 3: 60-100% -> k=0.05
            progress = step / args.steps
            target_k = 0.0
            stage = 0
            
            # 4-Stage Stepped Ramp-up for Full Scale Controlled Rollout
            if progress > 0.75:
                target_k = 0.50 # Full Force (1500-2000 steps)
                stage = 3
            elif progress > 0.50:
                target_k = 0.35 # Probe Point B (1000-1500 steps)
                stage = 2
            elif progress > 0.25:
                target_k = 0.20 # Probe Point A (500-1000 steps)
                stage = 1
            else:
                target_k = 0.00 # Calibration (0-500 steps)
                stage = 0
                
            if k_attract_pmi: k_attract_pmi.k = target_k
            if k_attract_def: k_attract_def.k = target_k * 1.0 # Similar weight
            
            # Log stage change
            if step % args.log_interval == 0:
                # Using a special print to distinguish phase transitions
                steps_per_stage = args.steps // 4
                if step in [steps_per_stage, 2 * steps_per_stage, 3 * steps_per_stage]:
                     print(f"\n>>> PHASE TRANSITION: Stage {stage} | Weight k={target_k:.3f} <<<\n", flush=True)

        step_start = time.time()
        
        # Pin Root
        state.M[root_idx] = 0.0
        state = integrator.step(state)
        state.M[root_idx] = 0.0
        
        torch.cuda.synchronize() # Wait for GPU
        step_end = time.time()
        step_dur = step_end - step_start
        
        # Collect Stats
        diag = state.diagnostics
        residuals.append(diag.get('solver_residual', 0.0))
        renorms.append(diag.get('renorm_magnitude', 0.0))
        step_times.append(step_dur * 1000.0) # ms
        
        if step % args.log_interval == 0:
            # Metrics (GPU -> CPU for print)
            with torch.no_grad():
                idx = torch.randint(0, N, (min(N, 1000),), device=device)
                x_sample = state.x[idx]
                radii = torch.acosh(torch.clamp(x_sample[:, 0], min=1.0))
                mean_R = torch.mean(radii).item()
            
            res = residuals[-1]
            renorm = renorms[-1]
            dt_used = diag.get('dt', args.dt)
            elapsed = time.time() - start_time
            
            print(f"Step {step}: T={elapsed:.1f}s | dt={dt_used:.1e} | R={mean_R:.2f} | Res={res:.1e} | Renorm={renorm:.1e} | {step_dur*1000:.1f}ms/step", flush=True)
            if args.semantic and step % (args.log_interval * 5) == 0:
                pmi_k = k_attract_pmi.k if k_attract_pmi else 0
                print(f"  [Anneal] Semantic Weights: {pmi_k:.3f}", flush=True)
            
    print("Training Complete.", flush=True)
    
    # Gate Analysis
    res_arr = np.array(residuals)
    time_arr = np.array(step_times)
    ren_arr = np.array(renorms)
    
    res_p99 = np.percentile(res_arr, 99)
    time_p99 = np.percentile(time_arr, 99)
    time_mean = np.mean(time_arr)
    ren_max = np.max(ren_arr)
    
    print("\n=== GATE CHECK REPORT ===")
    print(f"Metric\t\tValue\t\tThreshold\tStatus")
    print(f"Residual P99\t{res_p99:.2e}\t< 5.00e-03\t{'PASS' if res_p99 < 5e-3 else 'FAIL'}")
    print(f"Renorm Max\t{ren_max:.2e}\t< 1.00e-04\t{'PASS' if ren_max < 1e-4 else 'FAIL'}")
    print(f"Throughput P99\t{time_p99:.1f}ms\t< {2*time_mean:.1f}ms\t{'PASS' if time_p99 < 2*time_mean else 'WARN'}")
    print("=========================")
    
    # Save
    if args.save:
        save_path = f"checkpoints/aurora_base_gpu_{args.dataset}_{args.limit or 'full'}.pkl"
        depths_path = f"checkpoints/aurora_base_gpu_{args.dataset}_{args.limit or 'full'}_depths.npy"
        os.makedirs("checkpoints", exist_ok=True)
        # Convert to CPU numpy
        G_cpu = state.G.detach().cpu().numpy()
        M_cpu = state.M.detach().cpu().numpy()
        
        with open(save_path, 'wb') as f:
            pickle.dump({'G': G_cpu, 'M': M_cpu, 'nodes': nodes, 'edges': edges}, f)
        print(f"Saved to {save_path}", flush=True)
        
        # Save depths for Semantic Audit
        np.save(depths_path, depths)
        print(f"Saved depths to {depths_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--dataset", type=str, default="openhow", choices=["openhow", "cilin"], help="Dataset to use")
    parser.add_argument("--semantic", action="store_true", help="Enable Phase II Semantic Edges")
    
    args = parser.parse_args()
    train(args)

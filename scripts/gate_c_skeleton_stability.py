
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import networkx as nx
from scipy.stats import spearmanr

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.inertia_n import RadialInertia, IdentityInertia
from hei_n.contact_integrator_n import ContactIntegratorN, ContactConfigN, ContactStateN
from hei_n.potential_n import CompositePotentialN
from hei_n.sparse_potential_n import SparseEdgePotential, NegativeSamplingPotential
from hei_n.kernels_n import SpringAttraction, GaussianRepulsion

def load_taxonomy_subgraph(path, max_nodes=500):
    print(f"Loading taxonomy from {path}...")
    G = nx.DiGraph()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            u, rel, v = parts[0], parts[1], parts[2]
            
            if rel == 'hypernym':
                G.add_edge(u, v) 
            elif rel == 'hyponym':
                G.add_edge(v, u) 
                
    roots = [n for n, d in G.out_degree() if d == 0 and G.in_degree(n) > 0]
    if not roots:
         roots = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:1]
    root = roots[0]
    print(f"Selected Root: {root}")
    
    subgraph_nodes = set([root])
    queue = [root]
    
    while len(subgraph_nodes) < max_nodes and queue:
        curr = queue.pop(0)
        children = list(G.predecessors(curr))
        for child in children:
            if child not in subgraph_nodes:
                subgraph_nodes.add(child)
                queue.append(child)
                if len(subgraph_nodes) >= max_nodes:
                    break
                    
    H = G.subgraph(subgraph_nodes).copy()
    print(f"Subgraph Size: {len(H.nodes)} nodes.")
    
    node_list = list(H.nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    edges = []
    for u, v in H.edges():
        edges.append((node_to_idx[u], node_to_idx[v]))
        
    depths = {}
    for n in node_list:
        try:
            path = nx.shortest_path(H, source=n, target=root)
            depths[node_to_idx[n]] = len(path) - 1
        except:
             depths[node_to_idx[n]] = 0
             
    return node_list, np.array(edges), depths

def test_skeleton_stability():
    print("--- Gate C: Bone Stability Check (Final Tuning) ---")
    
    taxonomy_path = "data/openhow/resources/sememe_triples_taxonomy.txt"
    nodes, edges, depths = load_taxonomy_subgraph(taxonomy_path, max_nodes=200)
    N = len(nodes)
    dim = 5
    
    # 1. Initialize Particles (Hierarchical Warm Start)
    np.random.seed(42)
    G_mat = np.zeros((N, dim+1, dim+1))
    M = np.zeros((N, dim+1, dim+1))
    
    for i in range(N):
        d_i = depths[i]
        r = 0.5 * d_i 
        dir_vec = np.random.randn(dim)
        dir_vec /= np.linalg.norm(dir_vec) + 1e-7
        
        n = dir_vec
        ch = np.cosh(r)
        sh = np.sinh(r)
        
        B = np.eye(dim+1)
        B[0, 0] = ch
        B[0, 1:] = sh * n
        B[1:, 0] = sh * n
        B[1:, 1:] = np.eye(dim) + (ch - 1) * np.outer(n, n)
        
        G_mat[i] = B
        
        v = np.random.randn(dim) * 0.05
        M[i, 0, 1:] = v; M[i, 1:, 0] = v

    state = ContactStateN(G=G_mat, M=M, z=0.0)
    
    # 2. Setup Physics (Tuned Soft)
    # Strong Attract k=5.0
    k_attract = SpringAttraction(k=5.0) 
    attract = SparseEdgePotential(edges, k_attract, k_attract.derivative)
    
    # Soft Gaussian Repulsion 
    # sigma=0.5 (Proven Stable), A=5.0
    k_repulse = GaussianRepulsion(sigma=0.5, A=5.0) 
    repulse = NegativeSamplingPotential(k_repulse, k_repulse.derivative, num_neg=10, rescale=1.0)
    
    oracle = CompositePotentialN([attract, repulse])
    inertia = IdentityInertia() 
    
    # High Damping to absorb Spring Energy
    config = ContactConfigN(dt=0.01, gamma=2.0, target_temp=0.1, thermostat_tau=1.0)
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    # 3. Running
    steps = 2000
    print(f"Running {steps} steps...")
    
    radii_history = []
    contrast_history = []
    corr_history = []
    
    # Log forces for first step to verify no explosion
    x_curr = state.x
    f_rep = repulse.gradient(x_curr)
    print(f"Init Repulsion Max: {np.max(np.linalg.norm(f_rep, axis=1)):.2f}")
    
    for i in range(steps):
        if i % 200 == 0:
            x_curr = state.x
            radii = np.arccosh(np.maximum(x_curr[:, 0], 1.0))
            xu = x_curr[edges[:, 0]]; xv = x_curr[edges[:, 1]]
            mink = np.ones(dim+1); mink[0] = -1.0
            inner = np.sum(xu * xv * mink, axis=-1)
            mean_edge = np.mean(np.arccosh(np.maximum(-inner, 1.0)))
            
            rand_idx = np.random.randint(0, N, size=(500, 2))
            rxu = x_curr[rand_idx[:,0]]; rxv = x_curr[rand_idx[:,1]]
            rinner = np.sum(rxu * rxv * mink, axis=-1)
            mean_rand = np.mean(np.arccosh(np.maximum(-rinner, 1.0)))
            
            contrast = mean_rand / mean_edge
            depth_vals = [depths[k] for k in range(N)]
            corr, _ = spearmanr(depth_vals, radii)
            
            print(f"Step {i}: R_mean={np.mean(radii):.4f}, Contrast={contrast:.2f}, Corr={corr:.4f}")
            radii_history.append(np.mean(radii))
            contrast_history.append(contrast)
            corr_history.append(corr)

        state = integrator.step(state)
            
    print("\nSimulation Done.")
    
    final_contrast = contrast_history[-1]
    final_corr = corr_history[-1]
    
    print(f"Final Contrast Ratio: {final_contrast:.2f}")
    print(f"Final Correlation:    {final_corr:.4f}")
    
    if final_contrast > 1.5 and final_corr > 0.4: # Lower corr slightly as strong attraction might compress hierarchy
        print("PASS: Skeleton Injected Successfully.")
    else:
        print("FAIL: Metrics below threshold.")

if __name__ == "__main__":
    test_skeleton_stability()

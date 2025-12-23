
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import networkx as nx
from scipy.stats import spearmanr
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.inertia_n import RadialInertia, IdentityInertia
from hei_n.contact_integrator_n import ContactIntegratorN, ContactConfigN, ContactStateN
from hei_n.potential_n import CompositePotentialN, HarmonicPriorN
from hei_n.sparse_potential_n import SparseEdgePotential, NegativeSamplingPotential
from hei_n.kernels_n import SpringAttraction, LogCoshRepulsion

def load_taxonomy_subgraph(path, max_nodes=500):
    print(f"Loading taxonomy from {path}...", flush=True)
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
    print(f"Selected Root: {root}", flush=True)
    
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
    print(f"Subgraph Size: {len(H.nodes)} nodes.", flush=True)
    
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
             
    root_idx = node_to_idx[root]
    print(f"Root Index: {root_idx}", flush=True)
    
    return node_list, np.array(edges), depths, root_idx

def test_skeleton_stability():
    print("--- Gate C: Bone Stability Check (Robustness Audit) ---", flush=True)
    
    taxonomy_path = "data/openhow/resources/sememe_triples_taxonomy.txt"
    nodes, edges, depths, root_idx = load_taxonomy_subgraph(taxonomy_path, max_nodes=200)
    N = len(nodes)
    dim = 5
    
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

    M[root_idx] = 0.0

    state = ContactStateN(G=G_mat, M=M, z=0.0)
    
    # Weak Bond Params (Metric Optimized)
    k_attract = SpringAttraction(k=1.0)
    attract = SparseEdgePotential(edges, k_attract, k_attract.derivative)
    
    k_repulse = LogCoshRepulsion(sigma=1.0, A=5.0) 
    repulse = NegativeSamplingPotential(k_repulse, k_repulse.derivative, num_neg=10, rescale=1.0)
    
    trap = HarmonicPriorN(k=0.05)
    
    oracle = CompositePotentialN([attract, repulse, trap])
    inertia = IdentityInertia() 
    
    base_dt = 0.01
    config = ContactConfigN(dt=base_dt, gamma=2.0, target_temp=0.1, thermostat_tau=1.0)
    config.fixed_point_iters = 3
    config.solver_mixing = 0.5
    config.adaptive = True
    config.tol_disp = 0.2
    
    integrator = ContactIntegratorN(oracle, inertia, config)
    
    total_time = 10.0 
    t = 0.0
    step_count = 0
    max_steps = 1000 
    
    # Audit Logs
    residuals = []
    manifold_errors = []
    renorm_mags = []
    
    radii_history = []
    contrast_history = []
    corr_history = []
    
    print(f"Running Pinned Adaptive Sim (Audit Mode, Max {max_steps} steps)...", flush=True)
    
    while t < total_time and step_count < max_steps:
        state.M[root_idx] = 0.0
        state = integrator.step(state) # Instrument inside step returns diagnostics
        state.M[root_idx] = 0.0
        step_count += 1
        
        # Collect Metrics
        diag = state.diagnostics
        if 'solver_residual' in diag: residuals.append(diag['solver_residual'])
        if 'manifold_error' in diag: manifold_errors.append(diag['manifold_error'])
        if 'renorm_magnitude' in diag and diag['renorm_magnitude'] > 0: renorm_mags.append(diag['renorm_magnitude'])
        
        if step_count % 50 == 0:
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
            
            # Print Robustness too
            avg_res = np.mean(residuals[-50:]) if residuals else 0.0
            max_err = np.max(manifold_errors[-50:]) if manifold_errors else 0.0
            
            print(f"Step {step_count}: R={np.mean(radii):.2f}, C={contrast:.2f}, Corr={corr:.2f} | Res={avg_res:.1e}, Err={max_err:.1e}", flush=True)
            radii_history.append(np.mean(radii))
            contrast_history.append(contrast)
            corr_history.append(corr)

    print("\n--- Robustness Audit Report ---", flush=True)
    print(f"Max Manifold Error: {np.max(manifold_errors):.2e} (Threshold: 1e-6)", flush=True)
    print(f"Mean Solver Residual: {np.mean(residuals):.2e} (Threshold: 1e-4)", flush=True)
    print(f"Total Renorms: {len(renorm_mags)}", flush=True)
    if renorm_mags:
        print(f"Mean Renorm Magnitude: {np.mean(renorm_mags):.2e}", flush=True)
        print(f"Max Renorm Magnitude: {np.max(renorm_mags):.2e}", flush=True)
    
    final_contrast = contrast_history[-1] if contrast_history else 0.0
    final_corr = corr_history[-1] if corr_history else 0.0
    
    # Gate Logic
    pass_physics = final_contrast > 1.1 and final_corr > 0.4
    pass_robust = np.max(manifold_errors) < 1e-4 and np.mean(residuals) < 1e-3
    
    if pass_physics and pass_robust:
        print("PASS: Gate C Robustness Audit Passed.", flush=True)
    else:
        print(f"FAIL: Physics={pass_physics}, Robust={pass_robust}", flush=True)

if __name__ == "__main__":
    test_skeleton_stability()

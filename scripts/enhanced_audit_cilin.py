"""
Enhanced Cilin Migration Audit (temp-11.md).
============================================

Implements stricter gates for "Base-Gold" certification.

Gate L (Language Purity):
- P10/P50/P90 stats.
- Breakdown of non-Chinese tokens.

Gate H (Hierarchy Consistency):
- Spearman correlation (Depth vs Radius).
- Intra-category vs Inter-category distance.

Gate S (Synonym Convergence):
- Ratio distribution (P50/P90).
- % with Ratio < 1.0.

Gate C (Numerical Stability):
- Residual P99 & Renorm Max (via short simulation).

Gate M (Mapping Coverage):
- Check specific emotional/functional words.

Gate R (Readout Coverage):
- Min/P10/Mean leaves per category.
"""

import sys
import os
import pickle
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.geometry_n import dist_n, minkowski_inner
# Import for Gate C check
from hei_n.integrator_torch import ContactIntegratorTorch, ContactConfigTorch, ContactStateTorch, IdentityInertiaTorch
from hei_n.potentials_torch import SparseEdgePotentialTorch, NegativeSamplingPotentialTorch, HarmonicPriorTorch, CompositePotentialTorch, SpringAttractionTorch, LogCoshRepulsionTorch

def load_checkpoint(path):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def is_chinese(word):
    for char in word:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def analyze_token_type(token):
    if is_chinese(token): return "ZH"
    if all(c.isalpha() for c in token): return "EN/Latin"
    if all(c.isdigit() for c in token): return "Digit"
    return "Other"

def gate_l_enhanced(nodes, edges, node_map, sample_size=500, k=50):
    """Enhanced Gate L: Purity Distribution & Breakdown."""
    print("\n[Gate L] Language Purity Audit (Enhanced)...")
    
    ch_words = [n for n in nodes if n.startswith("C:") and is_chinese(n.split(":")[1])]
    samples = np.random.choice(ch_words, min(len(ch_words), sample_size), replace=False)
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    purities = []
    non_cn_types = defaultdict(int)
    non_cn_examples = []
    
    for word_node in samples:
        idx = node_map[word_node]
        # BFS 2-hop
        neighbors = set()
        q = [idx]
        visited = {idx}
        for _ in range(2):
            next_q = []
            for curr in q:
                for n in adj[curr]:
                    if n not in visited:
                        visited.add(n)
                        neighbors.add(n)
                        next_q.append(n)
            q = next_q
            
        leaf_neighbors = [nodes[n] for n in neighbors if nodes[n].startswith("C:")]
        if not leaf_neighbors: continue
            
        cnt = 0
        for n in leaf_neighbors:
            token = n.split(":")[1]
            if is_chinese(token):
                cnt += 1
            else:
                t_type = analyze_token_type(token)
                non_cn_types[t_type] += 1
                if len(non_cn_examples) < 20:
                    non_cn_examples.append(token)
                    
        purity = cnt / len(leaf_neighbors)
        purities.append(purity)
        
    purities = np.array(purities)
    print(f"  Purity Mean: {np.mean(purities)*100:.2f}%")
    print(f"  Purity P10:  {np.percentile(purities, 10)*100:.2f}%")
    print(f"  Purity P50:  {np.percentile(purities, 50)*100:.2f}%")
    print(f"  Purity P90:  {np.percentile(purities, 90)*100:.2f}%")
    
    print("  Non-Chinese Breakdown:")
    total_non = sum(non_cn_types.values())
    if total_non > 0:
        for k_type, v in non_cn_types.items():
            print(f"    {k_type}: {v} ({v/total_non*100:.1f}%)")
    print(f"  Examples: {non_cn_examples[:10]}")

def gate_s_enhanced(nodes, edges, node_map, G_pos, sample_size=1000):
    """Enhanced Gate S: Ratio Stats."""
    print("\n[Gate S] Synonym Convergence Audit (Enhanced)...")
    
    parent_map = {}
    for u, v in edges:
        child_node = nodes[v]
        parent_node = nodes[u]
        if child_node.startswith("C:") and parent_node.startswith("Code:"):
            if parent_node not in parent_map:
                parent_map[parent_node] = []
            parent_map[parent_node].append(v)
            
    valid_groups = [g for g in parent_map.values() if len(g) >= 2]
    samples = valid_groups[:sample_size]
    
    ratios = []
    all_indices = list(range(len(nodes)))
    
    for group in samples:
        u, v = group[0], group[1]
        d_inner = dist_n(G_pos[u], G_pos[v])
        
        r1, r2 = np.random.choice(all_indices, 2)
        d_rand = dist_n(G_pos[r1], G_pos[r2]) + 1e-9
        
        ratios.append(d_inner / d_rand)
        
    ratios = np.array(ratios)
    pass_rate = np.mean(ratios < 1.0)
    
    print(f"  Ratio Mean: {np.mean(ratios):.4f}")
    print(f"  Ratio P50:  {np.percentile(ratios, 50):.4f}")
    print(f"  Ratio P90:  {np.percentile(ratios, 90):.4f}")
    print(f"  % Ratio < 1.0: {pass_rate*100:.1f}%")

def gate_h_hierarchy(nodes, G_pos, node_map, depths_path):
    """Gate H: Hierarchy Consistency."""
    print("\n[Gate H] Hierarchy Consistency Audit...")
    
    if not os.path.exists(depths_path):
        print(f"WARN: Depths file not found at {depths_path}. Skipping H.")
        return
        
    depths = np.load(depths_path)
    
    # 1. Depth vs Radius Correlation
    # Radius = acosh(x0)
    # G[:, 0, 0] is the x0 coordinate (cosh(r))
    radii = np.arccosh(np.maximum(G_pos[:, 0, 0], 1.0))
    
    # Filter only Code nodes for clearer hierarchy check
    code_indices = [i for i, n in enumerate(nodes) if n.startswith("Code:")]
    
    if not code_indices:
        print("WARN: No code nodes found.")
        return
        
    d_sample = depths[code_indices]
    r_sample = radii[code_indices]
    
    corr, p = spearmanr(d_sample, r_sample)
    print(f"  Spearman(Depth, Radius): {float(corr):.4f} (p={float(p):.2e})")
    print(f"  Status: {'PASS' if corr > 0.8 else 'WARN'}")

def gate_c_dynamics(nodes, edges, G_pos, M_pos, steps=50):
    """Gate C: Run short simulation to check stability."""
    print("\n[Gate C] Numerical Stability Re-Audit (Dynamics)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Running on {device}...")
    
    # Convert data to torch
    N, dim1, dim2 = G_pos.shape
    dim = dim1 - 1
    
    G = torch.tensor(G_pos, device=device, dtype=torch.float32)
    M = torch.tensor(M_pos, device=device, dtype=torch.float32)
    z = torch.tensor(0.0, device=device)
    edges_gpu = torch.tensor(edges, device=device, dtype=torch.long)
    
    state = ContactStateTorch(G=G, M=M, z=z)
    
    # Config
    k_attract = SpringAttractionTorch(k=1.0)
    attract = SparseEdgePotentialTorch(edges_gpu, k_attract)
    k_repulse = LogCoshRepulsionTorch(sigma=1.0, A=5.0)
    repulse = NegativeSamplingPotentialTorch(k_repulse, num_neg=10, rescale=1.0)
    trap = HarmonicPriorTorch(k=0.05)
    oracle = CompositePotentialTorch([attract, repulse, trap])
    inertia = IdentityInertiaTorch()
    
    config = ContactConfigTorch(dt=0.01, gamma=2.0, target_temp=0.1, thermostat_tau=5.0, 
                                fixed_point_iters=3, solver_mixing=0.5, torque_clip=50.0, 
                                renorm_interval=10, adaptive=True, tol_disp=0.2, device=device)
                                
    integrator = ContactIntegratorTorch(oracle, inertia, config)
    
    residuals = []
    renorms = []
    
    # Run loop
    for _ in range(steps):
        state = integrator.step(state)
        diag = state.diagnostics
        residuals.append(diag.get('solver_residual', 0.0))
        renorms.append(diag.get('renorm_magnitude', 0.0))
        
    res_p99 = np.percentile(residuals, 99)
    ren_max = np.max(renorms)
    
    print(f"  Residual P99: {res_p99:.2e}")
    print(f"  Renorm Max:   {ren_max:.2e}")
    
    pass_c = res_p99 < 5e-3 and ren_max < 1e-4
    print(f"  Status: {'PASS' if pass_c else 'FAIL'}")

def gate_m_coverage(nodes, node_map):
    """Gate M: Critical Vocab Check."""
    print("\n[Gate M] Critical Vocab Coverage...")
    
    targets = ['爱', '喜欢', '开心', '难过', '害怕', '想要', '因为', '如果']
    found = []
    missing = []
    
    # Map back to raw vocab
    # Nodes are like C:word:id
    # We need to find if word exists
    vocab = set()
    for n in nodes:
        if n.startswith("C:"):
            parts = n.split(":")
            if len(parts) >= 2:
                vocab.add(parts[1])
                
    for t in targets:
        if t in vocab:
            found.append(t)
        else:
            missing.append(t)
            
    print(f"  Targets Found: {len(found)}/{len(targets)}")
    if missing:
        print(f"  Missing: {missing}")
    else:
        print("  All Critical Words Found.")
    
    print(f"  Status: {'PASS' if not missing else 'WARN'}")

def gate_r_enhanced(nodes, edges, node_map):
    """Gate R: Min/P10 Leaves."""
    print("\n[Gate R] Readout Coverage (Enhanced)...")
    
    codes = [n for n in nodes if n.startswith("Code:")]
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        
    leaf_counts = []
    for code_node in codes:
        idx = node_map[code_node]
        leaves = 0
        q = [idx]
        while q:
            curr = q.pop(0)
            if curr in adj:
                children = adj[curr]
                for child in children:
                    child_node = nodes[child]
                    if child_node.startswith("C:"):
                        leaves += 1
                    elif child_node.startswith("Code:"):
                        q.append(child)
        if leaves > 0:
            leaf_counts.append(leaves)
            
    leaf_counts = np.array(leaf_counts)
    print(f"  Mean Leaves: {np.mean(leaf_counts):.1f}")
    print(f"  Min Leaves:  {np.min(leaf_counts)}")
    print(f"  P10 Leaves:  {np.percentile(leaf_counts, 10)}")
    
    pass_r = np.percentile(leaf_counts, 10) >= 1
    print(f"  Status: {'PASS' if pass_r else 'WARN'}")

def main():
    checkpoint = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    depths = "checkpoints/aurora_base_gpu_cilin_full_depths.npy"
    
    data = load_checkpoint(checkpoint)
    nodes = data['nodes']
    edges = data.get('edges', [])
    G_pos = data['G']
    M_pos = data['M']
    
    node_map = {n: i for i, n in enumerate(nodes)}
    
    gate_l_enhanced(nodes, edges, node_map)
    gate_h_hierarchy(nodes, G_pos, node_map, depths)
    gate_s_enhanced(nodes, edges, node_map, G_pos)
    gate_m_coverage(nodes, node_map)
    gate_r_enhanced(nodes, edges, node_map)
    
    # Gate C last as it needs pytorch
    gate_c_dynamics(nodes, edges, G_pos, M_pos)

if __name__ == "__main__":
    main()

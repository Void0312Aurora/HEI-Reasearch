
import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import defaultdict

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora import AuroraDataset, GaugeField
from aurora.gauge import GaugeField

def compute_alignment_stats(gauge_field, J, edge_indices, name="Edges"):
    """
    Compute alignment stats for a subset of edges.
    """
    if len(edge_indices) == 0:
        print(f"--- {name}: No edges found ---")
        return
        
    u = gauge_field.edges[edge_indices, 0]
    v = gauge_field.edges[edge_indices, 1]
    
    # gauge_field.get_U() with x
    # We need x to be available? compute_alignment_stats signature update?
    # Or assume gauge_field handles it if we pass x?
    # But this function doesn't take x.
    # Quick fix: Pass x to compute_alignment_stats
    pass

def compute_alignment_stats(gauge_field, J, edge_indices, x, name="Edges"):
    """
    Compute alignment stats for a subset of edges.
    """
    if len(edge_indices) == 0:
        print(f"--- {name}: No edges found ---")
        return
        
    u = gauge_field.edges[edge_indices, 0]
    v = gauge_field.edges[edge_indices, 1]
    
    # Transport matrices
    # For Neural, we need to query SPECIFIC edges defined by edge_indices?
    # Or just get all and slice?
    # Neural backend get_U(x, edges) returns U for those edges.
    # If we call get_U(x) with no edges, it computes for self.edges.
    
    # Optimization: If Neural, we can just compute for the batch!
    # But get_U(x) returns (E_total, ...). Slicing is fine.
    # NOTE: If E_total is huge, this is wasteful. 
    # But for 18k edges it's fast.
    
    U_all = gauge_field.get_U(x=x)
    U_sub = U_all[edge_indices]
    
    J_u = J[u]
    J_v = J[v]
    
    # Raw Alignment <J_v, J_u>
    raw_align = torch.sum(J_v * J_u, dim=-1)
    
    # Gauge Alignment <J_v, U J_u>
    J_u_trans = torch.matmul(U_sub, J_u.unsqueeze(-1)).squeeze(-1)
    gauge_align = torch.sum(J_v * J_u_trans, dim=-1)
    
    # Stats
    raw_mean = torch.mean(raw_align).item()
    gauge_mean = torch.mean(gauge_align).item()
    gain = gauge_mean - raw_mean
    
    print(f"--- {name} (N={len(edge_indices)}) ---")
    print(f"  Raw Correlation:    {raw_mean:.4f}")
    print(f"  Gauge Alignment:    {gauge_mean:.4f}")
    print(f"  Gauge Gain:         {gain:+.4f}")
    
    return raw_mean, gauge_mean, gain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--semantic_path", type=str, required=True, help="Path for Training Semantic Edges")
    parser.add_argument("--holdout_path", type=str, default=None, help="Path for Holdout Semantic Edges (Optional override)")
    parser.add_argument("--dataset", type=str, default="cilin")
    parser.add_argument("--limit", type=int, default=None, help="Dataset limit (must match training)")
    parser.add_argument("--gauge_mode", type=str, default=None, help="Force backend type (table/neural)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.checkpoint} on {device}...")
    
    # 1. Load Checkpoint
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
        
    J = torch.tensor(ckpt['J'], device=device)
    x = torch.tensor(ckpt['x'], device=device)
    print(f"Loaded J: {J.shape}, x: {x.shape}")
    
    # 2. Load Dataset (Structural)
    # Checkpoint limit might be missing, use CLI arg if provided
    limit = args.limit if args.limit is not None else ckpt['config'].get('limit', None)
    print(f"Loading Dataset with limit={limit}...")
    ds = AuroraDataset(args.dataset, limit=limit)
    edges_struct = torch.tensor(ds.edges_struct, dtype=torch.long, device=device)
    num_struct = edges_struct.shape[0]
    print(f"Loaded {num_struct} Structural Edges.")
    
    # 3. Load Semantic (Split)
    # Using dataset method to ensure consistent mapping and splitting with training
    print("Loading Semantic Edges via AuroraDataset...")
    print("Loading Semantic Edges via AuroraDataset...")
    sem_train_list = ds.load_semantic_edges(args.semantic_path, split="train")
    
    holdout_file = args.holdout_path if args.holdout_path else args.semantic_path
    
    # If using dense file, we want strict holdout
    sem_test_list = ds.load_semantic_edges(holdout_file, split="holdout")
    
    # If overriding holdout path, we treat all edges in it as test edges (since they are new)
    if args.holdout_path:
        # If we use a separate file, assume it's ALL holdout
        sem_test_list = ds.load_semantic_edges(holdout_file, split="train") # "train" just means "load all" here from the list
        # Actually ds.load_semantic_edges splits by index.
        # If we want ALL edges from the dense file:
        with open(holdout_file, 'rb') as f:
             raw_edges = pickle.load(f)
        # Convert to list
        sem_test_list = []
        w2i = ds.vocab.word_to_id
        for u, v, w in raw_edges:
             if u in w2i and v in w2i:
                 sem_test_list.append((w2i[u], w2i[v], w))
             
    # Convert to tensors
    
    # Convert to tensors
    sem_train = torch.tensor([(u, v) for u, v, w in sem_train_list], dtype=torch.long, device=device)
    sem_test = torch.tensor([(u, v) for u, v, w in sem_test_list], dtype=torch.long, device=device)
    
    print(f"Semantic Split: {sem_train.shape[0]} Train, {sem_test.shape[0]} Test")
    
    # Validation
    N_j = J.shape[0]
    print(f"Validating Indices against J (N={N_j})...")
    
    if edges_struct.max() >= N_j:
        print(f"ERROR: Structural Edges contain index {edges_struct.max()} >= N!")
        return
    if sem_train.max() >= N_j:
        print(f"ERROR: Semantic Train Edges contain index {sem_train.max()} >= N!")
        return
    if sem_test.max() >= N_j:
        print(f"ERROR: Semantic Test Edges contain index {sem_test.max()} >= N!")
        return
        
    print("Indices Valid.")

    # Placeholder for Curvature Analysis (if it were present before)
    # For now, we'll just add the holdout curvature section.
    # If there was a previous curvature analysis, its output would be here.
    # Example of what might have been here:
    # if tri_h.shape[0] > 0:
    #     norms_h = torch.norm(Omega_h.reshape(Omega_h.shape[0], -1), dim=1)
    #     p99 = torch.quantile(norms_h, 0.99).item()
    #     max_val = torch.max(norms_h).item()
    #     print(f"  P99:  {p99:.4f}")
    #     print(f"  Max:  {max_val:.4f}")
    # else:
    #     print("No Triangles found (check topology).")
            

    
    # 4. Reconstruct Gauge Field
    # Gauge Field must match Training Topology exactly to load weights.
    # Training looked at: Struct + Sem_Train
    edges_train_all = torch.cat([edges_struct, sem_train], dim=0)
    
    # Determine backend
    backend = args.gauge_mode
    if backend is None:
         backend = ckpt.get('config', {}).get('gauge_mode', 'table')
         
    print(f"Initializing GaugeField with backend={backend}...")
    gauge_field = GaugeField(edges_train_all, logical_dim=3, group='SO',
                             backend_type=backend,
                             input_dim=5).to(device)
    
    # Load Weights
    try:
        gauge_field.load_state_dict(ckpt['gauge_field'])
        print("Gauge Field weights loaded successfully.")
    except Exception as e:
        print(f"Error loading Gauge Field: {e}")
        return

    # 5. Evaluate Subsets
    # Indices in gauge_field.edges:
    # 0 .. num_struct-1 : Structural
    # num_struct .. num_struct+num_train-1 : Semantic Train
    
    # 5. Evaluate Subsets
    # Indices in gauge_field.edges might differ due to deduplication.
    # We must lookup using edge_map.
    
    def get_indices(edge_list, edge_map):
        indices = []
        for u, v in edge_list.cpu().numpy():
            if (u, v) in edge_map:
                idx, _ = edge_map[(u, v)]
                indices.append(idx)
            elif (v, u) in edge_map:
                idx, _ = edge_map[(v, u)]
                indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=device)
        
    indices_struct = get_indices(edges_struct, gauge_field.edge_map)
    indices_sem_train = get_indices(sem_train, gauge_field.edge_map)
    
    print("\n=== Alignment Analysis per Edge Type ===")
    compute_alignment_stats(gauge_field, J, indices_struct, x, "Structural Edges (Tree)")
    compute_alignment_stats(gauge_field, J, indices_sem_train, x, "Semantic Edges (Train)")
    
    # 6. Out-of-Sample Evaluation (Test Set)
    # These edges are NOT in gauge_field.edges.
    # We must treat them as new connections.
    # How to get U_uv for new edges?
    # U_uv = P exp(int A). 
    # But A is discrete! We only have U defined on specific edges.
    # For a NEW semantic edge (u, v), the connection U_uv is NOT defined freely.
    # Ideally, U_uv should be the transport along a path in the computed manifold?
    # OR, if we assume the Gauge Field defines a geometry, then U_uv *should* be predictable?
    # Wait. In Lattice Gauge Theory, U is only defined on links. 
    # If (u,v) is not a link, we can defining U_uv via the shortest path on the valid graph?
    # OR, we simply check `Raw Correlation`.
    # If the geometry has shaped J, then J_u and J_v should be aligned naturally (Raw Correlation).
    # If Raw Correlation on Test Set is high, it means J's embedding captured the structure.
    # If Raw Correlation is low... can we compute 'Induced' U?
    # For holdout edges, we expect J to be aligned *without* explicit U (or U approx Identity?).
    # NO. The whole point of Gauge Theory is that J are NOT aligned in base space.
    # They are aligned via U.
    # If a new semantic relation exists between u and v, and we haven't learned U_uv,
    # then we can't measure Gauge Alignment!
    # UNLESS: The "correct" U_uv is induced by path dependence (Flatness hypothesis?).
    # If the manifold is consistent, U_uv ~ U_{path}(u->v).
    # So we should find a path u->...->v in the structural graph, compute transport U_path, and check alignment.
    # BUT, that's expensive.
    
    # Alternative: The "Prediction" is that J_u and J_v are related by specific rotation.
    # But we don't know that rotation without U.
    # Wait. The "Gauge Gain" logic implies J relies on U.
    # If we remove U, J is disordered.
    # So for TEST edges, if we don't have U, we see disorder.
    # This implies we cannot "predict" J correlation without inferring the connection.
    # But inference OF connection requires minimizing energy with J fixed?
    # i.e. Given J_u, J_v, what is the optimal U?
    # No, that's trivial (U = J_v J_u^T).
    
    # Re-reading "Out-of-Sample Gain":
    # "If V6 on holdout still high gain..."
    # Reviewer implies we CAN compute gain.
    # But Gain = GaugeAlign - Raw.
    # GaugeAlign requires U.
    # Maybe they assume we define U for holdout edges? How?
    # 1. Parameter Memory: If U is just a hashtable param, we CANNOT defined it for unseen edges.
    # 2. Generalization: U should be a function of position x? U(x, v).
    #    Our `compute_connection` (Wong force) uses `omega_params` to compute A(v).
    #    But `omega_params` IS the discrete field on edges.
    #    It projects to A(v) using `log_map`.
    #    So we CAN compute A_eff between arbitrary u and v!
    #    Using the `compute_connection` function!
    #    YES! `compute_connection` interpolates A from neighbors using continuous embedding x.
    #    This is the "Continuous Extension" of the Gauge Field.
    #    So for a test edge (u, v), we calculate v_u = Log_u(v), compute A(u, v_u),
    #    integrate to get U, and check alignment.
    
    print("\n=== Out-of-Sample Generalization (Test Set) ===")
    
    # We need to use `compute_connection` logic manually for test edges.
    # Or add temporary edges to a dummy GaugeField sharing weights?
    # Better: Use `compute_connection_for_edge` helper?
    # The existing `compute_connection` takes (x, v).
    # We can generate x, v for all test edges.
    
    u_test = sem_test[:, 0]
    v_test = sem_test[:, 1]
    
    J_u = J[u_test]
    J_v = J[v_test]
    
    # Raw Alignment
    raw_align = torch.sum(J_v * J_u, dim=-1)
    raw_mean = torch.mean(raw_align).item()
    
    X = torch.tensor(ckpt['x'], device=device) # (N, dim)
    
    x_u = X[u_test]
    x_v = X[v_test]
    
    # 1. Compute vector v for the edge (tangent at u pointing to v)
    # v_vec = Log_u(v)
    from aurora.geometry import log_map
    v_vec = log_map(x_u, x_v) # (E_test, dim)
    
    # 1. Compute vector v for the edge (tangent at u pointing to v)
    # v_vec = Log_u(v)
    from aurora.geometry import log_map
    v_vec = log_map(x_u, x_v) # (E_test, dim)
    
    # 2. Continuous Gauge Potential (Manual Implementation)
    # We cannot use gauge_field.compute_connection because it expects full fields.
    # We use the precomputed A_nodes below.
    
    print("Computing Continuous Gauge Potential for Generalization...")
    
    if gauge_field.backend_type == 'neural':
         print(">>> Using Neural Backend for Holdout Prediction")
         # Direct Prediction
         # gauge_field.get_U(x, edges)
         U_pred = gauge_field.get_U(x=X, edges=sem_test)
    else:
         print(">>> Using Manual Interpolation (Table Backend)")
         
         # Precompute Vector-Valued Connection A_mu at every node
         # A_mu = sum_{neighbor e} Omega_e * (dir_e / |dir_e|^2)
         # Result shape: (N, dim, k, k)
         
         N = x.shape[0]
         dim = x.shape[1]
         k = gauge_field.logical_dim
        
         # 1. Gather Edge Geometric Data (Training Edges)
         start = gauge_field.edges[:, 0]
         end = gauge_field.edges[:, 1]
        
         print(f"Gauge Edge Stats: Count={gauge_field.edges.shape[0]}, Min={gauge_field.edges.min()}, Max={gauge_field.edges.max()}")
         if gauge_field.edges.max() >= N:
             print(f"CRITICAL ERROR: Gauge Edge Index {gauge_field.edges.max()} >= N ({N})!")
             return
            
         x_start = x[start]
         x_end = x[end]
        
         dir_start = log_map(x_start, x_end) # (E_train, dim)
         dist_sq = torch.sum(dir_start ** 2, dim=-1, keepdim=True).clamp(min=1e-6)
         normalized_dir = dir_start / dist_sq # (E_train, dim)
        
         # 2. Gather Edge Omega
         # Enforce skew symmetry
         # TableBackend specific
         omega = 0.5 * (gauge_field.backend.omega_params - gauge_field.backend.omega_params.transpose(1, 2)) 
        
         # 3. Outer Product: Omega * dir
         A_contrib = omega.unsqueeze(1) * normalized_dir.unsqueeze(-1).unsqueeze(-1)
        
         # 4. Scatter Add to Nodes
         A_nodes = torch.zeros(N, dim, k, k, device=device)
         A_nodes.index_add_(0, start, A_contrib)
        
         # Incoming
         dir_end = log_map(x_end, x_start)
         dist_sq_end = torch.sum(dir_end ** 2, dim=-1, keepdim=True).clamp(min=1e-6)
         normalized_dir_end = dir_end / dist_sq_end
         A_contrib_end = (-omega).unsqueeze(1) * normalized_dir_end.unsqueeze(-1).unsqueeze(-1)
         A_nodes.index_add_(0, end, A_contrib_end)
        
         print("Continuous Gauge Field A_mu reconstructed at all nodes.")
         
         # Evaluate on Test Edges
         u_idx = sem_test[:, 0]
         v_idx = sem_test[:, 1]
         x_u = x[u_idx]
         x_v = x[v_idx]
         v_vec = log_map(x_u, x_v) 
         A_u = A_nodes[u_idx]
         A_val = torch.sum(v_vec.unsqueeze(-1).unsqueeze(-1) * A_u, dim=1)
         U_pred = torch.matrix_exp(A_val)
    
    # Transport
    J_u_trans = torch.matmul(U_pred, J_u.unsqueeze(-1)).squeeze(-1)
    gauge_align = torch.sum(J_v * J_u_trans, dim=-1)
    gauge_mean = torch.mean(gauge_align).item()
    
    gain = gauge_mean - raw_mean
    
    print(f"--- Holdout Generalization (N={sem_test.shape[0]}) ---")
    print(f"  Raw Correlation:    {raw_mean:.4f}")
    print(f"  Gauge Alignment:    {gauge_mean:.4f}")
    print(f"  Gauge Gain:         {gain:+.4f}")

    # Holdout Curvature Analysis
    # We construct a temporary GaugeField with Struct + Holdout to see induced curvature
    print("\n=== Holdout Subgraph Curvature Analysis ===")
    edges_holdout_all = torch.cat([edges_struct, sem_test], dim=0)
    gauge_holdout = GaugeField(edges_holdout_all, logical_dim=3, group='SO',
                               backend_type=backend, input_dim=5).to(device)
    # Share weights
    # Share weights (manually to avoid buffer mismatch)
    # Filter state dict to only backend parameters
    source_state = gauge_field.state_dict()
    target_state = {}
    for k, v in source_state.items():
        if k.startswith("backend."):
            target_state[k] = v
            
    gauge_holdout.load_state_dict(target_state, strict=False) 
    
    if backend == 'neural':
        gauge_holdout.eval()
        with torch.no_grad():
            Omega_h, tri_h, _ = gauge_holdout.compute_curvature(x=x)
            print(f"Holdout Triangles: {tri_h.shape[0]}")
            if tri_h.shape[0] > 0:
                norms_h = torch.norm(Omega_h.reshape(Omega_h.shape[0], -1), dim=1)
                print(f"  Mean: {torch.mean(norms_h).item():.4f}")
                print(f"  P50:  {torch.quantile(norms_h, 0.5).item():.4f}")
                print(f"  P90:  {torch.quantile(norms_h, 0.9).item():.4f}")
    else:
        print("Skipping Holdout Curvature (Table Backend cannot predict new edges).")

if __name__ == "__main__":
    main()

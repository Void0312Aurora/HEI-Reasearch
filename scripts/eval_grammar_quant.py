"""
Aurora Grammar Quantification.
==============================

Quantifies Typed Edge Prediction.
1. Train Typed Gauge Field on SVO Data (Train Split).
2. Measure Accuracy on Test Split.

Task: Given (Subject, Relation), predict Object/Predicate.
"""

import sys
import os
import torch
import numpy as np
import pickle
import argparse
import networkx as nx
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField, NeuralBackend
from aurora.ingest.dependency import DependencyGeometer
from aurora.training.streaming import StreamingTrainer

def generate_svo_dataset(ds, n_samples=2000):
    """
    Generate synthetic SVO dataset from Cilin nodes.
    Returns list of sentences.
    """
    # ds.nodes is a list of strings
    nodes = ds.nodes
    
    # Define Pools
    subjects = nodes[:500]
    verbs = nodes[500:1000]
    objects = nodes[1000:1500]
    
    # Create Structure: S[i] -> V[i] -> O[i] roughly
    # We split into 5 clusters
    n_clusters = 5
    cluster_size = 100
    
    data = []
    for _ in range(n_samples):
        c = np.random.randint(0, n_clusters)
        s_pool = subjects[c*cluster_size : (c+1)*cluster_size]
        v_pool = verbs[c*cluster_size : (c+1)*cluster_size]
        o_pool = objects[c*cluster_size : (c+1)*cluster_size]
        
        if not s_pool or not v_pool or not o_pool: continue
        
        s = np.random.choice(s_pool)
        v = np.random.choice(v_pool)
        o = np.random.choice(o_pool)
        data.append(f"{s} {v} {o}")
    return data

def evaluate_prediction(gauge_field, x, J, test_edges, ds, device):
    """
    Measure Top-k Accuracy for (u, r) -> v.
    """
    gauge_field.eval()
    
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    # Pre-compute J for all nodes
    all_J = J.to(device) # (N, k)
    
    print(f"Evaluating on {len(test_edges)} edges...")
    
    for u_id, v_id, r_id in test_edges:
        if r_id == 0: continue # Skip Neutral for typicality
        
        # 1. Transport u -> target
        u_t = torch.tensor([u_id], dtype=torch.long, device=device)
        r_t = torch.tensor([r_id], dtype=torch.long, device=device)
        
        # We need a dummy v to compute U(u, dummy, r) ?
        # Wait, NeuralBackend uses (u, v) to compute U.
        # It's U_{uv}. It DEPENDS on v.
        # This means we cannot "Predict v" by transporting u into empty space.
        # We compute Energy(u, cand, r) for all candidates.
        
        # Candidate pool: All 1500 vocab words? Or subset?
        # Efficient eval: Sample 100 negatives + 1 positive.
        
        pos_v = v_id
        neg_vs = np.random.randint(0, 1500, 100) # Assuming vocab ~1500 used in generation
        candidates = np.concatenate([[pos_v], neg_vs])
        candidates = list(set(candidates)) # Unique
        if pos_v not in candidates: candidates.append(pos_v)
        
        cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
        u_expanded = u_t.expand(len(candidates))
        r_expanded = r_t.expand(len(candidates))
        
        edges_t = torch.stack([u_expanded, cand_t], dim=1)
        
        U = gauge_field.get_U(x=x, edges=edges_t, relation_ids=r_expanded) # (C, k, k)
        
        J_u = all_J[u_expanded].unsqueeze(-1) # (C, k, 1)
        J_trans = torch.matmul(U, J_u).squeeze(-1) # (C, k)
        
        J_cand = all_J[cand_t]
        
        scores = torch.sum(J_cand * J_trans, dim=-1) # (C,)
        
        # Rank
        sorted_indices = torch.argsort(scores, descending=True)
        top_cands = cand_t[sorted_indices]
        
        if top_cands[0].item() == pos_v:
            correct_1 += 1
        
        # Top 5
        if pos_v in top_cands[:5].tolist():
            correct_5 += 1
            
        total += 1
        
    acc_1 = correct_1 / total
    acc_5 = correct_5 / total
    return acc_1, acc_5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    
    # 2. Setup Typed Gauge Field
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5)
    gauge_field.backend = NeuralBackend(input_dim=5, logical_dim=3, num_relations=4, relation_dim=8) # 4 rels
    gauge_field.to(device)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    
    # 3. Data Generation
    print("Generating SVO Dataset...")
    sentences = generate_svo_dataset(ds, n_samples=1000)
    
    print(f"Generated {len(sentences)} sentences. Sample: {sentences[0]}")
    
    geometer = DependencyGeometer()
    all_edges = []
    for txt in sentences:
        edges = geometer.text_to_edges(txt)
        # Convert to IDs
        valid = []
        for u_s, v_s, r in edges:
            if u_s in ds.vocab.word_to_id and v_s in ds.vocab.word_to_id:
                valid.append((ds.vocab.word_to_id[u_s], ds.vocab.word_to_id[v_s], r))
            else:
                pass # print(f"Invalid: {u_s} or {v_s}")
        all_edges.extend(valid)
        
    if not all_edges:
        print("Error: No valid edges found. Checking vocab...")
        print(f"Vocab size: {len(ds.vocab.word_to_id)}")
        print(f"Sample vocab key: {list(ds.vocab.word_to_id.keys())[0]}")
        return
        
    # Split
    split_idx = int(0.8 * len(all_edges))
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]
    
    print(f"Dataset: {len(train_edges)} Train, {len(test_edges)} Test.")
    
    # 4. Train
    print("Training Typed Gauge Field...")
    optimizer = torch.optim.Adam(gauge_field.parameters(), lr=0.01)
    
    for epoch in range(5):
        gauge_field.train()
        total_loss = 0
        
        # Batching
        batch_size = 32
        for i in range(0, len(train_edges), batch_size):
            batch = train_edges[i:i+batch_size]
            u_b = torch.tensor([e[0] for e in batch], device=device)
            v_b = torch.tensor([e[1] for e in batch], device=device)
            r_b = torch.tensor([e[2] for e in batch], device=device)
            
            optimizer.zero_grad()
            
            edges_t = torch.stack([u_b, v_b], dim=1)
            U = gauge_field.get_U(x=x, edges=edges_t, relation_ids=r_b)
            
            J_u = J[u_b].unsqueeze(-1)
            J_trans = torch.matmul(U, J_u).squeeze(-1)
            J_v = J[v_b]
            
            align = torch.sum(J_v * J_trans, dim=-1)
            loss = torch.mean(1.0 - align)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss / (len(train_edges)/batch_size):.4f}")
        
    # 5. Evaluate
    print("Evaluating...")
    acc1, acc5 = evaluate_prediction(gauge_field, x, J, test_edges, ds, device)
    print(f"Test Accuracy: Top-1 {acc1:.4f}, Top-5 {acc5:.4f}")
    
    if acc1 > 0.1: # Chance is 1/100 = 0.01
        print("Success: Typed Prediction significantly better than chance.")

if __name__ == "__main__":
    main()

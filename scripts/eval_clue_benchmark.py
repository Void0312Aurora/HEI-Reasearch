"""
Aurora CLUE Benchmark Evaluation (Phase 25).
============================================

Objective: Reproduce Performance Curves (0-shot -> k-shot).
Tasks:
1. OCNLI (Inference): 3-class (Entailment, Neutral, Contradiction).
2. AFQMC (Similarity): 2-class (Similar, Dissimilar).

Method:
- Extract Geometric Features (Cosine, Norms).
- 0-Shot: Heuristic Thresholds.
- k-Shot: Calibrate Linear Layer on Features.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField, NeuralBackend
from aurora.eval.clue_probe import GeometricProbe

# --- Mock Data Generators (Replace with Real Loaders) ---
def generate_mock_afqmc(ds, n_samples=1000):
    """
    Generate Positive (Similar) and Negative (Dissimilar) pairs.
    """
    nodes = ds.nodes
    data = []
    # Positive: Same Node or Synonym (if we knew synonyms). Same Node for now.
    # Negative: Random pair.
    for _ in range(n_samples // 2):
        u = np.random.choice(nodes)
        data.append((u, u, 1)) # Pos
        
    for _ in range(n_samples // 2):
        u = np.random.choice(nodes)
        v = np.random.choice(nodes)
        if u != v:
             data.append((u, v, 0)) # Neg
             
    np.random.shuffle(data)
    return data

def generate_mock_ocnli(ds, n_samples=1000):
    """
    Generate Entailment(0), Neutral(1), Contradiction(2).
    """
    nodes = ds.nodes
    data = []
    for _ in range(n_samples):
        # Random for now, just to test pipeline
        u = np.random.choice(nodes)
        v = np.random.choice(nodes)
        lbl = np.random.randint(0, 3)
        data.append((u, v, lbl))
    return data

# --- Calibration Model ---
class LinearCalibrator(nn.Module):
    def __init__(self, input_dim=3, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def run_calibration(train_feats, train_lbls, test_feats, test_lbls, num_classes, epochs=100):
    model = LinearCalibrator(input_dim=train_feats.shape[1], num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_train = torch.tensor(train_feats, dtype=torch.float32)
    y_train = torch.tensor(train_lbls, dtype=torch.long)
    X_test = torch.tensor(test_feats, dtype=torch.float32)
    y_test = torch.tensor(test_lbls, dtype=torch.long)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    with torch.no_grad():
        out_test = model(X_test)
        preds = torch.argmax(out_test, dim=1)
        acc = (preds == y_test).float().mean().item()
        
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Checkpoint...")
    # Custom Unpickler
    import io
    class DeviceUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=device)
            return super().find_class(module, name)
            
    with open(args.checkpoint, 'rb') as f:
        ckpt = DeviceUnpickler(f).load()
    x = torch.tensor(ckpt['x'], device=device)
    J = torch.tensor(ckpt['J'], device=device)
    
    # Init Gauge (200) - Note: Must match Demo Phase 24 config if we want consistency
    # But checkpoint `cycle_3` has Legacy Backend?
    # Or did we save the Phase 24 model?
    # The command line arg points to `checkpoint_cycle_3.pkl`.
    # That uses Legacy Backend.
    # I should reuse the Legacy Patch logic from previous scripts.
    
    from aurora.gauge import GaugeConnectionBackend
    from aurora.geometry import log_map
    
    class LegacyNeuralBackend(GaugeConnectionBackend):
        def __init__(self, input_dim=5, logical_dim=3, hidden_dim=64):
            super().__init__(logical_dim)
            input_size = 3 * input_dim
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, logical_dim * logical_dim)
            )
            self.rel_embed = None
        def get_omega(self, edges=None, x=None, relation_ids=None): return self.forward(x=x, edges_uv=edges)
        def forward(self, edge_indices=None, x=None, edges_uv=None, relation_ids=None):
            if x is None: raise ValueError("Need x")
            u, v = edges_uv[:, 0], edges_uv[:, 1]
            swap_mask = u > v
            u_canon = torch.where(swap_mask, v, u)
            v_canon = torch.where(swap_mask, u, v)
            xu = x[u_canon]; xv = x[v_canon]
            feat = torch.cat([xu, xv, log_map(xu, xv)], dim=-1)
            out = self.net(feat)
            out = 3.0 * torch.tanh(out)
            out = out.view(-1, self.logical_dim, self.logical_dim)
            omega = 0.5 * (out - out.transpose(1, 2))
            return omega * torch.where(swap_mask, -1.0, 1.0).view(-1, 1, 1)

    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5).to(device)
    gauge_field.backend = LegacyNeuralBackend().to(device)
    filtered = {k:v for k,v in ckpt['gauge_field'].items() if not k.startswith(('edges', 'tri'))}
    gauge_field.load_state_dict(filtered, strict=False)
    
    ds = AuroraDataset("cilin", limit=args.limit)
    probe = GeometricProbe(gauge_field, x, J, ds, device=device)
    
    # --- Benchmarking ---
    tasks = [('AFQMC', generate_mock_afqmc, 2), ('OCNLI', generate_mock_ocnli, 3)]
    shots = [0, 10, 50, 100, 200]
    
    for task_name, generator, num_classes in tasks:
        print(f"\n--- Benchmark: {task_name} ---")
        data = generator(ds, n_samples=1000)
        
        # Extract Features
        feats = []
        lbls = []
        for u, v, lbl in data:
            # We assume text is just the node string for Mock
            f = probe.extract_features(u, v)
            feats.append(f)
            lbls.append(lbl)
            
        feats = np.array(feats)
        lbls = np.array(lbls)
        
        # Accuracies
        results = []
        
        # Split Train/Test (Hold out last 200 for Test)
        test_size = 200
        X_test = feats[-test_size:]
        y_test = lbls[-test_size:]
        X_pool = feats[:-test_size]
        y_pool = lbls[:-test_size]
        
        for k in shots:
            if k == 0:
                # 0-Shot Heuristic
                if task_name == 'AFQMC':
                    # Cosine > 0.5
                    sims = X_test[:, 0]
                    preds = (sims > 0.5).astype(int)
                    acc = np.mean(preds == y_test)
                else:
                    # OCNLI: >0.6=0, <0.3=2, else=1
                    sims = X_test[:, 0]
                    preds = []
                    for s in sims:
                        if s > 0.6: preds.append(0)
                        elif s < 0.3: preds.append(2)
                        else: preds.append(1)
                    acc = np.mean(np.array(preds) == y_test)
            else:
                # k-Shot Calibration
                if k > len(X_pool): k = len(X_pool)
                X_train = X_pool[:k]
                y_train = y_pool[:k]
                acc = run_calibration(X_train, y_train, X_test, y_test, num_classes)
                
            print(f"{k}-Shot Accuracy: {acc:.4f}")
            results.append(acc)

if __name__ == "__main__":
    main()

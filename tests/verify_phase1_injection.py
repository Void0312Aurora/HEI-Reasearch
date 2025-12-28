
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.model.injector import ContinuousInjector
from aurora.engine.readout import ReadoutMechanism
from aurora.data.data_pipeline import GlobalEntropyStats

def test_phase1():
    print("=== Phase 1: Single-Point Learnability Test ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Setup minimal data
    # Use a small fixed vocabulary to guarantee overfitting is possible
    vocab_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    char_to_id = {c: i for i, c in enumerate(vocab_chars)}
    id_to_char = {i: c for i, c in enumerate(vocab_chars)}
    
    # Mock Entropy Stats
    class MockEntropy:
        def get_radial_target(self, char):
            return 0.8 # Fixed radius for simplicity
            
    entropy_stats = MockEntropy()
    
    # 2. Setup Model
    DIM = 16
    injector = ContinuousInjector(input_dim=1, hidden_dim=64, dim=DIM).to(device)
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    
    optimizer = optim.Adam(injector.parameters(), lr=0.01)
    
    # 3. Training Loop
    # Goal: Overfit these 10 characters perfectly
    
    targets = torch.arange(len(vocab_chars), device=device)
    x_list = [ord(c)/65535.0 for c in vocab_chars]
    x_t = torch.tensor(x_list, dtype=torch.float, device=device).unsqueeze(1)
    r_t = torch.tensor([0.8]*len(vocab_chars), dtype=torch.float, device=device).unsqueeze(1)
    
    print("Starting optimization loop...")
    for step in range(500):
        optimizer.zero_grad()
        
        # Inject
        m, q, J, p = injector(x_t, r_t)
        
        # Update Readout Prototypes (Crucial Fix Check)
        # Verify that prototypes move matching the injector
        if step % 10 == 0:
            readout.update_prototypes()
            
        # Readout
        probs = readout.read_prob(q, beta=10.0)
        
        # Loss (NLL)
        # Correct logic: NLL on log_softmax
        loss = F.nll_loss(torch.log(probs + 1e-9), targets)
        
        loss.backward()
        optimizer.step()
        
        # Accuracy
        preds = probs.argmax(dim=1)
        acc = (preds == targets).float().mean()
        
        if step % 50 == 0:
            print(f"Step {step}: Loss {loss.item():.4f} | Acc {acc.item():.2f}")
            
        if acc.item() == 1.0 and loss.item() < 0.01:
            print(f"Converged at Step {step}!")
            return True
            
    print("Failed to converge.")
    return False

if __name__ == "__main__":
    test_phase1()

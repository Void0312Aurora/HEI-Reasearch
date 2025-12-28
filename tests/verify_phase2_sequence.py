
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.model.injector import ContinuousInjector
from aurora.engine.readout import ReadoutMechanism
from aurora.physics import geometry

def test_phase2():
    print("=== Phase 2: Minimal Sequence Learning Test ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Data: Cyclic Sequence A->B->C...->J->A
    vocab_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    char_to_id = {c: i for i, c in enumerate(vocab_chars)}
    id_to_char = {i: c for i, c in enumerate(vocab_chars)}
    
    # Mock Entropy
    class MockEntropy:
        def get_radial_target(self, char): return 0.8
    entropy_stats = MockEntropy()
    
    # 2. Model
    DIM = 16
    injector = ContinuousInjector(input_dim=1, hidden_dim=64, dim=DIM).to(device)
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    optimizer = optim.Adam(injector.parameters(), lr=0.005)
    
    # 3. Training Loop
    # We want to learn the SEQUENCE.
    # Batch = Pairs (t, t+1)
    
    # Prepare Inputs
    x_list = [ord(c)/65535.0 for c in vocab_chars]
    x_all = torch.tensor(x_list, dtype=torch.float, device=device).unsqueeze(1)
    r_all = torch.tensor([0.8]*len(vocab_chars), dtype=torch.float, device=device).unsqueeze(1)
    ids_all = torch.arange(len(vocab_chars), device=device)
    
    # x_curr: 0..8, x_next: 1..9 (A->B, B->C...)
    x_curr = x_all[:-1]
    r_curr = r_all[:-1]
    ids_curr = ids_all[:-1] # For contrastive on current
    
    x_next = x_all[1:]
    r_next = r_all[1:]
    ids_next = ids_all[1:] # For contrastive on next (validation)
    
    print("Starting Sequence Loop...")
    dt = 0.1
    
    for step in range(1000):
        optimizer.zero_grad()
        
        # 1. Inject (Current)
        m_c, q_c, J_c, p_c = injector(x_curr, r_curr)
        # 2. Inject (Next Target)
        m_n, q_n, J_n, p_n = injector(x_next, r_next)
        
        # 3. Update Readout
        if step % 10 == 0: readout.update_prototypes()
        
        # 4. Contrastive Loss (Separation)
        # Verify q_c matches ids_curr
        probs = readout.read_prob(q_c, beta=10.0)
        loss_contrast = F.nll_loss(torch.log(probs + 1e-9), ids_curr)
        
        # 5. Flow Loss (Connection)
        # q_pred = Exp(q_c, p_c * dt)
        q_pred = geometry.exp_map(q_c, p_c * dt)
        
        # Target for Flow is q_n
        loss_flow = torch.mean((q_pred - q_n)**2)
        
        # Combined Loss
        loss = loss_contrast + 5.0 * loss_flow
        
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            # Check accuracy logic
            preds = probs.argmax(dim=1)
            acc = (preds == ids_curr).float().mean()
            print(f"Step {step}: Loss {loss.item():.4f} (C {loss_contrast.item():.2f} F {loss_flow.item():.4f}) | Acc {acc.item():.2f}")
            
        if loss_contrast.item() < 0.1 and loss_flow.item() < 0.001:
            print(f"Phase 2 Converged at Step {step}!")
            return True
            
    print("Phase 2 Failed.")
    return False

if __name__ == "__main__":
    test_phase2()

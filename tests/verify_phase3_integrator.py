
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.model.injector import ContinuousInjector
from aurora.engine.readout import ReadoutMechanism
from aurora.engine.integrator import LieIntegrator
from aurora.engine.forces import ForceField
from aurora.physics import geometry

def test_phase3():
    print("=== Phase 3: Integrator Consistency Test ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup
    vocab_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    id_to_char = {i: c for i, c in enumerate(vocab_chars)}
    
    class MockEntropy:
        def get_radial_target(self, char): return 0.8
    entropy_stats = MockEntropy()
    
    DIM = 16
    injector = ContinuousInjector(input_dim=1, hidden_dim=64, dim=DIM).to(device)
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    # Initialize Field with small constants to match Phase 2's zero-force assumption approx
    ff = ForceField(dim=DIM, G=0.001, lambda_gauge=0.001, k_geo=0.001, mu=1.0).to(device)
    integrator = LieIntegrator()
    
    optimizer = optim.Adam(list(injector.parameters()) + list(ff.parameters()), lr=0.005)
    
    # Dataset
    x_list = [ord(c)/65535.0 for c in vocab_chars]
    x_all = torch.tensor(x_list, dtype=torch.float, device=device).unsqueeze(1)
    r_all = torch.tensor([0.8]*len(vocab_chars), dtype=torch.float, device=device).unsqueeze(1)
    ids_all = torch.arange(len(vocab_chars), device=device)
    
    dt = 0.1
    print("Starting Consistency Loop...")
    
    for step in range(1000):
        optimizer.zero_grad()
        
        # Inject
        # Current: 0..8
        m_c, q_c, J_c, p_c = injector(x_all[:-1], r_all[:-1])
        # Next Target: 1..9
        m_n, q_n, J_n, p_n = injector(x_all[1:], r_all[1:])
        
        if step % 10 == 0: readout.update_prototypes()
        
        # Loss 1: Contrastive (Separation)
        probs = readout.read_prob(q_c, beta=10.0)
        loss_contrast = F.nll_loss(torch.log(probs + 1e-9), ids_all[:-1])
        
        # Loss 2: Flow (Kinematic)
        # q_pred = Exp(q, p * dt)
        q_flow = geometry.exp_map(q_c, p_c * dt)
        loss_flow = torch.mean((q_flow - q_n)**2)
        
        # Loss 3: Integrator Consistency
        # q_int = Integrator(q, p, J, m, ff, dt)
        # Note: Integrator uses force field. If ff is random, it will diverge from flow.
        # But force is included. Ideally Flow predicts Integrator result?
        # Or Flow predicts Target, and Integrator predicts Target?
        # "Consistency" usually means q_int ~ q_flow or q_int ~ q_target.
        # Temp-02 says: "confirm integrator ... matches flow semantics".
        # Let's align both to global target.
        
        # We need to simulate full interaction?
        # For minimal seq, q_c is batch. ForceField computes pairwise forces within batch.
        # This is fine.
        q_int, _, _ = integrator.step(q_c, p_c, J_c, m_c, ff, dt=dt)
        loss_consist = torch.mean((q_int - q_n)**2)
        
        loss = loss_contrast + 5.0 * loss_flow + 5.0 * loss_consist
        
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: Total {loss.item():.4f} | C {loss_contrast.item():.3f} Flow {loss_flow.item():.4f} Cons {loss_consist.item():.4f}")
            
        if loss_contrast.item() < 0.1 and loss_flow.item() < 0.001 and loss_consist.item() < 0.001:
            print(f"Phase 3 Converged at Step {step}!")
            return True
            
    print("Phase 3 Failed.")
    return False

if __name__ == "__main__":
    test_phase3()


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
from aurora.engine.memory import HyperbolicMemory
from aurora.physics import geometry

def test_phase4_5():
    print("=== Phase 4 & 5: Full Loop (Potentials + LTM) Test ===")
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
    ff = ForceField(dim=DIM, G=0.01, lambda_gauge=0.01, k_geo=0.01, mu=1.0).to(device)
    integrator = LieIntegrator()
    ltm = HyperbolicMemory(DIM, max_capacity=100, device=device).to(device)
    
    optimizer = optim.Adam(list(injector.parameters()) + list(ff.parameters()), lr=0.005)
    
    # Dataset
    x_list = [ord(c)/65535.0 for c in vocab_chars]
    x_all = torch.tensor(x_list, dtype=torch.float, device=device).unsqueeze(1)
    r_all = torch.tensor([0.8]*len(vocab_chars), dtype=torch.float, device=device).unsqueeze(1)
    ids_all = torch.arange(len(vocab_chars), device=device)
    
    dt = 0.1
    print("Starting Integration Loop...")
    
    for step in range(1500):
        optimizer.zero_grad()
        
        # Inject
        m_c, q_c, J_c, p_c = injector(x_all[:-1], r_all[:-1])
        m_n, q_n, J_n, p_n = injector(x_all[1:], r_all[1:])
        
        if step % 10 == 0: readout.update_prototypes()
        
        # LTM Query (Simulate Interference)
        mem = ltm.query(q_c, k=5)
        q_mem_l, m_mem_l, J_mem_l = [], [], []
        if mem:
             q_mem_l.append(mem['q'].reshape(-1, DIM))
             m_mem_l.append(mem['m'].reshape(-1, 1))
             J_mem_l.append(mem['J'].reshape(-1, DIM, DIM))
             
        # Combine
        # Using detach for memory to avoid graph explosion
        q_all = torch.cat(q_mem_l + [q_c], dim=0)
        m_all = torch.cat(m_mem_l + [m_c], dim=0)
        J_all = torch.cat(J_mem_l + [J_c], dim=0)
        
        # 1. Contrastive
        probs = readout.read_prob(q_c, beta=10.0)
        loss_contrast = F.nll_loss(torch.log(probs + 1e-9), ids_all[:-1])
        
        # 2. Flow
        q_flow = geometry.exp_map(q_c, p_c * dt)
        loss_flow = torch.mean((q_flow - q_n)**2)
        
        # 3. Potentials (Annealed)
        _, E_pot = ff.compute_forces(q_all, m_all, J_all, return_grads=False)
        w_pot = min(0.1, step / 1000.0) # Anneal to 0.1
        
        # 4. Consistency
        q_int, _, _ = integrator.step(q_c, p_c, J_c, m_c, ff, dt=dt)
        loss_consist = torch.mean((q_int - q_n)**2)
        
        loss = loss_contrast + 5.0 * loss_flow + 5.0 * loss_consist + w_pot * E_pot
        
        loss.backward()
        optimizer.step()
        
        # LTM Update (Simulate Noise injection)
        if step % 20 == 0:
             ltm.add(q_c.detach(), m_c.detach(), J_c.detach(), p_c.detach())
        
        if step % 100 == 0:
            print(f"Step {step}: Total {loss.item():.4f} | C {loss_contrast.item():.3f} F {loss_flow.item():.4f} Cons {loss_consist.item():.4f} E {E_pot.item():.2f}")
            
        if loss_contrast.item() < 0.1 and loss_flow.item() < 0.005 and loss_consist.item() < 0.005:
            print(f"Phase 4&5 Converged at Step {step}!")
            return True
            
    print("Phase 4&5 Failed.")
    return False

if __name__ == "__main__":
    test_phase4_5()

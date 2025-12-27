"""
Aurora Interactive Chat (CCD v3.1).
===================================

Demonstrates "Natural Language Interaction" via geometric field dynamics.
Pipeline:
1. Input Stream -> Injector -> Physics Evolution -> LTM.
2. Generation -> Readout -> Sampling -> Self-Injection.

Ref: Axiom 3.3 (Readout) & 4.3 (Memory).
"""

import sys
import os
import torch
import torch.nn.functional as F
import json
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data.data_pipeline import GlobalEntropyStats
from aurora.model.injector import AuroraInjector
from aurora.engine.forces import ForceField
from aurora.engine.integrator import LieIntegrator
from aurora.engine.memory import HyperbolicMemory
from aurora.engine.readout import ReadoutMechanism
from aurora.physics import geometry

# Config
DIM = 5
EMBED_DIM = 32
HIDDEN_DIM = 64
MEMORY_SIZE = 10000
CONTEXT_K = 5
DEVICE = 'cpu' # Prototype on CPU

def main():
    print("Initializing Aurora v3.1 Engine...")
    
    # 1. Load Resources
    stats_path = "data/entropy_stats.json"
    if not os.path.exists(stats_path):
        print("Error: data/entropy_stats.json missing.")
        return
        
    entropy_stats = GlobalEntropyStats(stats_path)
    vocab_list = sorted(list(entropy_stats.stats.keys()))
    char_to_id = {c:i for i,c in enumerate(vocab_list)}
    id_to_char = {i:c for i,c in enumerate(vocab_list)}
    VOCAB_SIZE = len(vocab_list)
    
    # 2. Model
    injector = AuroraInjector(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, DIM).to(DEVICE)
    model_path = "data/aurora_v3_1.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        injector.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("Warning: No trained model found. Using random weights.")
        
    # 3. Engines
    ff = ForceField(G=2.0, lambda_gauge=1.5, k_geo=0.2)
    integrator = LieIntegrator(ff)
    ltm = HyperbolicMemory(DIM, MEMORY_SIZE, device=DEVICE)
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    
    # State Buffers
    active_q = []
    active_m = []
    active_J = []
    active_p = []
    MAX_ATOMS = 10 
    
    print("\n--- Aurora v3.1 Chat ---")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # 1. Ingest Input
        # Process chars
        # For simplicity, we just inject them and let physics settle.
        
        last_q = None
        
        with torch.no_grad():
            for char in user_input:
                if char not in char_to_id:
                    continue
                    
                cid = char_to_id[char]
                r_tgt = entropy_stats.get_radial_target(char)
                
                # Inject
                cid_t = torch.tensor([[cid]], dtype=torch.long, device=DEVICE)
                r_t = torch.tensor([[r_tgt]], dtype=torch.float, device=DEVICE)
                
                m, q_rest, J, p_init = injector(cid_t, r_t)
                
                q_c = q_rest.view(1, DIM)
                
                # Noise & Clamp
                noise = torch.randn_like(q_c) * 0.01
                q_c = q_c + noise
                qn = torch.norm(q_c, dim=-1, keepdim=True)
                mask = qn > 0.95
                q_c = torch.where(mask, q_c * (0.95 / (qn + 1e-9)), q_c)

                m_c = m.view(1, 1)
                J_c = J.view(1, DIM, DIM)
                p_c = p_init.view(1, DIM)
                
                # Retrieval
                q_mem_l, m_mem_l, J_mem_l = [], [], []
                mem_data = ltm.query(q_c, k=CONTEXT_K)
                if mem_data:
                    qs = mem_data['q'].shape
                    if len(qs) == 3:
                         q_mem_l.append(mem_data['q'].view(-1, DIM))
                         m_mem_l.append(mem_data['m'].view(-1, 1))
                         J_mem_l.append(mem_data['J'].view(-1, DIM, DIM))
                    else:
                         q_mem_l.append(mem_data['q'].reshape(-1, DIM))
                         m_mem_l.append(mem_data['m'].reshape(-1, 1))
                         J_mem_l.append(mem_data['J'].reshape(-1, DIM, DIM))
                
                # Construct System
                # Note: We don't optimize here, just integrate?
                # Actually, integrator.step() moves particles.
                # In training we just optimized Potential.
                # Here we want Dynamics.
                
                q_parts = q_mem_l + active_q + [q_c]
                m_parts = m_mem_l + active_m + [m_c]
                J_parts = J_mem_l + active_J + [J_c]
                
                q_sys = torch.cat(q_parts, dim=0)
                m_sys = torch.cat(m_parts, dim=0)
                J_sys = torch.cat(J_parts, dim=0)
                p_sys = torch.zeros_like(q_sys) # Zero momentum start for context? Or stored?
                # Ideally retrieve p from memory too.
                # For now, simplistic.
                
                # Evolution Step (Micro-time)
                # state = {'q': q_sys, 'p': p_sys, 'm': m_sys, 'J': J_sys}
                # state = integrator.step(state, dt=0.01)
                # q_c updated.
                
                # Push to Buffer
                if len(active_q) >= MAX_ATOMS:
                    old_q = active_q.pop(0)
                    old_m = active_m.pop(0)
                    old_J = active_J.pop(0)
                    active_p.pop(0)
                    ltm.add(old_q, old_m, old_J, torch.zeros_like(old_q))
                    
                active_q.append(q_c)
                active_m.append(m_c)
                active_J.append(J_c)
                active_p.append(p_c)
                
                last_q = q_c
                
        # 2. Generate Reply
        print("Aurora: ", end='', flush=True)
        # Use last_q as seed?
        # Or evolve system?
        
        gen_len = 0
        max_len = 50
        
        curr_q = last_q if last_q is not None else torch.zeros(1, DIM)
        
        while gen_len < max_len:
             # Readout
             probs = readout.read_prob(curr_q, beta=10.0) # Sharp beta
             # Sample
             dist = torch.distributions.Categorical(probs)
             idx = dist.sample()
             char = id_to_char[idx.item()]
             
             print(char, end='', flush=True)
             gen_len += 1
             
             if char == '\n': break
             
             # Self-Inject to move thought forward
             cid = torch.tensor([[idx]], dtype=torch.long)
             r = torch.tensor([[entropy_stats.get_radial_target(char)]], dtype=torch.float)
             m, q_rest, J, p = injector(cid, r)
             
             # Perturb curr_q towards q_rest?
             # Or just set curr_q = q_rest?
             # For continuity, we should integrate.
             # "Thought moves from A to B".
             # Start at curr_q, Target is q_rest.
             # One step evolution towards target?
             
             # Simple autoregression:
             curr_q = q_rest.view(1, DIM)
             
             # Add to memory context
             if len(active_q) >= MAX_ATOMS:
                active_q.pop(0)
                active_m.pop(0)
                active_J.pop(0)
                active_p.pop(0)
             
             active_q.append(curr_q)
             active_m.append(m.view(1,1))
             active_J.append(J.view(1,DIM,DIM))
             active_p.append(p.view(1,DIM))
             
        print() 
        
if __name__ == "__main__":
    main()

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
    # Phase 38 Config: Landau/Kac Physics + Flow
    ff = ForceField(G=1.0, lambda_gauge=1.5, k_geo=1.0, mu=1.0, lambda_quartic=0.01)
    integrator = LieIntegrator(ff)
    ltm = HyperbolicMemory(DIM, MEMORY_SIZE, device=DEVICE)
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    
    # State Buffers
    active_q = []
    active_m = []
    active_J = []
    
    # Max Context usually small for chat flow, relying on LTM for history
    MAX_ATOMS = 20 
    
    print("\n--- Aurora v3.1 Chat (Physics Enhanced) ---")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # 1. Ingest Input
        last_q = None
        last_p = None
        
        with torch.no_grad():
            for char in user_input:
                if char not in char_to_id: continue
                    
                cid = char_to_id[char]
                r_tgt = entropy_stats.get_radial_target(char)
                
                # Inject
                cid_t = torch.tensor([[cid]], dtype=torch.long, device=DEVICE)
                r_t = torch.tensor([[r_tgt]], dtype=torch.float, device=DEVICE)
                
                m, q_rest, J, p_init = injector(cid_t, r_t)
                
                q_c = q_rest.view(1, DIM)
                p_c = p_init.view(1, DIM)
                m_c = m.view(1, 1)
                J_c = J.view(1, DIM, DIM)
                
                # Push to Buffer
                if len(active_q) >= MAX_ATOMS:
                    old_q = active_q.pop(0)
                    old_m = active_m.pop(0)
                    old_J = active_J.pop(0)
                    ltm.add(old_q, old_m, old_J, torch.zeros_like(old_q))
                    
                active_q.append(q_c)
                active_m.append(m_c)
                active_J.append(J_c)
                
                last_q = q_c
                last_p = p_c
                
        # 2. Generate Reply
        print("Aurora: ", end='', flush=True)
        
        gen_len = 0
        max_len = 50
        
        # Start from end of user input
        curr_q = last_q if last_q is not None else torch.zeros(1, DIM, device=DEVICE)
        curr_p = last_p if last_p is not None else torch.zeros(1, DIM, device=DEVICE)
        
        while gen_len < max_len:
             # DYNAMICS: Flow along Momentum
             # q_{t+1} = Exp(q_t, p_t * dt)
             # This is the "Grammatical River"
             curr_q = geometry.exp_map(curr_q, curr_p * 1.0)
             
             # Readout at new position
             probs = readout.read_prob(curr_q, beta=10.0) 
             dist = torch.distributions.Categorical(probs)
             idx = dist.sample()
             char = id_to_char[idx.item()]
             
             print(char, end='', flush=True)
             gen_len += 1
             
             if char == '\n': break
             
             # Self-Inject to update Momentum for NEXT step
             cid = torch.tensor([[idx]], dtype=torch.long, device=DEVICE)
             r = torch.tensor([[entropy_stats.get_radial_target(char)]], dtype=torch.float, device=DEVICE)
             
             m, q_new, J, p_new = injector(cid, r)
             
             # Update State
             # Note: We trust the physics prediction (curr_q) or the injector's target (q_new)?
             # Ideally they converge. For strict flow, we use q_new as the "grounded" point 
             # from which we launch the next momentum.
             curr_q = q_new.view(1, DIM)
             curr_p = p_new.view(1, DIM)
             
             # Memory
             if len(active_q) >= MAX_ATOMS:
                active_q.pop(0)
                active_m.pop(0)
                active_J.pop(0)
             
             active_q.append(curr_q)
             active_m.append(m.view(1,1))
             active_J.append(J.view(1,DIM,DIM))
             
        print() 
        
if __name__ == "__main__":
    main()

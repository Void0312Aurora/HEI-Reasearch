"""
Aurora Trainer v3.1 (Phase 36: Semantic Nucleation - Experiment 2).
===================================================================

Upgrades:
1. Batched Training (Temporal Chunks) for GPU utilization.
2. Tuned Physics (Lower G, Higher k_geo).
3. Scale: 50k steps.

"""

import sys
import os
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import argparse
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data.data_pipeline import GlobalEntropyStats, CharStreamProcessor
from aurora.model.injector import AuroraInjector
from aurora.engine.forces import ForceField
from aurora.engine.integrator import LieIntegrator
from aurora.engine.rheology import RheologicalOptimizer
from aurora.engine.memory import HyperbolicMemory
from aurora.engine.readout import ReadoutMechanism

# Experiment 2 Config
DIM = 5
EMBED_DIM = 32
HIDDEN_DIM = 64
LR = 0.01
YIELD_STRESS = 0.001 
ELASTIC_K = 0.0001
MAX_ATOMS = 100 # Increased Buffer
CONTEXT_K = 10 
MEMORY_SIZE = 10000
BATCH_SIZE = 64 # Chunks per step

def eval_probe(injector, entropy_stats, id_to_char, prompt="你好", dim=DIM, device='cpu'):
    """Generate text to probe nucleation."""
    injector.eval()
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    char_to_id = {v:k for k,v in id_to_char.items()}
    
    gen_text = ""
    curr_q = torch.zeros(1, dim, device=device)
    
    with torch.no_grad():
        for char in prompt:
             if char not in char_to_id: continue
             cid = torch.tensor([[char_to_id[char]]], dtype=torch.long, device=device)
             r = torch.tensor([[entropy_stats.get_radial_target(char)]], dtype=torch.float, device=device)
             m, q, J, p = injector(cid, r)
             curr_q = q.view(1, dim)
    
    for _ in range(20): # Longer probe
        probs = readout.read_prob(curr_q, beta=5.0) # Lower beta for sampling
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample().item()
        char = id_to_char[idx]
        gen_text += char
        
        cid = torch.tensor([[idx]], dtype=torch.long, device=device)
        r = torch.tensor([[entropy_stats.get_radial_target(char)]], dtype=torch.float, device=device)
        m, q, J, p = injector(cid, r)
        curr_q = q.view(1, dim)
        
    injector.train()
    return gen_text

def train(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger("AuroraTrain")
    
    device = args.device
    logger.info(f"Device: {device}")
    
    # 1. Data
    logger.info("Loading Entropy Stats...")
    stats_path = "data/entropy_stats.json"
    entropy_stats = GlobalEntropyStats(stats_path)
    
    vocab_list = sorted(list(entropy_stats.stats.keys()))
    char_to_id = {c:i for i,c in enumerate(vocab_list)}
    id_to_char = {i:c for i,c in enumerate(vocab_list)}
    VOCAB_SIZE = len(vocab_list)
    logger.info(f"Vocab Size: {VOCAB_SIZE}")
    
    stream = CharStreamProcessor("data/CLUE/CLUECorpusSmall.txt")
    
    # 2. Model
    injector = AuroraInjector(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, DIM).to(device)
    
    # Physics (Tuned)
    # Higher G (1.0) for Nucleation, Higher k_geo (1.0)
    ff = ForceField(G=1.0, lambda_gauge=1.5, k_geo=1.0)
    ltm = HyperbolicMemory(DIM, MEMORY_SIZE, device=device).to(device)
    
    # Optimizer
    rheo_opt = RheologicalOptimizer(injector.parameters(), lr=LR, yield_stress=YIELD_STRESS, elastic_k=ELASTIC_K)
    
    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.model_path):
        logger.info(f"Resuming from {args.model_path}")
        injector.load_state_dict(torch.load(args.model_path, map_location=device))
    
    logger.info(f"Starting Batched Training for {args.steps} steps...")
    
    active_m = []
    active_q = []
    active_J = []
    
    step_cnt = start_step
    loss_smooth = 0
    
    # Batch Stream
    batch_gen = stream.stream_batches(batch_size=BATCH_SIZE)
    
    try:
        pbar = tqdm(range(args.steps))
        for _ in pbar:
            step_cnt += 1
            
            # Annealing
            warmup_steps = 10000
            w_start = 0.001
            w_end = 0.1
            if step_cnt < warmup_steps:
                w_pot = w_start + (w_end - w_start) * (step_cnt / warmup_steps)
            else:
                w_pot = w_end
            
            try:
                batch_chars = next(batch_gen)
            except StopIteration:
                batch_gen = stream.stream_batches(batch_size=BATCH_SIZE)
                batch_chars = next(batch_gen)
            
            # Prepare Batch Tensors
            cids = []
            rs = []
            valid_mask = []
            
            for char in batch_chars:
                if char in char_to_id:
                    cids.append(char_to_id[char])
                    rs.append(entropy_stats.get_radial_target(char))
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            
            if not cids: continue
            
            # To Tensor (B, 1) each? No, (B, 1) total input for Injector?
            # Injector(ids, r) returns (B, ...).
            # Yes.
            
            cid_t = torch.tensor(cids, dtype=torch.long, device=device).unsqueeze(1) # (B, 1)
            r_t = torch.tensor(rs, dtype=torch.float, device=device).unsqueeze(1) # (B, 1)
            
            # 1. Inject Batch
            m_curr, q_rest, J_curr, p_init = injector(cid_t, r_t)
            
            # Output Shapes: (B, 1, ...)
            m_batch = m_curr.squeeze(1) # (B, 1)
            q_batch = q_rest.squeeze(1) # (B, D)
            J_batch = J_curr.squeeze(1) # (B, D, D)
            p_batch = p_init.squeeze(1) # (B, D)
            
            # Noise & Clamp
            noise = torch.randn_like(q_batch) * 0.01
            q_batch = q_batch + noise
            qn = torch.norm(q_batch, dim=-1, keepdim=True)
            mask = qn > 0.95
            q_batch = torch.where(mask, q_batch * (0.95 / (qn + 1e-9)), q_batch)
            
            # 2. Retrieve Batch Context?
            # Querying LTM for EACH particle in batch is expensive if sequential.
            # But we can query for the Mean? Or just query for Last?
            # Or query all in parallel if LTM supports it.
            # memory.query: q_query (B, D) -> returns (B, k, D).
            # Yes, my implementation supports Batch Query!
            
            mem_data = ltm.query(q_batch, k=CONTEXT_K)
            
            q_mem_l, m_mem_l, J_mem_l = [], [], []
            if mem_data:
                 # mem_q: (B, k, D). We flatten to (B*k, D) for the background field?
                 # Physics compute forces on (N_total).
                 # N_total = N_mem + N_buffer + N_batch.
                 # N_mem = B * K.
                 
                 # Flatten memory
                 q_mem_l.append(mem_data['q'].reshape(-1, DIM))
                 m_mem_l.append(mem_data['m'].reshape(-1, 1))
                 J_mem_l.append(mem_data['J'].reshape(-1, DIM, DIM))

            # 3. Physics
            # We treat the WHOLE batch as a "cloud" entering the system.
            # Plus existing Buffer.
            
            q_parts = q_mem_l + active_q + [q_batch]
            m_parts = m_mem_l + active_m + [m_batch]
            J_parts = J_mem_l + active_J + [J_batch]
            
            q_in = torch.cat(q_parts, dim=0)
            m_in = torch.cat(m_parts, dim=0)
            J_in = torch.cat(J_parts, dim=0)
            
            # Force Calculation
            # Normalize Energy by N^2 to prevent explosion with System Size
            N_sys = q_in.size(0)
            _, E_pot_raw = ff.compute_forces(q_in, m_in, J_in)
            E_pot = E_pot_raw / (N_sys**2 + 1e-9)
            
            # Loss
            # Constraint: Sum over batch
            r_curr = torch.norm(q_batch, dim=-1) # (B)
            # Match r_t (B)
            loss_r = torch.mean((r_curr - r_t.squeeze(1))**2)
            
            loss = loss_r + w_pot * E_pot
            
            if torch.isnan(loss):
                logger.warning(f"NaN Loss at step {step_cnt}. Resetting Buffer.")
                active_q = []
                active_m = []
                active_J = []
                injector.zero_grad()
                continue
                
            injector.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(injector.parameters(), max_norm=1.0)
            rheo_opt.step()
            
            loss_smooth = 0.9 * loss_smooth + 0.1 * loss.item()
            pbar.set_description(f"L:{loss_smooth:.4f} W:{w_pot:.3f}")
            
            # Update Buffer & LTM
            # We push the WHOLE batch to buffer.
            # If buffer full, pop oldest batch?
            # active_q is list of Tensors.
            # Let's keep it simply: active_q stores Batches.
            
            if len(active_q) >= MAX_ATOMS // BATCH_SIZE: 
                # Pop chunks
                old_q = active_q.pop(0)
                old_m = active_m.pop(0)
                old_J = active_J.pop(0)
                
                # Push Chunk to LTM
                # Must supply dummy p
                ltm.add(old_q, old_m, old_J, torch.zeros_like(old_q))
                
            active_q.append(q_batch.detach())
            active_m.append(m_batch.detach())
            active_J.append(J_batch.detach())
            
            # Probe
            if step_cnt % args.probe_freq == 0:
                 gen = eval_probe(injector, entropy_stats, id_to_char, device=device)
                 logger.info(f"\n[Step {step_cnt}] Probe: {gen}")
                 
            # Save
            if step_cnt % args.save_freq == 0:
                torch.save(injector.state_dict(), args.model_path)
                
    except KeyboardInterrupt:
        logger.info("Training interrupted.")
        
    logger.info("Training Finished.")
    torch.save(injector.state_dict(), args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_path", type=str, default="data/aurora_v3_1.pth")
    parser.add_argument("--probe_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=5000)
    
    args = parser.parse_args()
    if args.device == 'cpu' and torch.cuda.is_available():
        args.device = 'cuda'
        
    train(args)

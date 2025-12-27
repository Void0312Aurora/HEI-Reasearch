"""
Aurora Trainer v3.1 (Phase 39: Full Theoretical Implementation).
================================================================

Implements the user-requested "Full Theory" overhaul.
1. Continuous Inputs (Axiom 0.1.2): Normalized Codepoints.
2. Lie Integrator Dynamics (Axiom 2.4): Consistency Loss.
3. Gauge Parallel Transport (Axiom 3.1.2): Route B (Discrete Edge).

"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
import argparse
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data.data_pipeline import GlobalEntropyStats, CharStreamProcessor
from aurora.model.injector import ContinuousInjector
from aurora.engine.forces import ForceField
from aurora.physics import geometry
from aurora.engine.integrator import LieIntegrator
from aurora.engine.rheology import RheologicalOptimizer
from aurora.engine.memory import HyperbolicMemory
from aurora.engine.readout import ReadoutMechanism

# Config
DIM = 16
HIDDEN_DIM = 256
LR = 0.005 # Lower LR for stability with new dynamics
YIELD_STRESS = 0.001 
ELASTIC_K = 0.0001
MAX_ATOMS = 64 # Reduced for DIM=16 and full graph
CONTEXT_K = 5
MEMORY_SIZE = 20000
BATCH_SIZE = 32 # OOM Fix

def eval_probe(injector, entropy_stats, id_to_char, ff, prompt="你好", dim=DIM, device='cpu'):
    """Generate text WITH Full Dynamics (Route B PT)."""
    injector.eval()
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char) # id_to_char needed for decoding output, not input
    integrator = LieIntegrator()
    
    gen_text = ""
    active_q = []
    active_m = []
    active_J = []
    
    with torch.no_grad():
        # 1. Inject Prompt
        for char in prompt:
             codepoint = ord(char)
             x_norm = float(codepoint) / 65535.0
             r_val = entropy_stats.get_radial_target(char)
             
             x_t = torch.tensor([[x_norm]], dtype=torch.float, device=device)
             r_t = torch.tensor([[r_val]], dtype=torch.float, device=device)
             
             m, q, J, p = injector(x_t, r_t)
             # Store initial kinematic state? Probes usually just inject.
             active_q.append(q)
             active_m.append(m)
             active_J.append(J)
    
    # Generate
    if not active_q: return ""
    curr_q = active_q[-1]
    curr_p = torch.zeros_like(curr_q) # Start with zero momentum if not tracked
    curr_J = active_J[-1]
    curr_m = active_m[-1] # Mass of cursor
    
    for _ in range(20): 
        # 1. Physics Context
        q_in = torch.cat(active_q, dim=0)
        m_in = torch.cat(active_m, dim=0)
        J_in = torch.cat(active_J, dim=0) # (K, D, D)
        
        # 2. Integration Step
        # To simulate interaction, we treat the 'Cursor' (curr) as interacting with Context (in).
        # Lie Integrator Step
        # We need p! Let's assume some momentum kick from previous?
        dt = 0.1
        
        # Step: q, p, J -> q_next, p_next, J_next
        # But compute_forces needs (N, ...)
        # We append cursor to context? Cursor IS last of q_in.
        
        # We update the *last* particle in the chain (the cursor)
        # But wait, cursor should be separate?
        # Let's simple simulation: Cursor is free particle acted on by Context.
        
        # Call Integrator on the WHOLE system? Or just the cursor?
        # Simpler: Just evaluate forces on the cursor due to others.
        
        # For probe, let's use the explicit detailed integration step.
        # But integrator.step computes forces internally for the batch.
        # Let's pass the whole system to integrator, but only take the last particle's update.
        
        q_next, p_next, J_next = integrator.step(q_in, curr_p.expand(q_in.size(0), -1), J_in, m_in, ff, dt=dt)
        
        # Take the last one
        curr_q = q_next[-1].unsqueeze(0)
        curr_p = p_next[-1].unsqueeze(0)
        curr_J = J_next[-1].unsqueeze(0)
        
        # 3. Readout
        probs = readout.read_prob(curr_q, beta=10.0)
        idx = probs.argmax().item()
        char = id_to_char[idx] # Decode using vocab
        gen_text += char
        
        # 4. Inject Feedback (Self-Injection)
        codepoint = ord(char)
        x_norm = float(codepoint) / 65535.0
        r_val = entropy_stats.get_radial_target(char)
        
        x_t = torch.tensor([[x_norm]], dtype=torch.float, device=device)
        r_t = torch.tensor([[r_val]], dtype=torch.float, device=device)
        
        m, q, J, p_kick = injector(x_t, r_t)
        
        # Update Cursor State (Soft Projection?)
        # For probe, let's just replace/anchor like before for stability.
        # Or better: Add to context, update cursor position to new anchor?
        active_q.append(q)
        active_m.append(m)
        active_J.append(J)
        
        curr_q = q
        curr_J = J
        # Momentum kick?
        curr_p = curr_p + p_kick
        
        if len(active_q) > 20:
            active_q.pop(0); active_m.pop(0); active_J.pop(0)
            
    injector.train()
    return gen_text

def train(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger("AuroraTrain")
    # File Handler
    fh = logging.FileHandler("training_log.txt")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    device = args.device
    logger.info(f"Device: {device}")
    
    # 1. Data Helper
    logger.info("Loading Entropy Stats...")
    entropy_stats = GlobalEntropyStats("data/entropy_stats.json")
    
    # For Readout Only
    vocab_list = sorted(list(entropy_stats.stats.keys()))
    char_to_id = {c:i for i,c in enumerate(vocab_list)}
    id_to_char = {i:c for i,c in enumerate(vocab_list)}
    # Note: Injector does NOT use these. Only ReadoutMechanism does.
    
    stream = CharStreamProcessor("data/CLUE/CLUECorpusSmall.txt")
    
    # 2. Model: Continuous Injector
    # input_dim=1 (normalized codepoint)
    injector = ContinuousInjector(input_dim=1, hidden_dim=HIDDEN_DIM, dim=DIM).to(device)
    
    # Physics: Valid ForceField
    # Tune to prevent Mode Collapse (Weak Gravity, Weak Gauge initially)
    # Strong Contrastive must dominate early training.
    # Reduce k_geo to prevents boundary pinning (-300 energy).
    ff = ForceField(dim=DIM, G=0.01, lambda_gauge=0.05, k_geo=0.01, mu=1.0).to(device)
    ltm = HyperbolicMemory(DIM, MEMORY_SIZE, device=device).to(device)
    lie_integrator = LieIntegrator()
    
    # Readout
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    
    # Optimizer (Include FF parameters for Connection Learning)
    all_params = list(injector.parameters()) + list(ff.parameters())
    rheo_opt = RheologicalOptimizer(all_params, lr=LR, yield_stress=YIELD_STRESS, elastic_k=ELASTIC_K)
    
    logger.info(f"Starting Training for {args.steps} steps...")
    
    active_m = []
    active_q = []
    active_J = []
    
    step_cnt = 0
    loss_smooth = 0
    
    batch_gen = stream.stream_batches(batch_size=BATCH_SIZE)
    
    pbar = tqdm(range(args.steps))
    for _ in pbar:
        step_cnt += 1
        
        # Get Batch
        try:
            batch_chars = next(batch_gen)
        except StopIteration:
            batch_gen = stream.stream_batches(batch_size=BATCH_SIZE)
            batch_chars = next(batch_gen)
            
        # Process Continuous Inputs
        x_list = []
        r_list = []
        targets = [] # For Contrastive Loss Only
        
        valid_indices = []
        for i, char in enumerate(batch_chars):
            # 1. Normalize Codepoint
            if len(char) == 1:
                cp = ord(char)
            else:
                cp = 0
            x_norm = float(cp) / 65535.0
            
            # 2. Get R target
            r = entropy_stats.get_radial_target(char)
            
            x_list.append(x_norm)
            r_list.append(r)
            
            if char in char_to_id:
                targets.append(char_to_id[char])
                valid_indices.append(i)
        
        if not x_list: continue
        
        x_t = torch.tensor(x_list, dtype=torch.float, device=device).unsqueeze(1) # (B, 1)
        r_t = torch.tensor(r_list, dtype=torch.float, device=device).unsqueeze(1) # (B, 1)
        target_ids = torch.tensor(targets, dtype=torch.long, device=device) # (B,)
        
        # === 1. Inject (Mapping Sigma -> Phase Space) ===
        m_batch, q_batch, J_batch, p_batch = injector(x_t, r_t)
        
        m_b = m_batch
        q_b = q_batch
        J_b = J_batch
        
        # Noise for Robustness
        q_b = q_b + torch.randn_like(q_b) * 0.005
        
        # === 2. Retrieve Context ===
        mem_data = ltm.query(q_b, k=CONTEXT_K)
        q_mem_l, m_mem_l, J_mem_l = [], [], []
        if mem_data:
            q_mem_l.append(mem_data['q'].reshape(-1, DIM))
            m_mem_l.append(mem_data['m'].reshape(-1, 1))
            J_mem_l.append(mem_data['J'].reshape(-1, DIM, DIM))
            
        # Combine System
        q_all = torch.cat(q_mem_l + active_q + [q_b], dim=0)
        m_all = torch.cat(m_mem_l + active_m + [m_b], dim=0)
        J_all = torch.cat(J_mem_l + active_J + [J_b], dim=0)
        
        # === 3. Dynamics & Losses ===

        # A. Constraint Loss
        r_curr = torch.norm(q_b, dim=-1)
        loss_r = torch.mean((r_curr - r_t.squeeze(1))**2)
        
        # B. Potential Energy
        _, E_pot = ff.compute_forces(q_all, m_all, J_all, return_grads=False)
        E_pot = E_pot 
        
        # C. Flow Loss
        # q_{t+1} target
        q_seq = q_b # (B, D)
        p_seq = p_batch # (B, D)
        
        loss_flow = torch.tensor(0.0, device=device)
        loss_consist = torch.tensor(0.0, device=device)
        
        if BATCH_SIZE > 1:
            q_curr = q_seq[:-1]
            p_curr = p_seq[:-1]
            q_next_target = q_seq[1:].detach()
            
            # Simple Flow
            q_pred_simple = geometry.exp_map(q_curr, p_curr)
            loss_flow = torch.mean((q_pred_simple - q_next_target)**2)
            
            # D. Integrator Consistency (Bridge to Phase 2) [New Axiom]
            # Requires that 1-step integration matches Flow Prediction
            dt = 0.1
            # Run integrator on the batch sequence (detached context for speed?)
            # Or just local pair? Integrator involves forces from neighbors.
            # Using integrator on q_curr with current context implies full N^2 force.
            # We can use q_all, but we need gradients flow back to p_curr.
            # Let's run integrator on q_curr (detached from context to save memory?)
            # ForceField handles inputs.
            
            # We only care about p_curr evolving q_curr correctly.
            # Forces might be noisy. Let's require q_next ~ Integrator(q_curr, p_curr)
            # including forces from q_next (future) is cheating? No, forces from q_curr neighbors.
            # For efficiency: Just use simple flow loss dominant, plus small consistency.
            
            # Consistency: q_integrator vs q_flow
            # q_int, p_int, J_int = lie_integrator.step(q_curr, p_curr, J_curr, m_curr, ff, dt)
            # If we run this, we backprop through ForceField.connection! This trains the gauge connection.
            # This is crucial for Route B PT.
            
            # We need J_curr and m_curr
            J_curr = J_b[:-1]
            m_curr = m_b[:-1]
            
            # 1-step Rollout
            q_int, _, _ = lie_integrator.step(q_curr, p_curr, J_curr, m_curr, ff, dt)
            
            # Consistency: Integrator result should match Target (Real Next Char)
            # This forces p and Forces/Connection to work together to reach q_next.
            loss_consist = torch.mean((q_int - q_next_target)**2)
            
        # E. Contrastive Loss (Meaning Basin)
        # Contrastive uses discrete target ID but continuous q
        # read_prob uses precomputed prototypes
        probs = readout.read_prob(q_b, beta=10.0)
        # targets is (B,)
        # Filter targets to match q_b (some might have been filtered out? No, x_list was synced)
        # valid_indices was for batch_chars vs char_to_id.
        # But x_list and targets are synced?
        # x_list has ALL chars. targets has only known chars.
        # Length mismatch possible?
        # x_list loops batch_chars. targets appends if in char_to_id.
        # So q_b corresponds to full batch. targets is smaller.
        # We need to mask q_b to valid indices.
        
        valid_mask = torch.tensor(valid_indices, device=device)
        if len(valid_mask) > 0:
            q_valid = q_b[valid_mask]
            probs_valid = readout.read_prob(q_valid, beta=10.0)
            loss_contrast = F.cross_entropy(torch.log(probs_valid + 1e-9), target_ids)
        else:
            loss_contrast = torch.tensor(0.0, device=device)
            
        # Total Loss
        # w_pot annealing: Delay physics until step 2000
        if step_cnt < 2000:
            w_pot = 0.0
        else:
            w_pot = min(0.1, 0.0 + (step_cnt - 2000) * 0.0001)
        loss = loss_r + w_pot * E_pot + 5.0 * loss_flow + 5.0 * loss_consist + 1.0 * loss_contrast
        
        if torch.isnan(loss):
            injector.zero_grad()
            active_q = [] # Reset
            continue
            
        rheo_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        rheo_opt.step()
        
        loss_smooth = 0.95 * loss_smooth + 0.05 * loss.item()
        pbar.set_description(f"L:{loss_smooth:.4f} Flow:{loss_flow.item():.3f} Cons:{loss_consist.item():.3f} C:{loss_contrast.item():.3f}")
        
        # Explicit Logging every 100 steps
        if step_cnt % 100 == 0:
            m_mean = torch.mean(m_b).item()
            with torch.no_grad():
                v_g = ff.potential_geometry(q_all).item()
                v_m = ff.potential_mass(q_all, m_all).item()
                v_u = ff.potential_gauge(q_all, J_all).item()
                v_c = ff.potential_chemical(m_all).item()
                v_r = 0.001 * torch.sum(J_all**2).item()
                
            log_msg = f"[Step {step_cnt}] L:{loss.item():.2f} | V_all: M {v_m:.1f} G {v_u:.1f} Geo {v_g:.1f} Ch {v_c:.1f} Rot {v_r:.1f} | C:{loss_contrast.item():.2f}"
            tqdm.write(log_msg)
            logger.info(log_msg)

        # Buffer Update
        if len(active_q) > 5:
             active_q.pop(0); active_m.pop(0); active_J.pop(0)
        active_q.append(q_b.detach())
        active_m.append(m_b.detach())
        active_J.append(J_b.detach())
        
        if step_cnt % args.probe_freq == 0:
             txt = eval_probe(injector, entropy_stats, id_to_char, ff, device=device)
             msg = f"\n[Step {step_cnt}] Probe: {txt}"
             tqdm.write(msg)
             logger.info(msg)
             
        if step_cnt % args.save_freq == 0:
            torch.save(injector.state_dict(), args.model_path)
            
    torch.save(injector.state_dict(), args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_path", type=str, default="data/aurora_v3_1.pth")
    parser.add_argument("--probe_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=5000)
    
    args = parser.parse_args()
    if args.device == 'cpu' and torch.cuda.is_available(): args.device = 'cuda'
    
    train(args)

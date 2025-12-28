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
MAX_ATOMS = 32 # Reduced for OOM
CONTEXT_K = 2 # Reduced for OOM
BATCH_SIZE = 16 # Reduced for OOM
MEMORY_SIZE = 20000

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

    
def eval_tf_probe_stats(injector, entropy_stats, id_to_char, char_to_id, ff, text="其中最重要的是", dim=DIM, device='cpu'):
    """TF Probe with Trend Metrics (Rank, Hits, Identity Bias)."""
    injector.eval()
    readout = ReadoutMechanism(injector, entropy_stats, id_to_char)
    integrator = LieIntegrator()
    
    active_q, active_m, active_J, active_p = [], [], [], []
    
    gt_ranks = []
    is_identity = []
    top50_hit = []
    top20_hit = []
    
    detailed_log = []
    
    with torch.no_grad():
         pass 
         
    for i in range(len(text) - 1):
        curr_char = text[i]
        next_char_gt = text[i+1]
        
        # Inject
        codepoint = ord(curr_char)
        x_norm = float(codepoint) / 65535.0
        r_val = entropy_stats.get_radial_target(curr_char)
        x_t = torch.tensor([[x_norm]], dtype=torch.float, device=device)
        r_t = torch.tensor([[r_val]], dtype=torch.float, device=device)
        
        with torch.no_grad():
             m, q, J, p = injector(x_t, r_t)
        q.requires_grad_(True)
        # p needs grad? In probe we don't train, so no.
        
        active_q.append(q); active_m.append(m); active_J.append(J); active_p.append(p)
        
        # Integrate with full history context
        q_in = torch.cat(active_q, dim=0)
        m_in = torch.cat(active_m, dim=0)
        J_in = torch.cat(active_J, dim=0)
        p_in = torch.cat(active_p, dim=0)
        
        dt = 0.1
        
        # Use actual momentum p_in
        q_next_pred, _, _ = integrator.step(q_in, p_in, J_in, m_in, ff, dt=dt)
        q_target_pred = q_next_pred[-1].unsqueeze(0)
        
        # Readout
        with torch.no_grad():
            probs = readout.read_prob(q_target_pred, beta=10.0)
            
            # 1. Rank & Hits
            gt_id = char_to_id.get(next_char_gt, -1)
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            
            if gt_id != -1:
                rank = (sorted_indices == gt_id).nonzero(as_tuple=True)[0].item() + 1
                gt_ranks.append(rank)
                top50_hit.append(1 if rank <= 50 else 0)
                top20_hit.append(1 if rank <= 20 else 0)
            else:
                rank = -1
                
            # 2. Identity Bias
            pred_id = probs.argmax().item()
            pred_char = id_to_char.get(pred_id, '?')
            if pred_char == curr_char:
                is_identity.append(1)
            else:
                is_identity.append(0)
            
            # Log Line
            gt_prob = probs[0, gt_id].item() if gt_id != -1 else 0.0
            detailed_log.append(f"{curr_char}->{next_char_gt} [GT:{gt_prob:.1e} R:{rank} | P:{pred_char}]")
            
        if len(active_q) > 20:
             active_q.pop(0); active_m.pop(0); active_J.pop(0); active_p.pop(0)

    injector.train()
    
    # Compute Aggregate Stats
    avg_rank = sum(gt_ranks) / len(gt_ranks) if gt_ranks else 0
    # median rank
    gt_ranks.sort()
    med_rank = gt_ranks[len(gt_ranks)//2] if gt_ranks else 0
    
    t50_rate = sum(top50_hit) / len(top50_hit) if top50_hit else 0
    t20_rate = sum(top20_hit) / len(top20_hit) if top20_hit else 0
    id_rate = sum(is_identity) / len(is_identity) if is_identity else 0
    
    summary = f"\n=== Trend Metrics ===\nAvg Rank: {avg_rank:.1f}\nMedian Rank: {med_rank}\nTop50 Rate: {t50_rate:.2f}\nTop20 Rate: {t20_rate:.2f}\nIdentity Bias: {id_rate:.2f}\nDetailed:\n" + "\n".join(detailed_log)
    return summary

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
    
    # Resume Logic
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        injector.load_state_dict(checkpoint['injector_state'])
        ff.load_state_dict(checkpoint['ff_state'])
        # Optimize: Check if ltm_state exists (for backward compatibility if needed, though this is v3.1 new)
        if 'ltm_state' in checkpoint:
            ltm.load_state_dict(checkpoint['ltm_state'])
            ltm.ptr = checkpoint.get('ltm_ptr', 0)
            ltm.size = checkpoint.get('ltm_size', 0)
        
        step_cnt = checkpoint.get('step', 0)
        logger.info(f"Resumed at step {step_cnt}")
    
    start_step = step_cnt # For safe w_pot ramp
    
    batch_gen = stream.stream_batches(batch_size=BATCH_SIZE)
    
    # Adjust range for resume
    pbar = tqdm(range(step_cnt, args.steps))
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
            # Consistent with Integrator: q_next = exp(q, v * dt)
            dt = 0.1
            q_pred_simple = geometry.exp_map(q_curr, p_curr * dt)
            loss_flow = torch.mean((q_pred_simple - q_next_target)**2)
            
            # D. Integrator Consistency
            # Requires that 1-step integration matches Flow Prediction
            dt = 0.1
            
            # Context for Integrator = Memory + Active History
            # DETACH CONTEXT to prevent OOM (Truncated BPTT)
            q_mem_t = torch.cat(q_mem_l + active_q, dim=0).detach() if (q_mem_l or active_q) else None
            m_mem_t = torch.cat(m_mem_l + active_m, dim=0).detach() if (m_mem_l or active_m) else None
            J_mem_t = torch.cat(J_mem_l + active_J, dim=0).detach() if (J_mem_l or active_J) else None

            # Consistency: q_integrator vs q_flow
            # q_int, p_int, J_int = lie_integrator.step(q_curr, p_curr, J_curr, m_curr, ff, dt)
            
            # We need J_curr and m_curr
            J_curr = J_b[:-1]
            m_curr = m_b[:-1]
            
            # 1-step Rollout with Context
            q_int, _, _ = lie_integrator.step(q_curr, p_curr, J_curr, m_curr, ff, dt, 
                                              ctx_q=q_mem_t, ctx_m=m_mem_t, ctx_J=J_mem_t)
            
            # Consistency: Integrator result should match Target (Real Next Char)
            # This forces p and Forces/Connection to work together to reach q_next.
            loss_consist = torch.mean((q_int - q_next_target)**2)
            
        # E. Contrastive Loss (Meaning Basin) - NEXT TOKEN PREDICTION
        # Critique Fix: Train Readout on Evolved State (q_int) -> Next Char
        # q_int is (B-1, DIM), predicting indices 1 to B-1
        
        next_chars = batch_chars[1:] # The targets corresponding to q_int
        next_valid_indices = []
        next_target_ids = []
        
        for i, char in enumerate(next_chars):
            if char in char_to_id:
                next_valid_indices.append(i)
                next_target_ids.append(char_to_id[char])
                
        next_target_tensor = torch.tensor(next_target_ids, dtype=torch.long, device=device)
        next_mask = torch.tensor(next_valid_indices, dtype=torch.long, device=device)
        
        if len(next_mask) > 0:
            # Use Evolved State (q_int) for Readout
            q_evolved_valid = q_int[next_mask]
            
            # Readout Probabilities
            probs_next = readout.read_prob(q_evolved_valid, beta=10.0)
            
            # NLL Loss
            loss_contrast = F.nll_loss(torch.log(probs_next + 1e-9), next_target_tensor)
        else:
            loss_contrast = torch.tensor(0.0, device=device)
            
        # Total Loss
        # w_pot annealing: Restart Ramp Logic (Safe Resume)
        # Ramp from 0.0 to 0.1 over 1000 steps from start_step
        w_pot = min(0.1, 0.0 + (step_cnt - start_step) * 0.0001)
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
                
                # Batch Diversity Stats
                # Recalculate probs for q_b (Current State Readout) to monitor diversity/collapse
                probs = readout.read_prob(q_b, beta=10.0)
                pred_ids = probs.argmax(dim=1)
                unique_count = len(torch.unique(pred_ids))
                max_probs = probs.max(dim=1)[0].mean().item()
                # Entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean().item()

            log_msg = f"[Step {step_cnt}] L:{loss.item():.2f} | V_all: M {v_m:.1f} G {v_u:.1f} Geo {v_g:.1f} Ch {v_c:.1f} Rot {v_r:.1f} | C:{loss_contrast.item():.2f} | Div: {unique_count}/{BATCH_SIZE} Ent:{entropy:.2f} MaxP:{max_probs:.2f} | w_pot:{w_pot:.3f}"
            tqdm.write(log_msg)
            logger.info(log_msg)
            
            # Update Readout Prototypes (Fix Staleness)
            readout.update_prototypes()

        # Update LTM
        ltm.add(q_b.detach(), m_b.detach(), J_b.detach(), p_batch.detach())

        # Buffer Update
        if len(active_q) > 2:
             active_q.pop(0); active_m.pop(0); active_J.pop(0)
        active_q.append(q_b.detach())
        active_m.append(m_b.detach())
        active_J.append(J_b.detach())
        
        if step_cnt % args.probe_freq == 0:
            logger.info("Generating probe text...")
            probe_out = eval_probe(injector, entropy_stats, id_to_char, ff, device=args.device)
            tqdm.write(f"\n[Step {step_cnt}] Probe: {probe_out}")
            logger.info(f"[Step {step_cnt}] Probe: {probe_out}")
            
            # Teacher Forcing Probe
            tf_prompt = "自然语言处理是人工智能的核心技术" # A standard balanced sentence
            tf_out = eval_tf_probe_stats(injector, entropy_stats, id_to_char, char_to_id, ff, text=tf_prompt, device=args.device)
            tqdm.write(f"[Step {step_cnt}] TF Probe:\n{tf_out}\n")
            logger.info(f"[Step {step_cnt}] TF Probe:\n{tf_out}")
             
        if step_cnt % args.save_freq == 0:
            checkpoint = {
                'step': step_cnt,
                'injector_state': injector.state_dict(),
                'ff_state': ff.state_dict(),
                'ltm_state': ltm.state_dict(),
                'ltm_ptr': ltm.ptr,
                'ltm_size': ltm.size
            }
            # Numbered Checkpoint
            step_suffix = f"_step_{step_cnt}.pth"
            numbered_path = args.model_path.replace(".pth", step_suffix)
            torch.save(checkpoint, numbered_path)
            # Latest
            torch.save(checkpoint, args.model_path)
            logger.info(f"Saved checkpoint to {numbered_path}")
            
    # Final Save
    checkpoint = {
        'step': step_cnt,
        'injector_state': injector.state_dict(),
        'ff_state': ff.state_dict(),
        'ltm_state': ltm.state_dict(),
        'ltm_ptr': ltm.ptr,
        'ltm_size': ltm.size
    }
    torch.save(checkpoint, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--model_path", type=str, default="data/aurora_v3_1.pth")
    parser.add_argument("--probe_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=5000)
    
    args = parser.parse_args()
    if args.device == 'cpu' and torch.cuda.is_available(): args.device = 'cuda'
    
    train(args)

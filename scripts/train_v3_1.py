"""
Train CCD v3.1 Rheological Model.
=================================

Implements the Fast/Slow dynamics loop.
1. Fast: Interaction & Kinematics.
2. Slow: Plastic Flow of Parameters.

Ref: Chap 4 (Rheological Learning).
"""

import sys
import os
import torch
import json
import logging
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data.data_pipeline import CharStreamProcessor, GlobalEntropyStats
from aurora.model.injector import AuroraInjector
from aurora.engine.forces import ForceField
from aurora.engine.integrator import LieIntegrator
from aurora.engine.rheology import RheologicalOptimizer

# Config
DIM = 5
EMBED_DIM = 32
HIDDEN_DIM = 64
LR = 0.01
YIELD_STRESS = 0.001 # Start low to encourage flow
ELASTIC_K = 0.0001
STEPS_PER_INJECT = 5 # Short burst of integration per char?
# Actually, Injector adds particle. Integrator evolves WHOLE system.
# Continuous injection:
# t=0: Inject char 0. Integrate K steps.
# t=1: Inject char 1. Integrate K steps.
# ...

def train():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger("AuroraTrain")
    
    # 1. Data
    logger.info("Loading Entropy Stats...")
    stats_path = "data/entropy_stats.json"
    if not os.path.exists(stats_path):
        logger.error("Please run scripts/prepare_data.py first.")
        return
        
    entropy_stats = GlobalEntropyStats(stats_path)
    
    # Build Vocab from stats
    vocab_list = sorted(list(entropy_stats.stats.keys()))
    char_to_id = {c:i for i,c in enumerate(vocab_list)}
    VOCAB_SIZE = len(vocab_list)
    logger.info(f"Vocab Size: {VOCAB_SIZE}")
    
    # Stream
    stream = CharStreamProcessor("data/CLUE/CLUECorpusSmall.txt")
    
    # 2. Model
    injector = AuroraInjector(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, DIM)
    
    # Physics
    ff = ForceField(G=2.0, lambda_gauge=1.5, k_geo=0.2)
    integrator = LieIntegrator(ff)
    
    # Optimizer (Rheology)
    # We optimize Injector parameters
    rheo_opt = RheologicalOptimizer(injector.parameters(), lr=LR, yield_stress=YIELD_STRESS, elastic_k=ELASTIC_K)
    
    # 3. Training Loop
    # State Buffer (The "Mind")
    # Max capacity N=50? (Short term memory)
    # Start empty.
    
    # Note: We need Autograd through the physics?
    # Rheology uses Stress = grad(Loss).
    # Loss?
    # Axiom 4.2: Stress arises from "Conflict".
    # Conflict between "Intrinsic Psi" and "Extrinsic u".
    # L ~ || u - Psi ||^2 ? (Structural Yield)
    # Or L ~ Energy?
    # If Energy is minimized, system is happy.
    # So we minimize E_total?
    # Plus constraint: Particles must stay at their Entropy Radius.
    # L = E_total + lambda * || ||q|| - r ||^2.
    
    logger.info("Starting Training...")
    
    # Running components
    active_m = []
    active_q = []
    active_J = []
    active_p = []
    active_r_target = [] # Track target radii
    
    MAX_ATOMS = 20
    
    step_cnt = 0
    total_loss = 0
    
    # Generator for chars
    char_gen = stream.stream()
    
    try:
        pbar = tqdm(range(2000)) # Train for 2000 chars as test
        for _ in pbar:
            char = next(char_gen)
            if char not in char_to_id:
                continue
                
            cid = char_to_id[char]
            r_tgt = entropy_stats.get_radial_target(char)
            
            # 1. Injection
            # Create new atom
            cid_t = torch.tensor([[cid]], dtype=torch.long)
            r_t = torch.tensor([[r_tgt]], dtype=torch.float)
            
            # Compute 'Current' Particle (Attached graph)
            m_curr, q_rest_curr, J_curr, p_init_curr = injector(cid_t, r_t)
            
            q_c = q_rest_curr.view(1, DIM)
            # Add small noise to avoid overlapping particles (Singularity in Potential)
            # Ensure we stay in ball (norm < 1).
            noise = torch.randn_like(q_c) * 0.01
            q_c = q_c + noise
            
            # Safe Clamp in Poincare Ball
            qn = torch.norm(q_c, dim=-1, keepdim=True)
            max_r = 0.95 # Safe margin
            mask = qn > max_r
            q_c = torch.where(mask, q_c * (max_r / (qn + 1e-9)), q_c)

            m_c = m_curr.view(1, 1)
            J_c = J_curr.view(1, DIM, DIM)
            p_c = p_init_curr.view(1, DIM)
            
            # 2. Build System State for Physics
            # q_in = [History (Detached), Current (Attached)]
            if len(active_q) > 0:
                 # active_q holds detached tensors from previous steps
                 q_old = torch.cat(active_q, dim=0)
                 m_old = torch.cat(active_m, dim=0)
                 J_old = torch.cat(active_J, dim=0)
                 
                 q_in = torch.cat([q_old, q_c], dim=0)
                 m_in = torch.cat([m_old, m_c], dim=0)
                 J_in = torch.cat([J_old, J_c], dim=0)
            else:
                 q_in = q_c
                 m_in = m_c
                 J_in = J_c

                
            # 2. Physics Step (Fast Dynamics)
            # Run a few steps of integration to let it settle/interact
            # But Integrator is discrete.
            # For Loss, we can just look at Instantaneous Potential + Constraint.
            
            # Calculate Force/Potential
            # We want Injector to output q, J such that E is minimized?
            # And constraints satisfied.
            
            # ForceField compute
            # F, V = ff.compute(q_in...)
            # We need V (Total Potential Energy).
            # forces.py returns (F, V).
            
            # Need to patch ForceField to handle autodiff correctly without detaching inside?
            # `compute_forces` does `detach().requires_grad_(True)` if not set.
            # If q_in has grad (from Injector), we should NOT detach.
            # Let's modify `forces.py` or just rely on `compute_forces` behavior?
            # `compute_forces`: "if not q.requires_grad: ..."
            # So if it has grad, it proceeds. Good.
            
            _, E_pot = ff.compute_forces(q_in, m_in, J_in)
            # E_pot is a float (item()). We need the Tensor for loss!
            # `compute_forces` returns `V_total.item()`.
            # I need `V_total` tensor.
            # Need to modify `forces.py` or replicate logic? 
            # I will replicate logic here or fix forces.py.
            # Fix forces.py to return tensor is better.
            # BUT for now, let's just create a local V calculation for training.
            
            # Replicating V_total calculation to ensure graph connectivity
            # (See bound_state.detect logic, but with Autograd)
            
            loss = 0
            
            # A. Constraint Loss: r matched?
            # q_c should be at radius r_t
            r_curr = torch.norm(q_in[-1])
            # tanh dist? No, Poincare radius is Euclidean norm in Ball model
            loss_r = (r_curr - r_t.item())**2
            
            # B. Energy Loss (Thermodynamic)
            # We want the particle to find a "Low Energy" slot.
            # So minimize V_potential?
            # Yes.
            
            # Pairwise potentials
            # Warning: High computation if naive.
            # Just compute interactions of LAST particle with OTHERS.
            if len(active_q) > 1:
                # q_curr vs q_old
                # dist
                # ...
                pass
            
            # Total Loss
            # E_pot is now a Tensor.
            # L = L_constraint + lambda * E_potential
            loss = loss_r # + 0.01 * E_pot 
            
            if torch.isnan(loss) or torch.isnan(E_pot):
                print(f"NaN Detected! L_r: {loss_r.item()} E_pot: {E_pot.item()}")
                print(f"q_in norm: {torch.norm(q_in, dim=-1)}")
                break

            
            # 3. Rheology (Slow Dynamics)
            # optimizer.zero_grad() # Rheology accumulates?
            # RheologyOptimizer doesn't have zero_grad, it acts on .grad directly.
            # But we need to clear previous grads manually?
            injector.zero_grad()
            
            loss.backward()
            
            # Step
            rheo_opt.step()
            
            # Update State Buffer with the Result (Simulated)
            # Actually, we should integrate position for next step.
            # state = integrator.step(...)
            # But here we just injected.
            
            step_cnt += 1
            total_loss += loss.item()
            
            pbar.set_description(f"L:{loss.item():.4f} r_tgt:{r_t.item():.2f}")
            
            # Update History (Detached)
            if len(active_q) >= MAX_ATOMS:
                active_q.pop(0)
                active_m.pop(0)
                active_J.pop(0)
                active_p.pop(0)
                
            active_q.append(q_c.detach())
            active_m.append(m_c.detach())
            active_J.append(J_c.detach())
            active_p.append(p_c.detach())
            
            if step_cnt % 100 == 0:
                # Save Checkpoint
                pass
                
    except StopIteration:
        pass
        
    logger.info(f"Training finished. Avg Loss: {total_loss/step_cnt:.4f}")
    
    # Save Model
    torch.save(injector.state_dict(), "data/aurora_v3_1.pth")

if __name__ == "__main__":
    train()

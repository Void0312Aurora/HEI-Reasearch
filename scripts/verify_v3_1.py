"""
Verify CCD v3.1 Engine.
=======================

Tests the full pipeline from Injector to Integrator.
"""

import sys
import os
import torch
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.model.injector import AuroraInjector
from aurora.engine.forces import ForceField
from aurora.engine.integrator import LieIntegrator
from aurora.engine.bound_state import MoleculeDetector

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AuroraV3.1")
    
    logger.info("Initializing Engine components...")
    
    # 1. Config
    DIM = 5
    VOCAB_SIZE = 100
    EMBED = 16
    HIDDEN = 32
    
    # 2. Injector
    injector = AuroraInjector(VOCAB_SIZE, EMBED, HIDDEN, DIM)
    
    # 3. Dummy Input
    chars = torch.tensor([[10, 20, 30]], dtype=torch.long) # "A B C"
    entropy = torch.tensor([[0.1, 0.5, 0.9]], dtype=torch.float) # r targets
    
    logger.info("Running Injector...")
    mass, q0, J0, p0 = injector(chars, entropy)
    logger.info(f"Injection Shapes: q={q0.shape}, m={mass.shape}")
    
    # Flatten Batch and Seq for Physics Simulation
    # Assume all particles coexist in the Ball
    q_sys = q0.view(-1, DIM)
    p_sys = p0.view(-1, DIM)
    m_sys = mass.view(-1, 1)
    J_sys = J0.view(-1, DIM, DIM)
    
    state = {'q': q_sys, 'p': p_sys, 'm': m_sys, 'J': J_sys}
    
    # 4. Integrator
    ff = ForceField(G=1.0, lambda_gauge=1.0, k_geo=0.1)
    integrator = LieIntegrator(ff)
    
    logger.info("Starting Simulation Loop (10 steps)...")
    for t in range(10):
        state = integrator.step(state, dt=0.01)
        q_norm = torch.norm(state['q'], dim=-1).mean().item()
        E = state['E']
        logger.info(f"Step {t}: E={E:.4f}, <|q|>={q_norm:.4f}")
        
        # Check Safety
        if q_norm >= 1.0:
            logger.error("Poincare Constraint Violated!")
            sys.exit(1)
            
    logger.info("Simulation Complete. Engine Operational.")
    
    # 5. Bound State
    monitor = MoleculeDetector(ff)
    bonds = monitor.detect(state)
    logger.info(f"Detected Bonds: {bonds}")

if __name__ == "__main__":
    main()

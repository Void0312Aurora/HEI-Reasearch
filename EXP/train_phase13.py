import torch
import torch.optim as optim
import argparse
import json
import logging
from typing import Dict, Any

from he_core.entity_v4 import UnifiedGeometricEntity
from EXP.run_phase12 import run_experiment, NumpyEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainPhase13")

def train_phase13(config: Dict[str, Any]):
    # 1. Init Entity
    entity = UnifiedGeometricEntity(config)
    entity.train() # Enable gradients
    
    # 2. Optimizer Selection
    params = []
    if not config.get('freeze_coupling', False):
        params += list(entity.generator.coupling.parameters())
    if not config.get('freeze_router', False):
        params += list(entity.atlas.router.parameters())
        
    if not params:
        logger.warning("No parameters to optimize!")
        return "NO_PARAMS"
        
    optimizer = optim.Adam(params, lr=config.get('lr', 0.01))
    
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 16)
    steps = config.get('steps', 50)
    
    # 3. Training Loop
    logger.info("Starting Training...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Batch Simulation
        # Since state is stateful, we simulate 'batch_size' independent trajectories 
        # or just 1 trajectory and backprop through time (BPTT)?
        # For simple structure learning, short bursts BPTT is fine.
        
        # Reset State batch? Entity v0.4 uses ContactState which supports batch?
        # UnifiedGeometricEntity initializes state as (1, dim).
        # We need to hack it to support batch if batch_size > 1.
        # But for now let's stick to batch=1 (Single Trajectory Learning) to avoid batch dim complexity in current v0.3 Core.
        
        entity.reset() 
        # Manually enable gradients on q/p if we wanted to differentiate wrt initial state.
        # But we learn params.
        
        total_loss = 0.0
        
        # BPTT
        trajectory_loss = 0.0
        
        curr_state_flat = entity.state.flat.detach() # Start fresh graph
        
        for t in range(steps):
             # Input: Driven or Null
             u_ext = torch.randn(1, entity.dim_q) * 0.1 # Replay Noise
             
             # Forward Tensor
             out = entity.forward_tensor(curr_state_flat, u_ext)
             
             # --- LOSSES (The Evidence Constraints) ---
             
             # [Task Loss]: Reconstruction? Or just Activity Minimization (Energy Efficiency)?
             # Let's say: Maximize 'Chart Entropy' (Diversity) + Minimize Action (Efficiency)
             loss_efficiency = out['action'].norm() * 0.1
             loss_entropy = -out['chart_weights'].max() # Crude entropy proxy (minimize max prob)
             
             # [Evidence: A3 Consistency]
             # Drift: H_new - H_old should be <= 0 (Dissipative)
             # H_new = out['H_val']
             # But H_old? We need previous H.
             # Approximate: Minimize H.
             loss_drift = out['H_val'] * 1.0 # Minimize Energy
             
             # [Evidence: Protocol 5 Port Gain]
             # Penalize amplification ||action|| / ||u||
             # u is 0.1 approx. action is out['action'].
             # loss_gain = ||action|| - Gain*||u||
             # Just minimizing action covers this.
             
             step_loss = loss_efficiency + loss_drift + loss_entropy * 0.1
             trajectory_loss += step_loss
             
             # Teacher Forcing or Auto-regressive?
             # Auto-regressive: next state is input to next step.
             curr_state_flat = out['next_state_flat']
             
        trajectory_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss {trajectory_loss.item():.4f}")
            
            # --- EVALUATION (The Gates) ---
            # Run run_phase12 logic (Numpy/NoGrad)
            # Create a CONFIG for run_experiment that matches current entity?
            # Or pass entity? run_experiment rebuilds entity currently!
            # We need to adapt run_experiment to accept existing entity 
            # Or we just save params and reload?
            # Easiest: Verify logic inside here using Entity directly.
            
            # Quick Gate Check: Spectral Gap
            # (Just print for monitoring, run_phase12 is for final report)
            pass

    logger.info("Training Complete.")
    
    # Save Model
    torch.save(entity.state_dict(), "entity_v0.4.pth")
    return "SUCCESS"

if __name__ == "__main__":
    conf = {'dim_q': 2, 'learnable_coupling': True, 'num_charts': 2, 'epochs': 20}
    train_phase13(conf)

import torch
import torch.optim as optim
import argparse
import json
import logging
from typing import Dict, Any

# Debug Print imports
print("DEBUG: LOADING EXP.train_phase13 START", flush=True)

from he_core.entity_v4 import UnifiedGeometricEntity
from EXP.run_phase12 import run_experiment, NumpyEncoder

print("DEBUG: EXP.train_phase13 IMPORTS DONE", flush=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainPhase13")

def train_phase13(config: Dict[str, Any]):
    print("DEBUG: TRACE - Entered train_phase13", flush=True)
    
    # 1. Init Entity
    try:
        entity = UnifiedGeometricEntity(config)
        print("DEBUG: TRACE - Entity Initialized", flush=True)
        entity.train() # Enable gradients
    except Exception as e:
        print(f"DEBUG: ERROR - Entity Init Failed: {e}", flush=True)
        return "FAIL_INIT"

    # 2. Optimizer Selection
    print("DEBUG: TRACE - Selecting Params", flush=True)
    params = []
    
    try:
        # Check Generator Coupling
        if not config.get('freeze_coupling', False):
            print("DEBUG: TRACE - Getting Coupling Params", flush=True)
            if hasattr(entity.generator.coupling, 'parameters'):
                c_params = list(entity.generator.coupling.parameters())
                print(f"DEBUG: TRACE - Got Coupling Params: {len(c_params)}", flush=True)
                params += c_params
            else:
                 print("DEBUG: WARNING - No parameters() on coupling", flush=True)

        # Check Router
        if not config.get('freeze_router', False):
            print("DEBUG: TRACE - Getting Router Params", flush=True)
            if hasattr(entity.atlas.router, 'parameters'):
                r_params = list(entity.atlas.router.parameters())
                print(f"DEBUG: TRACE - Got Router Params: {len(r_params)}", flush=True)
                params += r_params
    except Exception as e:
        print(f"DEBUG: ERROR - Param Select Failed: {e}", flush=True)
        return "FAIL_PARAMS"

    print(f"DEBUG: TRACE - Params Count: {len(params)}", flush=True)
    
    if not params:
        logger.warning("No parameters to optimize!")
        return "NO_PARAMS"
        
    optimizer = optim.Adam(params, lr=config.get('lr', 0.01))
    
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 16)
    steps = config.get('steps', 50)
    
    # 3. Training Loop
    logger.info("Starting Training...")
    print("DEBUG: TRACE - Starting Training Loop", flush=True)
    
    try:
        for epoch in range(epochs):
            # print(f"DEBUG: TRACE - Epoch {epoch}", flush=True) # Reduce spam
            optimizer.zero_grad()
            
            # Batch Simulation (Simplified)
            entity.reset() 
            
            total_loss = 0.0
            trajectory_loss = 0.0
            
            curr_state_flat = entity.state.flat.detach() # Start fresh graph
            
            for t in range(steps):
                 # Input: Driven or Null
                 u_ext = torch.randn(1, entity.dim_q) * 0.1
                 
                 # Forward Tensor
                 out = entity.forward_tensor(curr_state_flat, u_ext)
                 
                 # --- LOSSES (The Evidence Constraints) ---
                 
                 # Metric 1: Action Magnitude (Efficiency)
                 action_norm = out['action'].norm()
                 loss_efficiency = action_norm * 0.1
                 
                 # Metric 2: Coverage/Entropy (Maximize Diversity)
                 prob = out['chart_weights']
                 entropy = -(prob * (prob + 1e-9).log()).sum()
                 loss_entropy = -entropy * 0.05
                 
                 # Metric 3: Drift (Dissipativity) [A3]
                 loss_drift = torch.relu(out['H_val']) * 1.0 
                 
                 # Metric 4: Port Gain [P5] (Stability)
                 u_norm = u_ext.norm()
                 gain = action_norm / (u_norm + 1e-6)
                 
                 threshold_gain = 2.0
                 loss_gain = torch.relu(gain - threshold_gain) * 10.0
                 
                 # Total Step Loss
                 step_loss = loss_efficiency + loss_drift + loss_entropy + loss_gain
                 trajectory_loss += step_loss
                 
                 # Critical Check
                 if gain.item() > 10.0:
                     logger.error(f"Critical Failure: Gain Explosion ({gain.item()}) at step {t}")
                     return "FAIL_EXPLOSION"
                 
                 curr_state_flat = out['next_state_flat']
                 
            trajectory_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss {trajectory_loss.item():.4f}")

        logger.info("Training Complete.")
        
        # Save Model
        torch.save(entity.state_dict(), "entity_v0.4.pth")
        
    except Exception as e:
        print(f"DEBUG: ERROR - Training Loop Failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return "FAIL_LOOP"

    return "SUCCESS"

if __name__ == "__main__":
    conf = {'dim_q': 2, 'learnable_coupling': True, 'num_charts': 2, 'epochs': 20}
    train_phase13(conf)

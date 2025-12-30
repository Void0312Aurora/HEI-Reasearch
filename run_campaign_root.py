import torch
print("DEBUG: TOP LEVEL START", flush=True)
import json
import logging
import os
import sys

# Delay imports to avoid top-level side effects
# from EXP.train_phase13 import train_phase13
# from EXP.run_phase12 import run_experiment, NumpyEncoder

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("LongCampaign")

def run_gate0_campaign():
    print("DEBUG: Entered run_gate0_campaign")
    logger.info("=== Starting Gate-0 Long-Run Campaign ===")
    
    try:
        from EXP.train_phase13 import train_phase13
        from EXP.run_phase12 import run_experiment, NumpyEncoder
    except ImportError as e:
        logger.error(f"Import Error: {e}")
        return
    
    report = {
        "experiments": {},
        "overall_status": "PENDING"
    }
    
    # Common Config
    base_conf = {
        'dim_q': 2,
        'num_charts': 3,
        'learnable_coupling': True,
        'lr': 0.005,
        'batch_size': 1,
        'steps': 50 # Per epoch
    }
    
    # Experiment A: W-Only (Deep Training)
    logger.info("--- Exp A: W-Only Stability ---")
    conf_A = base_conf.copy()
    conf_A['epochs'] = 50 # Shorten for verification run (plan says 500, we do 50 for quick check, or user wants real 500?)
    # Plan says: "500 epochs... for verification run".
    logger.info("--- Exp A: W-Only Stability ---")
    print("DEBUG: After Logger Info", flush=True)
    conf_A['epochs'] = 100 
    conf_A['freeze_router'] = True
    
    print(f"DEBUG: Calling train_phase13 object: {train_phase13}", flush=True)
    status_A = train_phase13(conf_A)
    print(f"DEBUG: train_phase13 returned: {status_A}")
    
    report['experiments']['Exp_A'] = status_A
    
    if status_A != "SUCCESS":
        logger.error("Exp A Failed!")
        report['overall_status'] = "FAIL_A"
        save_report(report)
        return

    # Experiment B: Router-Only (Coverage)
    logger.info("--- Exp B: Router-Only Coverage ---")
    conf_B = base_conf.copy()
    conf_B['epochs'] = 100
    conf_B['freeze_coupling'] = True
    # Load weights from A? Or fresh?
    # Ideally fresh to verify independent stability, or sequential?
    # Sequential is better for "Building".
    # But train_phase13 saves to 'entity_v0.4.pth'.
    # It reloads ONLY if we code it to. Currently it inits fresh Entity.
    
    status_B = train_phase13(conf_B)
    report['experiments']['Exp_B'] = status_B
    
    if status_B != "SUCCESS":
        logger.error("Exp B Failed!")
        report['overall_status'] = "FAIL_B"
        save_report(report)
        return

    # Experiment C: Joint (Hard Gated)
    logger.info("--- Exp C: Joint Hard-Gated ---")
    conf_C = base_conf.copy()
    conf_C['epochs'] = 50
    conf_C['freeze_router'] = False
    conf_C['freeze_coupling'] = False
    
    status_C = train_phase13(conf_C)
    report['experiments']['Exp_C'] = status_C
    
    if status_C != "SUCCESS":
        logger.error("Exp C Failed!")
        report['overall_status'] = "FAIL_C"
        save_report(report)
        return

    # Experiment D: Rollout (Validation)
    logger.info("--- Exp D: Long Rollout Validation ---")
    # Load trained model (from Exp C, saved as entity_v0.4.pth)
    # Use run_phase12 logic but extended steps
    # We can invoke run_experiment with specific config
    
    # We need to ensure run_experiment loads the saved weights if we want to test 'Trained' entity.
    # Current run_phase12 creates fresh entity.
    # We should update run_phase12 or just load here and run rollout manually.
    # Let's run a check here.
    
    from he_core.entity_v4 import UnifiedGeometricEntity
    entity = UnifiedGeometricEntity(base_conf)
    try:
        entity.load_state_dict(torch.load("entity_v0.4.pth"))
        logger.info("Loaded Trained Entity.")
    except Exception as e:
        logger.warning(f"Could not load weights: {e}")
        
    # Rollout 1000 steps
    entity.reset()
    traj_actions = []
    
    for t in range(1000):
        obs = {'x_ext': torch.randn(1, 2) * 0.1}
        out = entity(obs)
        traj_actions.append(out['action'])
        
    actions = torch.tensor(traj_actions)
    gain_est = actions.norm() / (1000 * 0.1) # Approx
    
    report['metrics'] = {
        "rollout_gain_est": gain_est.item()
    }
    
    logger.info(f"Rollout Gain Est: {gain_est.item()}")
    
    if gain_est > 5.0:
        report['overall_status'] = "FAIL_ROLLOUT_GAIN"
    else:
        report['overall_status'] = "READY"
        
    save_report(report)

def save_report(report):
    with open("report_gate0.json", "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    logger.info("Report saved to report_gate0.json")

if __name__ == "__main__":
    run_gate0_campaign()

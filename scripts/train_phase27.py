"""
Phase 27: Large Scale Joint Training.
=====================================

Runs Joint Training (Semantics + Language) on full Wikipedia Dump.
Designed for long-running execution.

Usage:
    python scripts/train_phase27.py <checkpoint> <wiki_dump> --save_every 1000
"""

import sys
import os
import torch
import pickle
import argparse
import time
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.data import AuroraDataset
from aurora.gauge import GaugeField, NeuralBackend
from aurora.training.joint_trainer import JointTrainer
from aurora.ingest.wiki_dump import WikiDumpIngestor

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_phase27.log"),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to starting checkpoint (e.g. checkpoint_cycle_3.pkl)")
    parser.add_argument("wiki_dump", type=str, help="Path to Wikipedia raw text dump")
    parser.add_argument("--save_dir", type=str, default="checkpoints_p27")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N batches")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr_J", type=float, default=0.0001, help="Lower learning rate for J (Stability)")
    parser.add_argument("--lr_U", type=float, default=0.001, help="Standard learning rate for U")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 1. Load Checkpoint
    logging.info("Loading Checkpoint...")
    import io
    class DeviceUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=device)
            return super().find_class(module, name)
            
    with open(args.checkpoint, 'rb') as f:
        ckpt = DeviceUnpickler(f).load()
    
    # 2. Load Dataset FIRST (to get correct num_nodes)
    ds = AuroraDataset("cilin", limit=12000)
    
    # 3. Generate or Load x, J (Must match ds.num_nodes)
    ckpt_num_nodes = ckpt['x'].shape[0] if 'x' in ckpt else 0
    current_num_nodes = ds.num_nodes
    
    if ckpt_num_nodes == current_num_nodes:
        logging.info(f"Checkpoint matches dataset ({current_num_nodes} nodes). Loading x, J.")
        x = torch.tensor(ckpt['x'], device=device)
        J = torch.tensor(ckpt['J'], device=device)
    else:
        logging.warning(f"Checkpoint size mismatch ({ckpt_num_nodes} vs {current_num_nodes}). Generating fresh x, J.")
        x, J = ds.generate_initial_conditions(dim=5, logical_dim=3, device=device)
    
    if not J.requires_grad: J.requires_grad = True
    
    # 4. Init Gauge Field (Re-init or Load)
    logging.info("Initializing Gauge Field...")
    gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=device), 3, group='SO', backend_type='neural', input_dim=5).to(device)
    
    gauge_field.backend = NeuralBackend(input_dim=5, logical_dim=3, num_relations=200, relation_dim=8).to(device)
    
    if 'gauge_field' in ckpt:
        logging.info("Loading Gauge Weights from Checkpoint (Partial)...")
        gf_state = ckpt['gauge_field']
        model_dict = gauge_field.state_dict()
        pretrained_dict = {k: v for k, v in gf_state.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        gauge_field.load_state_dict(model_dict)
        logging.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers.")
    
    # 5. Ingestor & Trainer
    ingestor = WikiDumpIngestor(ds, neg_ratio=4)
    struct_edges = ds.edges_struct
    
    trainer = JointTrainer(gauge_field, x, J, lr_J=args.lr_J, lr_U=args.lr_U, device=device)
    
    # 5. Training Loop
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    logging.info(f"Starting Training on {args.wiki_dump}...")
    batch_idx = 0
    start_time = time.time()
    
    try:
        from tqdm import tqdm
        
        pbar = tqdm(ingestor.stream_dump(args.wiki_dump, batch_size=args.batch_size), desc="Training")
        
        for batch in pbar:
            loss_J, loss_U = trainer.train_step(struct_edges, batch)
            
            batch_idx += 1
            
            # exponential moving average for display
            if batch_idx == 1:
                avg_J, avg_U = loss_J, loss_U
            else:
                avg_J = 0.9 * avg_J + 0.1 * loss_J
                avg_U = 0.9 * avg_U + 0.1 * loss_U
                
            pbar.set_description(f"Loss J:{avg_J:.4f} U:{avg_U:.4f}")
            
            if batch_idx % 100 == 0:
                 # Still log to file
                 logging.info(f"Batch {batch_idx}: Loss_J={loss_J:.4f} Loss_U={loss_U:.4f}")
                
            if batch_idx % args.save_every == 0:
                save_path = os.path.join(args.save_dir, f"checkpoint_step_{batch_idx}.pkl")
                logging.info(f"Saving checkpoint to {save_path}...")
                state = {
                    'x': x.cpu().numpy(),
                    'J': trainer.J.detach().cpu().numpy(),
                    'gauge_field': gauge_field.state_dict(),
                    'step': batch_idx
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(state, f)
                    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving emergency checkpoint...")
        save_path = os.path.join(args.save_dir, "checkpoint_interrupted.pkl")
        state = {
            'x': x.cpu().numpy(),
            'J': trainer.J.detach().cpu().numpy(),
            'gauge_field': gauge_field.state_dict(),
            'step': batch_idx
        }
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
            
    logging.info("Training Complete.")

if __name__ == "__main__":
    main()

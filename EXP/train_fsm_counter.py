"""
Phase 23.2: FSM Task - 3-State Counter (mod 3).
Task: Count number of +1 pulses in sequence, output (count mod 3).
- Input: Sequence of pulses (0 or +1) at random times.
- Output: Class 0, 1, or 2 at final time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.adaptive_generator import AdaptiveDissipativeGenerator
from he_core.port_generator import PortCoupledGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== DATA ==========
def generate_counter_data(batch_size, dim_u, num_pulses=5, seq_len=30, pulse_mag=10.0, seed=None):
    """
    Generate sequences with random pulse positions.
    Label = (number of pulses) mod 3.
    Vectorized version for speed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Random number of pulses: (B,) in [0, num_pulses]
    counts = torch.randint(0, num_pulses + 1, (batch_size,), device=DEVICE)
    
    # Create inputs container
    inputs = torch.zeros(batch_size, seq_len, dim_u, device=DEVICE)
    
    # We need to place 'counts[b]' pulses at random positions for each b.
    # Vectorized approach:
    # 1. Generate random scores for all positions (batch_size, seq_len-1) (exclude t=0)
    scores = torch.rand(batch_size, seq_len - 1, device=DEVICE)
    
    # 2. Sort scores to get indices of top-k positions
    # But k varies per batch.
    # Alternative: Use a mask.
    # Create rank tensor: (B, seq_len-1) with values 1..seq_len-1
    # Actually, simpler:
    # Just sort indices by score? No, we need simply 'counts[b]' top positions.
    
    # Sort scores descending: indices corresponding to top values are our random positions
    _, sorted_indices = scores.sort(dim=1, descending=True)
    # sorted_indices: (B, seq_len-1)
    
    # We want to select the first counts[b] indices from sorted_indices.
    # Create a mask of shape (B, seq_len-1) where column j < counts[b]
    range_tensor = torch.arange(seq_len - 1, device=DEVICE).unsqueeze(0) # (1, seq_len-1)
    counts_col = counts.unsqueeze(1) # (B, 1)
    mask = range_tensor < counts_col # (B, seq_len-1)
    
    # Get the positions that are active
    # active_indices = sorted_indices[mask] won't work easily to scatter back.
    
    # Better: Scatter 1s into a temp tensor at sorted_indices, but masked?
    # No. 
    # Let's use scatter_.
    # Target tensor: temp (B, seq_len-1).
    # We want to set temp[b, sorted_indices[b, :counts[b]]] = 1.
    # But counts varies.
    
    # Trick:
    # 1. Create a "selection mask" in sorted order (111000...) using range comparison.
    selection_mask = (torch.arange(seq_len - 1, device=DEVICE).unsqueeze(0) < counts.unsqueeze(1)).float()
    
    # 2. We want to map this back to original temporal order.
    # 'sorted_indices' tells us: "Index X is Rank K".
    # No, active_indices = sorted_indices tells us "Rank K corresponds to Index X".
    # So if we scatter 'selection_mask' (values at Rank K) into 'Index X', we get the temporal sequence.
    
    # We need scatter dimension 1.
    # target.scatter_(1, index, src)
    # target: (B, seq_len-1) zeros
    # index: sorted_indices
    # src: selection_mask (ordered by rank: 1s first, then 0s)
    
    pulse_map = torch.zeros(batch_size, seq_len - 1, device=DEVICE)
    pulse_map.scatter_(1, sorted_indices, selection_mask)
    
    # Now copy to inputs (offset by 1 because we skipped t=0)
    inputs[:, 1:, 0] = pulse_map * pulse_mag
    
    labels = (counts % 3).long()
    
    return inputs, labels

# ========== MODEL ==========
def create_model(dim_q, dim_u):
    config = {'dim_q': dim_q, 'dim_u': dim_u, 'num_charts': 1, 'learnable_coupling': True, 'use_port_interface': False}
    entity = UnifiedGeometricEntity(config)
    net_V = nn.Sequential(nn.Linear(dim_q, 128), nn.Tanh(), nn.Linear(128, 1))
    for p in net_V.parameters(): nn.init.normal_(p, mean=0.0, std=1e-5)
    adaptive = AdaptiveDissipativeGenerator(dim_q, net_V=net_V)
    entity.generator = PortCoupledGenerator(adaptive, dim_u, True, 1)
    entity.internal_gen = adaptive
    return entity.to(DEVICE)

# ========== TRAIN ==========
def train_and_evaluate(entity, classifier, train_fn, val_fn, args):
    classifier = classifier.to(DEVICE)
    params = list(entity.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    
    # Optimize Rollout
    # torch.compile fails with double backward (due to ContactDynamics autograd.grad)
    # So we use standard Python loop.
    # Batch size 4096 ensures GPU saturation anyway.
    
    best_val = 0.0
    
    for epoch in range(args.epochs):
        # Generate new training data each epoch
        train_u, train_y = train_fn()
        
        entity.train()
        optimizer.zero_grad()
        
        curr = torch.zeros(train_u.shape[0], 2*args.dim_q + 1, device=DEVICE)
        
        # Standard Loop
        for t in range(train_u.shape[1]):
            out = entity.forward_tensor(curr, train_u[:, t, :], args.dt)
            curr = out['next_state_flat']
        
        logits = classifier(curr[:, :args.dim_q])
        loss = nn.functional.cross_entropy(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        train_acc = (logits.argmax(dim=1) == train_y).float().mean().item()
        
        # Validation
        entity.eval()
        val_u, val_y = val_fn()
        curr = torch.zeros(val_u.shape[0], 2*args.dim_q + 1, device=DEVICE)
        for t in range(val_u.shape[1]):
            out = entity.forward_tensor(curr, val_u[:, t, :], args.dt)
            curr = out['next_state_flat'].detach()
        
        val_logits = classifier(curr[:, :args.dim_q])
        val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()
        
        if val_acc > best_val:
            best_val = val_acc
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Train={train_acc:.2f}, Val={val_acc:.2f}")
        
        if val_acc > 0.95:
            print(f"  Early stop at epoch {epoch}: Val={val_acc:.2f}")
            break
    
    return best_val

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32768)
    parser.add_argument('--dim_q', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    
    dim_u = args.dim_q
    
    print("=" * 60)
    print(f"Phase 23.2: FSM Counter (mod 3) (Device: {DEVICE})")
    print("=" * 60)
    
    # Data generators
    train_fn = lambda: generate_counter_data(args.batch_size, dim_u, num_pulses=5, seed=None)
    val_fn = lambda: generate_counter_data(256, dim_u, num_pulses=5, seed=999)
    
    # Model
    entity = create_model(args.dim_q, dim_u)
    classifier = nn.Sequential(
        nn.Linear(args.dim_q, 64), nn.Tanh(), nn.Linear(64, 3)
    )
    
    # Train
    best_val = train_and_evaluate(entity, classifier, train_fn, val_fn, args)
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT: Best Val Acc = {best_val:.2f}")
    print("=" * 60)
    
    if best_val > 0.8:
        print("  >> SUCCESS: FSM Counter Learned!")
        torch.save(entity.state_dict(), 'phase23_fsm_entity.pth')
    else:
        print("  >> FAILURE: FSM Counter Not Learned.")

if __name__ == "__main__":
    main()

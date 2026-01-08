"""
Stage 0: Existence Training (Level 0)

Goal: Train the SoulEntity to exist stably in the absence of external input.
This validates Axiom A4 (Identity Continuity) and A2 (Offline Cognition).

The dynamics should:
1. Not collapse to the origin (trivial fixed point).
2. Not diverge to infinity (instability).
3. Maintain a bounded, non-trivial trajectory (limit cycle or strange attractor).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
from dataclasses import dataclass

# Ensure HEI is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from curriculum.common.base_trainer import BaseCurriculumTrainer, CurriculumConfig
from he_core.state import ContactState

@dataclass
class Stage0Config(CurriculumConfig):
    # Stage specific params
    evolution_steps: int = 50  # Number of offline steps per batch
    
    # Loss weights
    lambda_barrier_min: float = 1.0  # Penalty for collapsing to origin
    lambda_barrier_max: float = 1.0  # Penalty for diverging
    lambda_lyapunov: float = 0.1     # Penalty for F increase
    
    # Constraints
    radius_min: float = 0.5
    radius_max: float = 5.0
    
    # Override defaults
    num_charts: int = 1  # Start with single chart
    dim_q: int = 32      # Start small
    integrator_method: str = "semi"  # Vector-space semi-implicit (GPU-friendly, stable)

class Stage0Trainer(BaseCurriculumTrainer):
    def __init__(self, config: Stage0Config):
        super().__init__(config)
        self.config = config
        
    def compute_existence_loss(self, 
                             states: list[ContactState], 
                             free_energies: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute loss for existence training.
        
        L = L_barrier + L_lyapunov
        """
        loss_barrier = torch.tensor(0.0, device=self.config.device)
        loss_lyapunov = torch.tensor(0.0, device=self.config.device)
        
        # Collect all q magnitudes
        q_norms = []
        for s in states:
            q_norms.append(s.q.norm(dim=1)) # [B]
        q_norms = torch.stack(q_norms, dim=1) # [B, T]
        
        # 1. Barrier Loss: Keep q in [R_min, R_max]
        # Penalize if q < R_min
        under_min = torch.relu(self.config.radius_min - q_norms)
        # Penalize if q > R_max
        over_max = torch.relu(q_norms - self.config.radius_max)
        
        loss_barrier = (under_min.mean() * self.config.lambda_barrier_min + 
                       over_max.mean() * self.config.lambda_barrier_max)
        
        # 2. Lyapunov Loss: F should generally decrease or stay stable
        # We penalize positive changes in F (dF > 0)
        # F_diff = F_{t+1} - F_t
        # Note: free_energies are scalars (mean over batch), so stack gives [T]
        F_seq = torch.stack(free_energies) # [T]
        F_diff = F_seq[1:] - F_seq[:-1]
        # Only penalize increases (allow decrease)
        loss_lyapunov = torch.relu(F_diff).mean() * self.config.lambda_lyapunov
        
        total_loss = loss_barrier + loss_lyapunov
        
        # Calculate motion (average step size)
        q_seq = torch.stack([s.q for s in states], dim=1) # [B, T, D]
        motion = (q_seq[:, 1:] - q_seq[:, :-1]).norm(dim=2).mean()
        
        return {
            "loss": total_loss,
            "loss_barrier": loss_barrier,
            "loss_lyapunov": loss_lyapunov,
            "mean_radius": q_norms.mean(),
            "max_radius": q_norms.max(),
            "min_radius": q_norms.min(),
            "motion": motion
        }

    def train_loop(self):
        optimizer = optim.AdamW(self.entity.parameters(), lr=self.config.lr)
        
        print(f"Starting Stage 0 Training on {self.config.device}...")
        print(f"Config: {self.config}")
        
        start_time = time.time()
        
        for step in range(1, self.config.steps + 1):
            optimizer.zero_grad()
            
            # 1. Initialize random state
            self.entity.reset(self.config.batch_size, self.config.device)
            # Start within valid range to give it a chance
            with torch.no_grad():
                r = torch.rand(self.config.batch_size, 1, device=self.config.device) * \
                    (self.config.radius_max - self.config.radius_min) + self.config.radius_min
                dir = torch.randn(self.config.batch_size, self.config.dim_q, device=self.config.device)
                dir = dir / (dir.norm(dim=1, keepdim=True) + 1e-6)
                self.entity.state.q = dir * r
                self.entity.state.p = torch.randn_like(self.entity.state.p) * 0.1
            
            # 2. Offline Evolution
            states_history = []
            fe_history = []
            
            # Initial state
            states_history.append(self.entity.state.clone()) # Clone to keep graph? No, state is updated in place but we need snapshots
            # Actually, entity.state is updated. We need to store copies or use functional steps.
            # entity.step updates self.entity.state in place.
            # For BPTT, we need the graph to be preserved.
            # The standard entity.step keeps the graph.
            
            # We need to be careful about memory. 50 steps is fine.
            
            # Record initial F
            fe_history.append(self.entity.compute_free_energy(self.entity.state))
            
            for _ in range(self.config.evolution_steps):
                # Pure offline step (no input)
                out = self.entity.step(u_dict={}, dt=self.config.dt)
                
                # Store for loss
                # Note: We need to store the state object, but be careful not to break graph
                # The state object created in step() has the graph history.
                states_history.append(self.entity.state) 
                fe_history.append(out['free_energy'])
            
            # 3. Compute Loss
            metrics = self.compute_existence_loss(states_history, fe_history)
            loss = metrics["loss"]
            
            # 4. Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.entity.parameters(), 1.0)
            optimizer.step()
            
            # 5. Logging
            if step % self.config.log_every == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}: Loss={loss.item():.4f} "
                      f"Barrier={metrics['loss_barrier'].item():.4f} "
                      f"Lyapunov={metrics['loss_lyapunov'].item():.4f} "
                      f"R_mean={metrics['mean_radius'].item():.2f} "
                      f"[{metrics['min_radius'].item():.2f}, {metrics['max_radius'].item():.2f}] "
                      f"Motion={metrics['motion'].item():.4f} "
                      f"Time={elapsed:.1f}s")
                
        # Save final model
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_path = os.path.join(self.config.save_dir, "stage0_final.pt")
        self.save(save_path)
        print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    config = Stage0Config()
    trainer = Stage0Trainer(config)
    trainer.train_loop()

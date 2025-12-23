"""
Interaction Engine for Aurora Base.
====================================

Implements the "Dialogue as Trajectory" paradigm:
- Input = Impulse
- Thinking = Coasting
- Output = Collapse (Integrate-and-Fire)
"""

import torch
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from .local_force_field import LocalForceField
from .concept_mapper import ConceptMapper


@dataclass
class CursorState:
    """State of the Mind Cursor."""
    q: torch.Tensor  # Position (dim,)
    p: torch.Tensor  # Momentum (dim,)
    step: int = 0
    activation_buffer: List[int] = field(default_factory=list)
    refractory_map: Dict[int, int] = field(default_factory=dict)  # id -> release_step


class InteractionEngine:
    """
    Simulates Mind Cursor dynamics on fixed Aurora Base background.
    
    The cursor flies through the 100k-particle semantic space,
    activating concepts it passes near.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        gamma: float = 0.5,  # Damping
        activation_threshold: float = 0.5,  # Min distance to activate
        refractory_period: int = 50,  # Steps before re-activation
        neighbor_k: int = 256,  # KNN neighbors
    ):
        """
        Initialize Interaction Engine.
        
        Args:
            checkpoint_path: Path to Aurora Base checkpoint
            device: torch device
            gamma: Damping coefficient
            activation_threshold: Distance threshold for concept activation
            refractory_period: Refractory period in steps
            neighbor_k: Number of neighbors for force computation
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.activation_threshold = activation_threshold
        self.refractory_period = refractory_period
        self.neighbor_k = neighbor_k
        
        # Load Aurora Base
        print(f"Loading Aurora Base from {checkpoint_path}...")
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            
        G = data['G']  # (N, dim, dim)
        self.nodes = data.get('nodes', [])
        
        # Extract positions (first column of G)
        self.positions = G[:, :, 0]  # (N, dim)
        self.N, self.dim = self.positions.shape
        print(f"Loaded {self.N} particles in H^{self.dim-1}.")
        
        # Initialize concept mapper
        self.mapper = ConceptMapper(checkpoint_path=checkpoint_path)
        
        # Initialize force field
        self.force_field = LocalForceField(self.positions, device=device)
        self.positions_tensor = torch.tensor(self.positions, dtype=torch.float32, device=self.device)
        
        # Initialize cursor state
        self.reset()
        
    def reset(self):
        """Reset cursor to origin."""
        # Origin in H^n: (1, 0, 0, ...)
        q = torch.zeros(self.dim, device=self.device)
        q[0] = 1.0
        
        p = torch.zeros(self.dim, device=self.device)
        
        self.state = CursorState(q=q, p=p)
        
    def inject_impulse(self, target_ids: List[int], strength: float = 5.0):
        """
        Inject impulse toward target concepts.
        
        Args:
            target_ids: List of particle IDs to attract toward
            strength: Impulse strength
        """
        if not target_ids:
            return
            
        # Get target positions
        target_pos = self.positions_tensor[target_ids]  # (k, dim)
        
        # Compute direction (average toward targets)
        cursor_pos = self.state.q
        
        # Simple: direction = mean(target) - cursor
        mean_target = torch.mean(target_pos, dim=0)
        direction = mean_target - cursor_pos
        
        # Normalize and scale
        norm = torch.norm(direction)
        if norm > 1e-6:
            direction = direction / norm * strength
            
        # Add to momentum
        self.state.p = self.state.p + direction
        
    def step(self, external_force: torch.Tensor = None, dt: float = 0.01) -> Optional[int]:
        """
        Run one step of cursor dynamics.
        
        Args:
            external_force: Additional external force (optional)
            dt: Time step
            
        Returns:
            Activated concept ID, or None if no activation.
        """
        self.state.step += 1
        
        # --- A. Compute Forces ---
        # Background force from local neighbors
        bg_force = self.force_field.get_force_at(
            self.state.q, 
            k=self.neighbor_k,
            step=self.state.step
        )
        
        # External force (if any)
        if external_force is not None:
            total_force = bg_force + external_force
        else:
            total_force = bg_force
            
        # --- B. Update Momentum (with damping) ---
        # dp/dt = F - gamma * p
        self.state.p = self.state.p + dt * (total_force - self.gamma * self.state.p)
        
        # --- C. Update Position ---
        # Simple Euler: dq/dt = p
        # Note: For proper hyperbolic dynamics, we should use exp map
        # MVP: Use tangent space approximation
        self.state.q = self.state.q + dt * self.state.p
        
        # Project back to hyperboloid (renormalize)
        self._renormalize_cursor()
        
        # --- D. Check for Concept Activation ---
        activated_id = self._check_activation()
        
        return activated_id
        
    def _renormalize_cursor(self):
        """Project cursor back onto hyperboloid."""
        q = self.state.q
        
        # Hyperboloid constraint: -q0^2 + sum(qi^2) = -1
        # Solve for q0: q0 = sqrt(1 + sum(qi^2))
        spatial = q[1:]
        q0 = torch.sqrt(1.0 + torch.sum(spatial ** 2))
        
        self.state.q = torch.cat([q0.unsqueeze(0), spatial])
        
    def _check_activation(self) -> Optional[int]:
        """
        Check if cursor is near any concept to activate.
        Uses hyperbolic distance.
        """
        # Compute distance to all particles (expensive, but only for nearest check)
        # In practice, use KNN result from force field
        
        if self.force_field._cached_ids is not None:
            neighbor_ids = self.force_field._cached_ids
            neighbor_pos = self.force_field._cached_pos
        else:
            return None
            
        # Hyperbolic distances
        J = torch.ones(self.dim, device=self.device)
        J[0] = -1.0
        
        inner = torch.sum(self.state.q * neighbor_pos * J, dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dists = torch.acosh(-inner)
        
        # Find nearest
        min_dist, min_idx = torch.min(dists, dim=0)
        nearest_id = int(neighbor_ids[min_idx.item()])
        nearest_dist = min_dist.item()
        
        # Check activation
        if nearest_dist < self.activation_threshold:
            if not self._in_refractory(nearest_id):
                # Activate!
                self.state.activation_buffer.append(nearest_id)
                self.state.refractory_map[nearest_id] = self.state.step + self.refractory_period
                return nearest_id
                
        return None
        
    def _in_refractory(self, concept_id: int) -> bool:
        """Check if concept is in refractory period."""
        if concept_id not in self.state.refractory_map:
            return False
        return self.state.step < self.state.refractory_map[concept_id]
        
    def process_input(self, text: str, steps: int = 100) -> List[str]:
        """
        Process user input and generate response concepts.
        
        Args:
            text: User input text
            steps: Number of simulation steps
            
        Returns:
            List of activated concept words
        """
        # Map text to particle IDs
        target_ids = self.mapper.text_to_particles(text)
        
        if not target_ids:
            print(f"Warning: No concepts found for '{text}'")
            return []
            
        print(f"Input concepts: {self.mapper.particles_to_text(target_ids)}")
        
        # Inject impulse
        self.inject_impulse(target_ids, strength=5.0)
        
        # Run simulation
        activated = []
        for _ in range(steps):
            concept_id = self.step()
            if concept_id is not None:
                word = self.mapper.particles_to_text([concept_id])[0]
                activated.append(word)
                
        return activated
        
    def get_cursor_position(self) -> np.ndarray:
        """Get current cursor position."""
        return self.state.q.detach().cpu().numpy()
        
    def get_activation_history(self) -> List[str]:
        """Get all activated concepts in this session."""
        return self.mapper.particles_to_text(self.state.activation_buffer)

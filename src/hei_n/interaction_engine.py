"""
Interaction Engine for Aurora Base (V2 - Fixed).
=================================================

Fixes based on temp-08.md analysis:
1. Sustained observation force (not one-time impulse)
2. Trajectory integration readout (not single-point KNN)
3. Lower temperature/noise for controlled exploration
"""

import torch
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import Counter

from .local_force_field import LocalForceField
from .concept_mapper import ConceptMapper
from .skeleton_index import SkeletonIndex


@dataclass
class CursorState:
    """State of the Mind Cursor."""
    q: torch.Tensor  # Position (dim,)
    p: torch.Tensor  # Momentum (dim,)
    step: int = 0
    activation_buffer: List[int] = field(default_factory=list)
    refractory_map: Dict[int, int] = field(default_factory=dict)
    # NEW: Trajectory history for integration readout
    trajectory_neighbors: Counter = field(default_factory=Counter)


class InteractionEngine:
    """
    Simulates Mind Cursor dynamics on fixed Aurora Base background.
    
    V2 Improvements:
    - Sustained observation force (持续锚定)
    - Trajectory integration readout (轨迹积分读出)
    - Lower noise for controlled exploration
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        gamma: float = 0.8,  # Higher damping for stability
        activation_threshold: float = 0.8,  # Looser threshold
        refractory_period: int = 30,
        neighbor_k: int = 512,  # More neighbors
        observation_strength: float = 10.0,  # Sustained anchor strength
        temperature: float = 0.1,  # Low noise
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.activation_threshold = activation_threshold
        self.refractory_period = refractory_period
        self.neighbor_k = neighbor_k
        self.observation_strength = observation_strength
        self.temperature = temperature
        
        # Active observation targets (sustained force)
        self.observation_targets: Optional[torch.Tensor] = None
        self.observation_target_ids: List[int] = []
        
        # Load Aurora Base
        print(f"Loading Aurora Base from {checkpoint_path}...")
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
            
        G = data['G']
        self.nodes = data.get('nodes', [])
        
        self.positions = G[:, :, 0]
        self.N, self.dim = self.positions.shape
        print(f"Loaded {self.N} particles in H^{self.dim-1}.")
        
        self.mapper = ConceptMapper(checkpoint_path=checkpoint_path)
        self.force_field = LocalForceField(self.positions, device=device)
        self.positions_tensor = torch.tensor(self.positions, dtype=torch.float32, device=self.device)
        
        # Load skeleton index if edges available
        edges = data.get('edges', None)
        if edges is not None:
            self.skeleton = SkeletonIndex(edges, self.N)
        else:
            print("Warning: No edges in checkpoint. Skeleton-first readout disabled.")
            self.skeleton = None
        
        self.reset()
        
    def reset(self):
        """Reset cursor to origin."""
        q = torch.zeros(self.dim, device=self.device)
        q[0] = 1.0
        p = torch.zeros(self.dim, device=self.device)
        
        self.state = CursorState(q=q, p=p)
        self.observation_targets = None
        self.observation_target_ids = []
        
    def set_observation(self, target_ids: List[int], strength: float = None):
        """
        Set sustained observation targets.
        These will continuously attract the cursor.
        
        Args:
            target_ids: List of particle IDs to observe
            strength: Observation strength (default: self.observation_strength)
        """
        if not target_ids:
            self.observation_targets = None
            self.observation_target_ids = []
            return
            
        self.observation_target_ids = target_ids
        self.observation_targets = self.positions_tensor[target_ids]
        if strength is not None:
            self.observation_strength = strength
            
    def _compute_observation_force(self) -> torch.Tensor:
        """
        Compute sustained observation force toward targets.
        This is applied every step, not just as initial impulse.
        """
        if self.observation_targets is None:
            return torch.zeros(self.dim, device=self.device)
            
        cursor_pos = self.state.q
        target_pos = self.observation_targets
        
        # Hyperbolic attraction force
        J = torch.ones(self.dim, device=self.device)
        J[0] = -1.0
        
        # Compute distances and directions
        inner = torch.sum(cursor_pos * target_pos * J, dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dists = torch.acosh(-inner)
        
        # Force magnitude: inverse distance (stronger when closer)
        # Use tanh for bounded force
        force_mag = self.observation_strength * torch.tanh(dists)
        
        # Direction: toward each target
        denom = torch.sqrt(inner**2 - 1.0)
        denom = torch.clamp(denom, min=1e-7)
        
        grad_d = (-1.0 / denom).unsqueeze(-1) * (target_pos * J)
        
        # Attractive force (point toward targets)
        forces = force_mag.unsqueeze(-1) * (-grad_d)
        total_force = torch.sum(forces, dim=0)
        
        return total_force
        
    def step(self, dt: float = 0.01) -> None:
        """
        Run one step of cursor dynamics.
        No longer returns activation - use trajectory readout instead.
        """
        self.state.step += 1
        
        # --- A. Compute Forces ---
        # 1. Sustained observation force (NEW)
        obs_force = self._compute_observation_force()
        
        # 2. Background force from local neighbors (weaker)
        bg_force = self.force_field.get_force_at(
            self.state.q, 
            k=self.neighbor_k,
            step=self.state.step
        ) * 0.1  # Reduce background influence
        
        total_force = obs_force + bg_force
        
        # Add small noise for exploration (controlled by temperature)
        if self.temperature > 0:
            noise = torch.randn(self.dim, device=self.device) * self.temperature
            noise[0] = 0  # No noise in time direction
            total_force = total_force + noise
            
        # --- B. Update Momentum (with damping) ---
        self.state.p = self.state.p + dt * (total_force - self.gamma * self.state.p)
        
        # --- C. Update Position ---
        self.state.q = self.state.q + dt * self.state.p
        self._renormalize_cursor()
        
        # --- D. Record trajectory for integration readout (NEW) ---
        self._record_trajectory_neighbors()
        
    def _renormalize_cursor(self):
        """Project cursor back onto hyperboloid."""
        q = self.state.q
        spatial = q[1:]
        q0 = torch.sqrt(1.0 + torch.sum(spatial ** 2))
        self.state.q = torch.cat([q0.unsqueeze(0), spatial])
        
    def _record_trajectory_neighbors(self):
        """
        Record neighbors along trajectory for integration readout.
        Called every step to accumulate activation scores.
        """
        if self.force_field._cached_ids is None:
            return
            
        neighbor_ids = self.force_field._cached_ids
        neighbor_pos = self.force_field._cached_pos
        
        # Compute distances
        J = torch.ones(self.dim, device=self.device)
        J[0] = -1.0
        
        inner = torch.sum(self.state.q * neighbor_pos * J, dim=-1)
        inner = torch.clamp(inner, max=-1.0 - 1e-7)
        dists = torch.acosh(-inner)
        
        # Score by distance (closer = higher score)
        # Use exponential decay
        scores = torch.exp(-dists).cpu().numpy()
        
        # Accumulate scores
        for i, nid in enumerate(neighbor_ids):
            if nid not in self.observation_target_ids:  # Don't count input concepts
                self.state.trajectory_neighbors[nid] += float(scores[i])
                
    def get_trajectory_output(self, top_k: int = 10, skeleton_only: bool = True) -> List[Tuple[str, float]]:
        """
        Get top-K concepts using skeleton-first readout.
        
        Args:
            top_k: Number of concepts to return
            skeleton_only: If True, only return skeleton neighbors (Priority Fix)
            
        Returns:
            List of (concept_word, score) tuples
        """
        combined_scores = Counter()
        
        # 1. Add skeleton neighbors (Priority Fix from temp-09.md)
        skeleton_neighbors = set()
        if self.skeleton is not None and self.observation_target_ids:
            for anchor_id in self.observation_target_ids:
                # Get 1-hop and 2-hop skeleton neighbors
                skel_neighbors = self.skeleton.get_neighbors(anchor_id, max_hop=2)
                skeleton_neighbors.update(skel_neighbors)
                
                # Score skeleton neighbors by their trajectory activation (if any)
                for sn in skel_neighbors:
                    trajectory_score = self.state.trajectory_neighbors.get(sn, 0)
                    combined_scores[sn] = trajectory_score + 1.0  # Base score for being skeleton neighbor
                    
        # 2. If skeleton_only, only keep skeleton neighbors
        if not skeleton_only:
            # Add trajectory neighbors not in skeleton
            for nid, score in self.state.trajectory_neighbors.items():
                if nid not in skeleton_neighbors:
                    combined_scores[nid] += score * 0.1  # Lower weight for non-skeleton
                    
        # 3. Filter out input concepts
        for anchor_id in self.observation_target_ids:
            combined_scores.pop(anchor_id, None)
            
        if not combined_scores:
            # Fallback to pure trajectory if no skeleton
            top_ids = self.state.trajectory_neighbors.most_common(top_k)
            results = []
            for concept_id, score in top_ids:
                word = self.mapper.particles_to_text([concept_id])[0]
                results.append((word, score))
            return results
            
        # Get top concepts
        top_ids = combined_scores.most_common(top_k)
        
        results = []
        for concept_id, score in top_ids:
            word = self.mapper.particles_to_text([concept_id])[0]
            results.append((word, score))
            
        return results
            
        return results
        
    def process_input(self, text: str, steps: int = 200, top_k: int = 10) -> List[str]:
        """
        Process user input and generate response concepts.
        
        V2: Uses sustained observation + trajectory readout.
        
        Args:
            text: User input text
            steps: Number of simulation steps
            top_k: Number of top concepts to return
            
        Returns:
            List of activated concept words (by trajectory score)
        """
        # Map text to particle IDs
        target_ids = self.mapper.text_to_particles(text)
        
        if not target_ids:
            print(f"Warning: No concepts found for '{text}'")
            return []
            
        input_concepts = self.mapper.particles_to_text(target_ids)
        print(f"Input concepts: {input_concepts}")
        
        # Set sustained observation (NEW)
        self.set_observation(target_ids, strength=self.observation_strength)
        
        # Run simulation
        for _ in range(steps):
            self.step()
            
        # Get trajectory output (NEW)
        results = self.get_trajectory_output(top_k=top_k)
        
        return [word for word, score in results]
        
    def get_cursor_position(self) -> np.ndarray:
        """Get current cursor position."""
        return self.state.q.detach().cpu().numpy()
        
    def get_activation_history(self) -> List[str]:
        """Get all accumulated trajectory concepts."""
        return [self.mapper.particles_to_text([cid])[0] 
                for cid in self.state.trajectory_neighbors.keys()]

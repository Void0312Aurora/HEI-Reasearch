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
                
    # Pragmatic Patch: Stop Concepts (Do not output these)
    STOP_CONCEPTS = {
        "我", "我们", "咱们", "吾侪", "你", "你们", "他", "他们", "它", "它们",
        "的", "了", "吗", "呢", "吧", "啊", "这", "那", "这个", "那个",
        "什么", "怎么", "为什么", "如果", "那么", "因为", "所以", "但是",
        "是", "在", "有", "和", "跟", "与", "就", "都", "也", "还", "只",
        "want", "need", "should", "can", "will", "would",
        "想要", "希望", "打算", "需要", "会", "要", "想"
    }

    def process_input(self, text: str, steps: int = 200, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Process user input and return activated concepts.
        """
        # 1. Map text to particles
        target_ids = self.mapper.text_to_particles(text)
        if not target_ids:
            print(f"Warning: No known concepts in input '{text}'")
            return []
            
        # 2. Set observation targets (Sustained Force)
        # Pragmatic Patch: Focus Selection
        
        content_ids = []
        function_ids = []
        
        for tid in target_ids:
            c_text_list = self.mapper.particles_to_text([tid])
            if not c_text_list: continue
            c_text = c_text_list[0]
            print(f"DEBUG: Processing '{c_text}' (Stop: {c_text in self.STOP_CONCEPTS})")
            
            if c_text in self.STOP_CONCEPTS or c_text in ["想要", "希望", "打算", "需要", "会", "要", "想"]:
                function_ids.append(tid)
            else:
                content_ids.append(tid)
                
        # Strategy: If we have content words, ONLY observe content words.
        # If we only have function words (e.g. "Who am I?"), observe function words.
        
        if content_ids:
             final_target_ids = content_ids
             print(f"Strict Focus: Observing only content ({len(content_ids)})")
        else:
             final_target_ids = function_ids
             print(f"Strict Focus: Observing function words ({len(function_ids)})")
             
        targets_tensor = torch.tensor(self.positions[final_target_ids], dtype=torch.float32, device=self.device)
        self.observation_targets = targets_tensor
        self.observation_target_ids = target_ids # Keep ALL for neighbor boosting logic (Context)?
        # NO. If we boost neighbors of function words, we get "吾辈" again.
        # But we ALREADY filtered boosting in Step 6517.
        # So we can keep self.observation_target_ids as full input for Context checking?
        # Step 6517 logic iterates self.observation_target_ids.
        # So let's keep it full, but we use final_target_ids for FORCE.
        
        self.observation_target_ids = target_ids 
        
        # 3. Simulate (Think)
        self.state.trajectory_neighbors.clear()
        
        for _ in range(steps):
            self.step()
            
        # 4. Readout (Collapse)
        results = self.get_trajectory_output(top_k=top_k)
        
        # 5. Output Fallback
        if len(results) < 3 and self.skeleton:
             # Try to expand parents of NON-STOP input concepts
             print("Triggering Fallback: Expanding parents of input...")
             for tid in target_ids:
                 c_text = self.mapper.particles_to_text([tid])[0]
                 if c_text in self.STOP_CONCEPTS or c_text in ["想要", "希望", "打算", "需要"]:
                     continue # Don't expand "want" or "I"
                     
                 parents = self.skeleton.get_neighbors(tid)
                 parent_codes = [p for p in parents if self.nodes[p].startswith('Code:')]
                 for p_code in parent_codes[:2]: 
                     expansion_results = self._expand_category_code(p_code, score=1.0)
                     results.extend(expansion_results)
                     
        # Deduplicate and Filter Stop Concepts
        seen = set()
        final_results = []
        for w, s in sorted(results, key=lambda x: x[1], reverse=True):
            if w in self.STOP_CONCEPTS:
                continue
            if w not in seen and w not in text: 
                seen.add(w)
                final_results.append((w, s))
                
        return final_results[:top_k]

    def _expand_category_code(self, code_id: int, score: float) -> List[Tuple[str, float]]:
        """Helper to expand a category code to leaf words."""
        leaves = []
        if self.skeleton is None:
            return leaves
            
        q = [code_id]
        visited = {code_id}
        expansion_limit = 5 
        
        while q and len(leaves) < expansion_limit:
            curr = q.pop(0)
            children = self.skeleton.get_children(curr)
            for child in children:
                child_node = self.nodes[child]
                if child_node.startswith('C:'):
                     # Found word
                     word_list = self.mapper.particles_to_text([child])
                     if word_list:
                         leaves.append((word_list[0], score * 0.9))
                elif child_node.startswith('Code:') and child not in visited:
                    visited.add(child)
                    q.append(child)
        return leaves

    def get_trajectory_output(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get concepts based on trajectory integration + Skeleton.
        """
        # ... (Previous Logic but refactored to use helper) ...
        # Combine trajectory neighbors
        combined_scores = Counter()
        
        # Base scores from trajectory frequency
        for nid, count in self.state.trajectory_neighbors.items():
            combined_scores[nid] += count
            
        # Skeleton Boost (Skeleton-First Readout)
        if self.skeleton:
            # For every active concept in input AND trajectory, boost its neighbors
            # Boost Sequence: Input > Trajectory
            
            # 1. Boost Input Neighbors (Context)
            for tid in self.observation_target_ids:
                # Pragmatic Patch: Focus Selection
                # Do not boost neighbors of Stop/Function concepts
                c_text_list = self.mapper.particles_to_text([tid])
                if not c_text_list: continue
                c_text = c_text_list[0]
                
                if c_text in self.STOP_CONCEPTS or c_text in ["想要", "希望", "打算", "需要", "会", "要"]:
                    continue # Skip boosting neighbors for function words
                    
                neighbors = self.skeleton.get_neighbors(tid)
                for n in neighbors:
                    combined_scores[n] += 100.0  # Huge boost for CONTENT neighbors
                    
            # 2. Boost Trajectory Neighbors (drifted thought)
            # Only if we aren't strict skeleton-only. 
            # For Phase I, we want strict skeleton control.
            pass
            
        # Filter for output
        # If skeleton is present, we ONLY return nodes that are in the skeleton neighborhood of inputs
        # or their children.
        
        # V3 Strategy: Return highest scored items, resolving Codes to Words
        top_ids = combined_scores.most_common(top_k * 2) # Get more candidates
        
        results = []
        for concept_id, score in top_ids:
            raw_node = self.nodes[concept_id]
            
            if raw_node.startswith('Code:') or raw_node == 'CilinRoot':
                # Expand
                leaves = self._expand_category_code(concept_id, score)
                results.extend(leaves)
            else:
                # Word
                word_list = self.mapper.particles_to_text([concept_id])
                if word_list:
                    results.append((word_list[0], score))
                    
        return results
            

        
    def get_cursor_position(self) -> np.ndarray:
        """Get current cursor position."""
        return self.state.q.detach().cpu().numpy()
        
    def get_activation_history(self) -> List[str]:
        """Get all accumulated trajectory concepts."""
        return [self.mapper.particles_to_text([cid])[0] 
                for cid in self.state.trajectory_neighbors.keys()]

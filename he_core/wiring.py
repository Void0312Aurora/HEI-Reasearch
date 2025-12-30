"""
Wiring Module for Multi-Entity Composition (Phase 14).

Provides:
- Edge: Defines a connection from one entity's output to another's input.
- WiringDiagram: Holds a collection of edges defining the network topology.
- TwoEntityNetwork: Minimal 2-entity network for verification.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from he_core.entity_v4 import UnifiedGeometricEntity


@dataclass
class Edge:
    """
    Defines a directed connection between two entities.
    source_id: ID of the source entity.
    target_id: ID of the target entity.
    gain: Scaling factor for the signal (default 1.0).
    delay: Number of steps to delay the signal (default 0, not implemented yet).
    """
    source_id: str
    target_id: str
    gain: float = 1.0
    delay: int = 0


class WiringDiagram:
    """
    Holds the topology of a network of entities.
    Stores a list of Edges and provides methods for routing signals.
    """
    def __init__(self, edges: List[Edge]):
        self.edges = edges
        self._build_adjacency()
        
    def _build_adjacency(self):
        """Build adjacency lists for efficient routing."""
        self.outgoing: Dict[str, List[Edge]] = {}
        self.incoming: Dict[str, List[Edge]] = {}
        
        for edge in self.edges:
            if edge.source_id not in self.outgoing:
                self.outgoing[edge.source_id] = []
            self.outgoing[edge.source_id].append(edge)
            
            if edge.target_id not in self.incoming:
                self.incoming[edge.target_id] = []
            self.incoming[edge.target_id].append(edge)
            
    def get_inputs_for(self, target_id: str) -> List[Edge]:
        """Get all edges that feed into a given entity."""
        return self.incoming.get(target_id, [])


class TwoEntityNetwork(nn.Module):
    """
    Minimal 2-Entity Network for Compositional Verification.
    
    Entities: A, B.
    Topology (configurable):
    - A -> B: A's action becomes part of B's input.
    - B -> A: B's action becomes part of A's input (optional, for feedback loop).
    
    Step Logic:
    1. Compute outputs y_A, y_B from current states.
    2. Route signals: u_A = gain * y_B (if B->A), u_B = gain * y_A (if A->B).
    3. Step both entities with routed inputs.
    """
    def __init__(self, config_A: Dict, config_B: Dict, wiring: WiringDiagram):
        super().__init__()
        self.entity_A = UnifiedGeometricEntity(config_A)
        self.entity_B = UnifiedGeometricEntity(config_B)
        self.wiring = wiring
        
        # Map entity IDs to objects
        self.entities = {
            'A': self.entity_A,
            'B': self.entity_B
        }
        
    def reset(self):
        self.entity_A.reset()
        self.entity_B.reset()
        
    def step(self, external_obs: Optional[Dict[str, torch.Tensor]] = None, dt: float = 0.1) -> Dict[str, Any]:
        """
        Perform one step of the network.
        
        external_obs: Optional external inputs for each entity (e.g., {'A': tensor, 'B': tensor}).
        Returns: Dict with actions, states, and diagnostics for each entity.
        """
        # 1. Collect current outputs (actions) from each entity
        # We need to compute forward_tensor to get action, but we also need to route
        # Strategy: First get current state actions (without stepping), then route, then step.
        # But forward_tensor STEPS the state.
        # So we should:
        #   a) Get current state flats
        #   b) Compute routed inputs based on previous step's actions (or zero for first step)
        #   c) Step each entity with routed inputs
        
        # For simplicity in v1: use previous action as routing signal.
        # Store previous actions as attributes.
        if not hasattr(self, '_prev_actions'):
            self._prev_actions = {
                'A': torch.zeros(1, self.entity_A.dim_u),
                'B': torch.zeros(1, self.entity_B.dim_u)
            }
            # For port-based entities, store port output (y) separately
            self._prev_outputs = {
                'A': torch.zeros(1, self.entity_A.dim_u),
                'B': torch.zeros(1, self.entity_B.dim_u)
            }
            
        # 2. Route signals based on wiring
        routed_inputs = {}
        for entity_id in ['A', 'B']:
            incoming_edges = self.wiring.get_inputs_for(entity_id)
            u_total = torch.zeros(1, self.entities[entity_id].dim_u)
            
            # Add external input if provided
            if external_obs and entity_id in external_obs:
                u_total += external_obs[entity_id]
                
            # Add routed signals from other entities
            # Use port output (y) if available, otherwise use action
            for edge in incoming_edges:
                source_entity = self.entities[edge.source_id]
                source_output = self._prev_outputs.get(edge.source_id, self._prev_actions[edge.source_id])
                u_total += edge.gain * source_output
                
            routed_inputs[entity_id] = u_total
            
        # 3. Step each entity
        results = {}
        new_actions = {}
        
        for entity_id in ['A', 'B']:
            entity = self.entities[entity_id]
            u_ext = routed_inputs[entity_id]
            
            # Use forward_tensor for differentiable step
            state_flat = entity.state.flat.detach()
            out = entity.forward_tensor(state_flat, u_ext, dt)
            
            # Update entity state
            entity.state.flat = out['next_state_flat'].detach()
            
            # Store action for next routing
            new_actions[entity_id] = out['action'].detach()
            
            results[entity_id] = {
                'action': out['action'],
                'H_val': out['H_val'],
                'chart_weights': out['chart_weights'],
                'state_flat': out['next_state_flat']
            }
            
        # Update previous actions and outputs
        self._prev_actions = new_actions
        
        # Store port outputs (y) for routing
        new_outputs = {}
        for entity_id in ['A', 'B']:
            entity = self.entities[entity_id]
            if hasattr(entity, 'use_port_interface') and entity.use_port_interface:
                # Get port output y from current state
                new_outputs[entity_id] = entity.interface.write_y(entity.state).detach()
            else:
                new_outputs[entity_id] = new_actions[entity_id]
        self._prev_outputs = new_outputs
        
        return results
    
    def rollout(self, steps: int, external_obs_seq: Optional[List[Dict]] = None, dt: float = 0.1) -> Dict[str, List]:
        """
        Rollout the network for multiple steps.
        Returns trajectories for each entity.
        """
        self.reset()
        trajectories = {'A': [], 'B': []}
        
        for t in range(steps):
            ext_obs = external_obs_seq[t] if external_obs_seq else None
            results = self.step(ext_obs, dt)
            
            for entity_id in ['A', 'B']:
                trajectories[entity_id].append({
                    'action': results[entity_id]['action'].detach().numpy(),
                    'H_val': results[entity_id]['H_val'].item()
                })
                
        return trajectories

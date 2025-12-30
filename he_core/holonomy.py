
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from he_core.entity_v4 import UnifiedGeometricEntity
from he_core.state import ContactState

class HolonomyAnalyzer:
    """
    Phase 21.2: Logic Interface (L2 Generator Analysis).
    Measures the 'Holonomy' (Geometric Phase / Displacement) of the system
    under cyclic driving forces.
    
    Theoretical Basis:
    Logic emerges from the path-dependence (or independence) of state evolution.
    - Zero Holonomy (Path Independence) -> "Conservative Logic" (Combinatorial)
    - Structured Holonomy -> "Sequential Logic" (Finite State Machine)
    - Chaotic Holonomy -> "No Logic" (Instability)
    """
    
    @staticmethod
    def measure_cycle_displacement(
        entity: UnifiedGeometricEntity,
        start_state: ContactState,
        action_sequence: List[torch.Tensor],
        dt: float = 0.1
    ) -> dict:
        """
        Runs the entity through the sequence of actions and measures displacement.
        Ideally, if sequence is 'structurally cyclic' (e.g. A -> -A), 
        we check if state returns to start.
        
        Args:
            entity: The system.
            start_state: Initial state (q,p,s).
            action_sequence: List of (B, dim_u) tensors.
            dt: Time step per action.
            
        Returns:
            dict with:
                'final_state': ContactState
                'q_displacement': norm(q_final - q_start)
                'p_displacement': norm(p_final - p_start)
                's_displacement': norm(s_final - s_start)
                'trajectory': List of states
        """
        # 1. Clone start to avoid mutating original
        # We need a flat tensor clone
        current_flat = start_state.flat.clone().detach()
        batch_size = start_state.batch_size
        device = start_state.device
        
        dim_q = start_state.dim_q
        
        # Initial Components
        q0 = current_flat[:, :dim_q]
        p0 = current_flat[:, dim_q:2*dim_q]
        s0 = current_flat[:, 2*dim_q:]
        
        # 2. Integrate Path
        path_q = [q0]
        
        for u_t in action_sequence:
            # Step
            # u_t: (B, dim_u)
            out = entity.forward_tensor(current_flat, u_t, dt)
            current_flat = out['next_state_flat']
            
            # Track
            q_t = current_flat[:, :dim_q]
            path_q.append(q_t)
            
        # 3. Measure Displacement
        q_final = current_flat[:, :dim_q]
        p_final = current_flat[:, dim_q:2*dim_q]
        s_final = current_flat[:, 2*dim_q:]
        
        q_disp = (q_final - q0).norm(dim=1).mean().item()
        p_disp = (p_final - p0).norm(dim=1).mean().item()
        
        # For s, it accumulates, so displacement is expected. 
        # But for 'logic', we might care about Phase consistency (s mod something).
        # For now just measure raw diff.
        s_disp = (s_final - s0).abs().mean().item()
        
        return {
            'q_disp': q_disp,
            'p_disp': p_disp,
            's_disp': s_disp,
            'final_flat': current_flat
        }

    @staticmethod
    def generate_cyclic_sequence(
        dim_u: int, 
        steps: int = 10, 
        magnitude: float = 1.0, 
        batch_size: int = 1,
        style: str = 'A_minus_A'
    ) -> List[torch.Tensor]:
        """
        Generates a test sequence of drives that sums to zero (conceptually).
        """
        seq = []
        
        if style == 'A_minus_A':
            # Half steps +A, Half steps -A
            half = steps // 2
            
            # Random Direction A per batch
            A = torch.randn(batch_size, dim_u)
            A = A / (A.norm(dim=1, keepdim=True) + 1e-6) * magnitude
            
            for _ in range(half):
                seq.append(A)
            
            for _ in range(steps - half):
                seq.append(-A)
                
        elif style == 'Box':
            # A, B, -A, -B
            quarter = steps // 4
            A = torch.randn(batch_size, dim_u)
            B = torch.randn(batch_size, dim_u)
            
            for _ in range(quarter): seq.append(A)
            for _ in range(quarter): seq.append(B)
            for _ in range(quarter): seq.append(-A)
            for _ in range(steps - 3*quarter): seq.append(-B)
            
        return seq

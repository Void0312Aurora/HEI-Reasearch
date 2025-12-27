"""
Aurora Geometric Language Model (GeometricLM).
==============================================

Implements Language Generation as Trajectory Inference on the Semantic Manifold.

Core Concept:
1. Context is a gauge frame J_ctx.
2. Generating a word moves the frame via parallel transport U.
3. Next word is chosen to minimize Interaction Energy with the current frame.

Ref: Phase 14 Plan.
"""

import torch
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
from .gauge import GaugeField

class GeometricLM:
    def __init__(self, gauge_field: GaugeField, G: nx.Graph, x: torch.Tensor, J: torch.Tensor, device='cuda'):
        """
        Args:
            gauge_field: Truly trained GaugeField.
            G: Graph (Structure + Semantics/Wormholes).
            x: Node embeddings (N, dim).
            J: Logical Charges (N, k).
        """
        self.gauge_field = gauge_field
        self.G = G
        self.x = x
        self.J = J # Base semantics for nodes
        self.device = device
        self.logical_dim = gauge_field.logical_dim
        
    def get_candidates(self, u_idx: int, k: int = 50) -> torch.Tensor:
        """
        Get candidate next tokens.
        Strategy: Neighbors in G (Struct + Sem) + Random/KNN?
        For a language model, we want 'next viable concepts'.
        Graph neighbors are "Connected Concepts".
        """
        neighbors = list(self.G.neighbors(u_idx))
        # If too few, maybe sample random?
        if len(neighbors) < k:
            # Add some randoms? Or just return what we have
             return torch.tensor(neighbors, dtype=torch.long, device=self.device)
        
        # If too many, sample? Or use all?
        # Scoring all neighbors is cheap.
        return torch.tensor(neighbors, dtype=torch.long, device=self.device)

    def generate_stream(self, seed_idx: int, length: int = 10, temperature: float = 0.5) -> List[Dict]:
        """
        Generate a "Stream of Consciousness" trajectory.
        
        Args:
            seed_idx: Starting word index.
            length: Number of steps.
            temperature: Softmax temperature (lower = more greedy).
            
        Returns:
            List of dicts: {'word_idx': int, 'energy': float, 'type': str}
        """
        trajectory = []
        
        curr_idx = seed_idx
        # Initial Context J is the seed's J
        J_ctx = self.J[seed_idx].clone()
        
        trajectory.append({'word_idx': curr_idx, 'energy': 0.0, 'type': 'seed'})
        
        visited = set()
        visited.add(curr_idx)
        
        for _ in range(length):
            candidates = self.get_candidates(curr_idx)
            if candidates.numel() == 0:
                break
                
            # Filter visited to prevent immediate loops (A->B->A)
            # But language can repeat? "Great, really great".
            # Let's just prevent 'curr' (self-loop).
            mask = candidates != curr_idx
            if mask.sum() == 0:
                 break
            candidates = candidates[mask]
            
            # 1. Compute Candidate J's
            J_cand = self.J[candidates] # (C, k)
            
            # 2. Compute Interaction Energy E = 1 - <J_cand, J_ctx>
            # IMPORTANT: J_ctx is properly transported to 'curr' frame space?
            # Actually:
            # We assume J_ctx is IN THE FRAME OF 'curr'.
            # So we compare directly with neighbors J (which are in their own frames??)
            # NO.
            # In Gauge Field theory:
            # Interaction Energy E_uv = 1 - <J_v, U_uv J_u>.
            # Here J_ctx is at 'curr' (u). We want to pick 'next' (v).
            # So we must transport J_ctx to v -> (U_uv J_ctx).
            # Then compare with J_v.
            # Metric: <J_v, U_{curr->v} J_ctx>.
            
            C = candidates.shape[0]
            # U_{curr->cand}
            curr_tensor = torch.full((C,), curr_idx, dtype=torch.long, device=self.device)
            edges_t = torch.stack([curr_tensor, candidates], dim=1)
            
            U = self.gauge_field.get_U(x=self.x, edges=edges_t) # (C, k, k)
            
            # Transport J_ctx (which is at curr) to cand
            # J_ctx: (k,) -> (1, k, 1) broadcast?
            # U: (C, k, k)
            # J_transported = U * J_ctx
            J_ctx_expanded = J_ctx.unsqueeze(0).unsqueeze(-1).expand(C, -1, -1) # (C, k, 1)
            J_transported = torch.matmul(U, J_ctx_expanded).squeeze(-1) # (C, k)
            
            # Dot Product
            alignment = torch.sum(J_cand * J_transported, dim=-1) # (C,)
            
            # Energy = 1 - alignment (assuming normalized)
            scores = alignment # Higher is better
            
            # Selection
            # Softmax sampling
            probs = torch.softmax(scores / temperature, dim=0)
            next_i = torch.multinomial(probs, 1).item()
            
            next_idx = candidates[next_i].item()
            energy_val = 1.0 - scores[next_i].item()
            
            # Update Context
            # Move J_ctx to the new word's frame
            # J_ctx_new = U_{curr->next} * J_ctx_old
            # We already computed this! J_transported[next_i]
            # BUT: Should we strictly adopt the new word's J?
            # Or keep the "Drifting Thought"?
            # "Trajectory" implies preserving momentum/intent.
            # If we just reset to J_next, it's Markovian (memoryless).
            # If we keep transporting J_ctx, it accumulates history (Geometric LSTM).
            # Let's doing: J_ctx_new = alpha * J_next + (1-alpha) * Transported_J_ctx ?
            # Pure GeometricLM: Keep transporting J_ctx. 
            # But numerical error might accumulate.
            # Let's Normalize.
            
            J_ctx_new = J_transported[next_i].clone()
            J_ctx_new = J_ctx_new / torch.norm(J_ctx_new)
            
            # Determine Edge Type (Tree/Tunnel)
            edge_data = self.G.get_edge_data(curr_idx, next_idx)
            edge_type = edge_data.get('type', 'unknown') if edge_data else 'unknown'
            
            trajectory.append({
                'word_idx': next_idx, 
                'energy': energy_val,
                'type': edge_type
            })
            
            curr_idx = next_idx
            J_ctx = J_ctx_new
            visited.add(curr_idx)
            
        return trajectory

    def cloze_test(self, left_idx: int, right_idx: int, k_candidates: int = 100) -> Tuple[int, float]:
        """
        Fill [MASK] in "Left [MASK] Right".
        
        Logic:
        1. Forward Transport: J_left -> U_{L->M} -> J_L_at_M
        2. Backward Transport: J_right -> U_{R->M} -> J_R_at_M ??
           Wait, U is directional. U_{M->R}.
           So J_R_at_M = U_{M->R}^{-1} * J_right = U_{R->M} * J_right (if orthogonal).
           Yes U_{ji} = U_{ij}^T.
           
        But we don't know M yet!
        We need to iterate candidates M.
        
        Candidate M:
        - Must be neighbor of Left? Or neighbor of Right? Or both?
        - If "Left M Right" is a sequence, M likely neighbor of Left.
        - And Right likely neighbor of M.
        - Intersection of N(Left) and N(Right)?
        - If null intersection, expand search?
        
        Let's try Candidates = Neighbors(Left).
        Score = E(Left->M) + E(M->Right).
        """
        candidates = list(self.G.neighbors(left_idx))
        if not candidates:
            return -1, 999.0
            
        cand_tensor = torch.tensor(candidates, dtype=torch.long, device=self.device)
        C = len(candidates)
        
        # 1. Score Left -> M
        # edges (Left, M)
        left_tensor = torch.full((C,), left_idx, dtype=torch.long, device=self.device)
        edges_LM = torch.stack([left_tensor, cand_tensor], dim=1)
        U_LM = self.gauge_field.get_U(x=self.x, edges=edges_LM)
        
        J_L = self.J[left_idx_t := torch.tensor(left_idx, device=self.device)]
        J_L_exp = J_L.view(1, -1, 1).expand(C, -1, -1)
        J_L_at_M = torch.matmul(U_LM, J_L_exp).squeeze(-1)
        
        J_M = self.J[cand_tensor]
        align_LM = torch.sum(J_M * J_L_at_M, dim=-1)
        
        # 2. Score M -> Right
        # We need J_M transported to Right? 
        # Or J_Right transported to M?
        # Energy is symmetric if U is unitary. 
        # let's behave as flow: Left -> M -> Right.
        # Transmit J_L_at_M -> U_MR -> J_at_R?
        # And compare with J_R.
        # This models "Sentence Continuity".
        
        # edges (M, Right)
        right_tensor = torch.full((C,), right_idx, dtype=torch.long, device=self.device)
        edges_MR = torch.stack([cand_tensor, right_tensor], dim=1)
        U_MR = self.gauge_field.get_U(x=self.x, edges=edges_MR)
        
        # Transport J_L_at_M ("The Thought so far") to Right
        J_L_at_M_exp = J_L_at_M.unsqueeze(-1)
        J_flow_at_R = torch.matmul(U_MR, J_L_at_M_exp).squeeze(-1)
        
        J_R = self.J[right_idxt := torch.tensor(right_idx, device=self.device)]
        align_MR = torch.sum(J_R * J_flow_at_R, dim=-1)
        
        # Total Score
        # Maximize Alignment
        total_align = align_LM + align_MR
        
        best_idx = torch.argmax(total_align).item()
        best_cand = candidates[best_idx]
        best_score = total_align[best_idx].item()
        
        return best_cand, best_score


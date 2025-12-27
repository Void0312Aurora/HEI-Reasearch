"""
Aurora Geometric Grammar Engine.
================================

Implements Syntax as Geometry.
Defines `GrammarTrajectory` to evolve semantic frames under typed relations (Subject, Object, etc.).

Ref: Phase 15 Plan.
"""

import torch
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
from .gauge import GaugeField

class GrammarTrajectory:
    def __init__(self, gauge_field: GaugeField, x: torch.Tensor, J: torch.Tensor, device='cuda'):
        self.gauge_field = gauge_field
        self.x = x
        self.J = J
        self.device = device
        
    def step(self, curr_idx: int, relation_id: int, candidates: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """
        Perform one grammatical step: Transport J_curr via relation_id to find best candidate.
        
        Args:
            curr_idx: Current word index.
            relation_id: ID of the relation (0=Neutral, 1=Subj, 2=Obj...).
            candidates: Tensor of candidate indices.
            
        Returns:
            (best_idx, energy, J_new)
        """
        # 1. Prepare Inputs
        C = candidates.shape[0]
        curr_tensor = torch.full((C,), curr_idx, dtype=torch.long, device=self.device)
        edges_t = torch.stack([curr_tensor, candidates], dim=1)
        
        # Relation IDs
        rel_tensor = torch.full((C,), relation_id, dtype=torch.long, device=self.device)
        
        # 2. Compute U_{relation}
        U = self.gauge_field.get_U(x=self.x, edges=edges_t, relation_ids=rel_tensor)
        
        # 3. Transport J_curr
        # Assuming J_curr is the logical charge of the current word
        # In full continuous mode, we track J_traj.
        # Here we use J[curr_idx] as base state for simplicity, or we can pass dynamic J.
        # Let's align with GeometricLM: Context is Dynamic.
        # BUT for this function signature I assume we just transition from Node to Node.
        # Let's return J_new so caller can maintain context if desired.
        
        J_curr = self.J[curr_idx] # (k,)
        J_curr_exp = J_curr.view(1, -1, 1).expand(C, -1, -1)
        J_transported = torch.matmul(U, J_curr_exp).squeeze(-1) # (C, k)
        
        # 4. Compare with J_cand
        J_cand = self.J[candidates]
        alignment = torch.sum(J_cand * J_transported, dim=-1)
        
        # 5. Selection
        best_i = torch.argmax(alignment).item()
        best_idx = candidates[best_i].item()
        best_energy = 1.0 - alignment[best_i].item()
        J_new = J_transported[best_i] # This is J_curr transported to new frame
        
        return best_idx, best_energy, J_new

class SentenceGenerator:
    def __init__(self, traj_engine: GrammarTrajectory, G: nx.Graph, ds: 'AuroraDataset'):
        self.engine = traj_engine
        self.G = G
        self.ds = ds # For vocab lookup
        
        # Define Simple Grammar (Relation IDs)
        # 0: Neutral (Association)
        # 1: Subject->Predicate
        # 2: Predicate->Object
        self.REL_SUBJ_PRED = 1
        self.REL_PRED_OBJ = 2
        
    def generate_svo(self, subject_word: str) -> Dict:
        """
        Generate a Subject-Predicate-Object sentence starting from subject.
        """
        try:
            subj_idx = self.ds.vocab.word_to_id[subject_word]
        except KeyError:
            return {"error": f"Word {subject_word} not found."}
            
        # 1. Subject -> Predicate
        # Candidates: Neighbors?
        # Ideally we want Verbs. Hard to filter without POS tags.
        # We rely on the Typed Gauge Field to prefer 'Predicates' given relation 1.
        candidates = list(self.G.neighbors(subj_idx))
        if not candidates: return {"error": "No neighbors."}
        cand_t = torch.tensor(candidates, dtype=torch.long, device=self.engine.device)
        
        pred_idx, e1, _ = self.engine.step(subj_idx, self.REL_SUBJ_PRED, cand_t)
        
        # 2. Predicate -> Object
        candidates_2 = list(self.G.neighbors(pred_idx))
        if not candidates_2: return {"frames": [subj_idx, pred_idx]}
        cand_t_2 = torch.tensor(candidates_2, dtype=torch.long, device=self.engine.device)
        
        obj_idx, e2, _ = self.engine.step(pred_idx, self.REL_PRED_OBJ, cand_t_2)
        
        return {
            "subject": self.ds.nodes[subj_idx],
            "predicate": self.ds.nodes[pred_idx],
            "object": self.ds.nodes[obj_idx],
            "energy_sp": e1,
            "energy_po": e2
        }


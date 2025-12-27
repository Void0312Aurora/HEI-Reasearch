"""
Aurora Geometric Decoder (Riemannian Beam Search).
==================================================

Generates text by finding the path of least action on the Gauge Manifold.
Objective: Go from J_start to J_target.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from ..gauge import GaugeField
from ..data import AuroraDataset

class RiemannianBeamSearchDecoder:
    def __init__(self, gauge_field: GaugeField, x: torch.Tensor, J_vocab: torch.Tensor, 
                 ds: AuroraDataset, device='cuda'):
        self.gauge_field = gauge_field
        self.x = x
        self.J_vocab = J_vocab # J of all words in vocab
        self.ds = ds
        self.device = device
        
    def decode(self, J_start: torch.Tensor, J_target: torch.Tensor, 
               max_len: int = 10, beam_width: int = 5) -> List[str]:
        """
        Find sequence of words w1, w2... such that J_start -> w1 -> J1 -> w2 -> J2 ... ~ J_target.
        """
        # Beam State: (current_J, path_ids, total_energy)
        beam = [(J_start, [], 0.0)]
        
        # Pre-compute vocab IDs for efficiency?
        # Ideally we only search relevant words.
        # Simple Logic: Search all 10k words? Expensive for GaugeField?
        # Optimization: Filter by Semantic Similarity to J_target first?
        # "Guided Beam Search".
        
        # Let's use Top-50 neighbors of J_target as candidates?
        # No, we need words that Validly Extend J_start (Flow), not just synonyms of Target.
        # But Flow depends on Previous Word.
        # So we look for words w that have Low Energy with w_prev.
        # AND lead towards Target.
        
        # Assuming we trained U on Flow.
        # U_flow(u, v) is low energy if u->v is valid text.
        
        vocab_size = self.J_vocab.shape[0]
        
    def _get_candidates(self, current_J: torch.Tensor, last_id: int, top_k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Candidate Words (Layered Approach).
        L1: Grammar Neighbors (High probability flow/phrase).
        L2: Semantic Neighbors (KNN in J space).
        """
        # Strategy:
        # Since we don't have an explicit adjacency list for "Grammar Neighbors" (U is continuous),
        # we rely on Semantic KNN as the primary retrieval mechanism for now, 
        # BUT we weight them by Flow Energy in the scoring phase.
        
        # Ideally, we would have a 'Next Token Prediction' head (Language Model style), 
        # but here we are purely geometric.
        
        # So we stick to KNN on J_vocab relative to (current_J + step_direction) or just current_J?
        # Actually, we want words that are physically close to the Target trajectory.
        
        # Let's return Top 2*k indices based on raw closeness to Target AND Current.
        return torch.arange(self.J_vocab.shape[0], device=self.device), self.J_vocab
        # For full vocab search (efficient enough for 14k on GPU). 
        # If vocab > 100k, we need FAISS or ScaNN.

    def decode(self, J_start: torch.Tensor, J_target: torch.Tensor, 
               max_len: int = 10, beam_width: int = 5,
               diversity_penalty: float = 0.5) -> List[str]:
        """
        Find sequence of words w1, w2...
        Score = Energy_Flow + beta * Energy_Goal + gamma * Diversity.
        """
        # Beam: (current_J, path_ids, cumulative_energy, visited_set)
        beam = [(J_start, [], 0.0, set())]
        
        vocab_size = self.J_vocab.shape[0]
        id2word = self.ds.vocab.id_to_word
        
        for step in range(max_len):
            candidates_pool = []
            
            for (J_curr, path, energy, visited) in beam:
                last_id = path[-1] if path else -1
                
                # 1. Evaluate Full Vocab (Batch)
                # Optimization: Run in one giant batch if V < 20k
                
                # Flow Energy: 1 - <J_v, U * J_curr>
                if last_id == -1:
                    # First step: Pure Semantic Distance to Target
                    # Or Distance from Start?
                    # Let's say Cost = Dist(J_v, J_target)
                    dist_target = torch.norm(self.J_vocab - J_target, dim=-1)
                    total_scores = dist_target
                else:
                    # U_flow * J_curr
                    u_tensor = torch.full((vocab_size,), last_id, dtype=torch.long, device=self.device)
                    v_tensor = torch.arange(vocab_size, device=self.device)
                    # Try both Flow (100) and Phrase (101) edges?
                    # Let's default to Flow (100)
                    r_tensor = torch.full((vocab_size,), 100, dtype=torch.long, device=self.device)
                    
                    # Compute U in chunks to save memory
                    chunk_size = 1024
                    flow_energies = []
                    
                    for i in range(0, vocab_size, chunk_size):
                        end = min(i + chunk_size, vocab_size)
                        e_b = torch.stack([u_tensor[i:end], v_tensor[i:end]], dim=1)
                        r_b = r_tensor[i:end]
                        
                        U = self.gauge_field.get_U(x=self.x, edges=e_b, relation_ids=r_b)
                        J_u = self.J_vocab[last_id].view(1, -1, 1)
                        J_trans = torch.matmul(U, J_u).squeeze(-1)
                        
                        J_v = self.J_vocab[i:end]
                        align = torch.sum(J_v * J_trans, dim=-1)
                        flow_energies.append(1.0 - align)
                        
                    e_flow = torch.cat(flow_energies)
                    e_goal = torch.norm(self.J_vocab - J_target, dim=-1)
                    
                    # Total Score
                    total_scores = e_flow + 0.5 * e_goal
                
                # Top-K for this beam
                k_local = beam_width * 2
                vals, idxs = torch.topk(total_scores, k_local, largest=False)
                
                for v, idx in zip(vals, idxs):
                    idx_item = idx.item()
                    
                    # Diversity / Repetition Penalty
                    if idx_item in visited:
                        penalty = 10.0 # Huge penalty for repeat
                    else:
                        penalty = 0.0
                        
                    # Soft Diversity (Semantic) - Avoid words too close to existing path?
                    # Skip for speed.
                        
                    new_energy = energy + v.item() + penalty
                    new_path = path + [idx_item]
                    new_visited = visited.copy()
                    new_visited.add(idx_item)
                    
                    candidates_pool.append((self.J_vocab[idx], new_path, new_energy, new_visited))
            
            # Select Global Top-K
            candidates_pool.sort(key=lambda x: x[2])
            
            # Prune duplicates (same end node)
            seen_ends = set()
            new_beam = []
            for c in candidates_pool:
                end_node = c[1][-1]
                if end_node not in seen_ends:
                    new_beam.append(c)
                    seen_ends.add(end_node)
                if len(new_beam) >= beam_width: break
            
            beam = new_beam
            
            # Stop Condition
            best_J = beam[0][0]
            if torch.norm(best_J - J_target) < 0.6: # Relaxed threshold
                break
                
        # Return text
        best_path = beam[0][1]
        res = []
        for i in best_path:
            word = id2word[i] if id2word and 0 <= i < len(id2word) else "UNK"
            res.append(word)
        return res

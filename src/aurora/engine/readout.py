"""
Aurora Readout Mechanism (CCD v3.1).
====================================

Decodes physical state u(t) into characters.
Apply Soft Projection (Bayesian Update).

Ref: Axiom 3.3.
"""

import torch
import torch.nn.functional as F
from ..model.injector import AuroraInjector
from ..data.data_pipeline import GlobalEntropyStats
from ..physics import geometry

class ReadoutMechanism:
    def __init__(self, injector: AuroraInjector, entropy_stats: GlobalEntropyStats, vocab: dict):
        self.injector = injector
        self.entropy = entropy_stats
        self.vocab = vocab # id -> char
        self.inv_vocab = {v:k for k,v in vocab.items()}
        
        self.psi_cache = {} # id -> (m, q, J)
        self._precompute_prototypes()
        
    def _precompute_prototypes(self):
        """Precompute Psi for all chars in vocab (Context-free)."""
        print("Precomputing readout prototypes...")
        device = next(self.injector.parameters()).device
        ids = torch.arange(len(self.vocab), device=device)
        
        # We need entropy 'r' for each char.
        # This is slow if loop. Batch it.
        # But GlobalEntropyStats is CPU dict.
        r_list = []
        for i in range(len(self.vocab)):
            char = self.vocab[i]
            r = self.entropy.get_radial_target(char)
            r_list.append(r)
        
        r_tensor = torch.tensor(r_list, device=device).unsqueeze(1) # (V, 1)
        ids_in = ids.unsqueeze(1) # (V, 1)
        
        with torch.no_grad():
            # Injector expects (Batch, Seq). 
            # We treat each char as independent sequence of len 1.
            m, q, J, _ = self.injector(ids_in, r_tensor) 
            # Output: (V, 1, ...)
            
        self.psi_m = m.squeeze(1)
        self.psi_q = q.squeeze(1)
        self.psi_J = J.squeeze(1)
        
    def read_prob(self, state_q: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
        """
        Compute P(c | q).
        P ~ exp( -beta * dist(q, q_c)^2 )
        """
        # q: (N, dim). vocab: (V, dim).
        # Pairwise distance.
        # Expand
        q_exp = state_q.unsqueeze(1) # (N, 1, D)
        vocab_q = self.psi_q.unsqueeze(0) # (1, V, D)
        
        # Euclidean approx for speed? Or Mobius diff?
        # Readout is usually local.
        # V3.1 says ||u - Psi||^2.
        # Let's use Euclidean norm of Mobius diff (Chordal distance equivalent).
        diff = geometry.mobius_add(-q_exp, vocab_q)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, V)
        
        prob = F.softmax(-beta * dist_sq, dim=-1)
        return prob
        
    def collapse(self, state: dict, char_idx: int, alpha: float = 0.5):
        """
        Soft Projection of state towards char.
        q_new = (1-alpha)q + alpha q_target?
        Geodesic Interp: Exp_q( alpha * Log_q(q_target) ).
        """
        q = state['q']
        target_q = self.psi_q[char_idx].unsqueeze(0) # (1, D)
        
        if q.shape[0] > 1:
            # We assume we are collapsing the 'Cursor' particle?
            # Or the whole system?
            # Usually readout focuses on the lead particle.
            pass
            
        # Geodesic Interpolation
        # v = Log_q(target)
        v = geometry.log_map(q, target_q)
        # q_new = Exp_q(alpha * v)
        q_new = geometry.exp_map(q, alpha * v)
        
        state['q'] = q_new
        # Momentum recoil?
        # p_new = p - dE.
        # Skip for prototype.
        return state

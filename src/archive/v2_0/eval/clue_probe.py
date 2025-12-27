"""
Aurora CLUE Probe Engine.
=========================

Zero-shot evaluation of Logic/Inference tasks using Geometric Energy.
Hypothesis:
- Entailment (A->B): High Alignment / Low Energy.
- Contradiction (A!=B): Low Alignment / High Energy / High Curvature.
- Neutral: Moderate Energy.

Method:
1. Parse Sent A, Sent B into Geometric Frames J_A, J_B (e.g. Bag-of-Frames or Dependency Root).
2. Measure Alignment <J_A, U_AB J_B> (or just <J_A, J_B> if assuming common frame).
3. Thresholding.
"""

import torch
import numpy as np
from typing import List, Tuple
from ..gauge import GaugeField
from ..data import AuroraDataset
from ..ingest.wikipedia import WikipediaIngestor

class GeometricProbe:
    def __init__(self, gauge_field: GaugeField, x: torch.Tensor, J: torch.Tensor, ds: AuroraDataset, device='cuda'):
        self.gauge_field = gauge_field
        self.x = x
        self.J = J
        self.ds = ds
        self.ingestor = WikipediaIngestor(ds) # For tokenization
        self.device = device
        
    def _get_sentence_frame(self, text: str) -> torch.Tensor:
        """
        Convert text to a single Frame J_sent.
        Simple View: Average of Word Frames (Bag-of-Words).
        Better View: Root of Dependency Tree.
        We use Average for robustness.
        """
        tokens = self.ingestor.tokenize(text)
        ids = [self.ds.vocab.word_to_id.get(t) for t in tokens]
        valid_ids = [i for i in ids if i is not None]
        
        if not valid_ids:
            return torch.zeros(self.J.shape[1], device=self.device)
            
        # Average J
        valid_t = torch.tensor(valid_ids, dtype=torch.long, device=self.device)
        J_words = self.J[valid_t] # (N, k)
        J_sent = torch.mean(J_words, dim=0) # (k,)
        # Normalize?
        # J_sent = J_sent / torch.norm(J_sent)
        return J_sent
        
    def score_similarity(self, text_a: str, text_b: str) -> float:
        """
        AFQMC Task: Are A and B similar?
        Metric: Cosine Similarity of J_A, J_B.
        """
        J_a = self._get_sentence_frame(text_a)
        J_b = self._get_sentence_frame(text_b)
        
        # Cosine Sim
        norm_a = torch.norm(J_a)
        norm_b = torch.norm(J_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
            
        sim = torch.sum(J_a * J_b) / (norm_a * norm_b)
        return sim.item()
        
    def extract_features(self, text_a: str, text_b: str) -> np.ndarray:
        """
        Extract geometric features for downstream calibration.
        Features: [CosineSim, Norm_A, Norm_B, Energy?]
        """
        J_a = self._get_sentence_frame(text_a)
        J_b = self._get_sentence_frame(text_b)
        
        norm_a = torch.norm(J_a).item()
        norm_b = torch.norm(J_b).item()
        
        if norm_a < 1e-6 or norm_b < 1e-6:
            sim = 0.0
        else:
            sim = torch.sum(J_a * J_b) / (norm_a * norm_b)
            sim = sim.item()
            
        # TODO: Add Holonomy / Path Energy if U is used.
        # For now, just Cosine Sim is the primary feature.
        return np.array([sim, norm_a, norm_b])

    def predict_entailment(self, text_a: str, text_b: str) -> str:
        """
        OCNLI Task: Entailment, Neutral, Contradiction.
        Metric: Alignment Score.
        High -> Entailment.
        Low/Negative -> Contradiction.
        Mid -> Neutral.
        """
        sim = self.score_similarity(text_a, text_b)
        
        # Thresholds (Calibration needed)
        # Assuming Hyperbolic/Compact space:
        # > 0.7: Entailment
        # < 0.3: Contradiction? (Or < 0?)
        # Else: Neutral
        
        if sim > 0.6:
            return "entailment"
        elif sim < 0.3:
            return "contradiction"
        else:
            return "neutral"


"""
Aurora Agent Planner.
=====================

Generates Geometric Trajectories.
"""

from .base import Planner, AgentContext, AgentPath
from ..generation.decoder import RiemannianBeamSearchDecoder
from ..data import AuroraDataset
import torch

class GeometricPlanner(Planner):
    def __init__(self, decoder: RiemannianBeamSearchDecoder):
        self.decoder = decoder
        
    def plan(self, context: AgentContext) -> AgentPath:
        """
        Generate a path connecting concepts in the context.
        Strategy:
        - If 2+ anchors: Path from A1 -> A2.
        - If 1 anchor: Path from A1 -> Random/Central Node? (Exploration)
        """
        anchors = list(context.anchors.values())
        path = AgentPath()
        
        if not anchors:
            # Fallback: No anchors found.
            return path
            
        if len(anchors) >= 1:
            # For simplicity in Phase 29, let's assume we want to "Expand" the first anchor
            # If we have 2, we connect them.
            start_id = anchors[0]
            
            if len(anchors) >= 2:
                target_id = anchors[1]
                target_J = self.decoder.J_vocab[target_id]
            else:
                # Explore: Target is self? Or a neighbor?
                # Let's just return a trivial path for now
                target_id = start_id
                target_J = self.decoder.J_vocab[target_id]
            
            start_J = self.decoder.J_vocab[start_id]
            
            # Use Decoder
            # Note: Decoder returns List[str]. We might want List[int] from decoder to keep it geometric?
            # The modified decoder returns List[str]. 
            # We should probably modify decoder to return IDs or handle conversion here.
            # But Planner output is AgentPath which has 'steps' (IDs).
            
            # Hack: We need decoder to give us IDs. 
            # Let's peek at decoder internals or subclass it.
            # Actually, `decoder.decode` returns strings.
            # Let's verify decoder code. It converts to strings at the very end.
            
            # For now, let's map strings back to IDs (inefficient but safe).
            word_path = self.decoder.decode(start_J, target_J, max_len=6, beam_width=5)
            
            ids = []
            ds = self.decoder.ds
            for w in word_path:
                idx = ds.vocab.word_to_id.get(w)
                if idx is not None:
                    ids.append(idx)
                    
            path.steps = ids
            
        return path

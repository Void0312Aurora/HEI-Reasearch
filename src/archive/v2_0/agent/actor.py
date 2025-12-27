"""
Aurora Agent Actor.
===================

Surface Realization: Path -> Text.
"""

from .base import Actor, AgentContext, AgentPath
from ..data import AuroraDataset

class GeometricActor(Actor):
    def __init__(self, ds: AuroraDataset):
        self.ds = ds
        
    def act(self, path: AgentPath, context: AgentContext) -> str:
        """
        Convert path IDs to a natural language sentence using templates.
        """
        if not path.steps:
            if context.intent == "greeting":
                return "Hello! I am Aurora, a Geometric Agent. I can calculate, search, or explain semantic connections."
            elif context.intent == "question":
                return "I understand this is a question, but I couldn't map the key concepts to my knowledge base."
            elif context.intent == "unknown":
                return "I didn't understand that. Could you rephrase using concepts I know (like Person, Eat, Plant)?"
            return "I couldn't find a logical path to answer that."
            
        words = []
        for idx in path.steps:
            if idx < 0 or idx >= len(self.ds.vocab):
                w = "UNK"
            else:
                w = self.ds.vocab.id_to_word[idx]
            
            # Clean Cilin Format
            if w.startswith("C:") and ":" in w:
                parts = w.split(":")
                if len(parts) >= 3:
                     w = parts[1]
            words.append(w)
            
        if len(words) == 1:
            return f"The core concept is {words[0]}."
            
        # Template Selection based on Intent?
        # For now, just a Flow Template.
        
        # "A leads to B, which implies C."
        if len(words) == 2:
            return f"Thinking geometrically, '{words[0]}' is directly connected to '{words[1]}'."
            
        if len(words) >= 3:
            mid = ", then " + ", then ".join(words[1:-1])
            return f"Starting from '{words[0]}'{mid}, we finally reach '{words[-1]}'."
            
        return " -> ".join(words)

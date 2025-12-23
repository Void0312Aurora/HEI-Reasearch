"""
Language Realizer for Aurora Interaction Engine.
================================================

Converts activated concept sequence to natural language.
MVP: Simple concatenation with punctuation.
"""

from typing import List


class LanguageRealizer:
    """Convert concept sequences to human-readable text."""
    
    def __init__(self, style: str = "simple"):
        """
        Initialize realizer.
        
        Args:
            style: "simple" (MVP) or "template" or "llm" (future)
        """
        self.style = style
        
    def realize(self, concepts: List[str]) -> str:
        """
        Convert concept list to sentence.
        
        Args:
            concepts: List of concept words
            
        Returns:
            Human-readable sentence
        """
        if not concepts:
            return "..."
            
        if self.style == "simple":
            return self._realize_simple(concepts)
        else:
            return self._realize_simple(concepts)
            
    def _realize_simple(self, concepts: List[str]) -> str:
        """
        Simple concatenation with ellipsis.
        
        Example: ["apple", "fruit", "red"] -> "……apple……fruit……red……"
        """
        if len(concepts) == 1:
            return f"……{concepts[0]}……"
            
        # Join with ellipsis
        parts = ["……"]
        for c in concepts:
            parts.append(c)
            parts.append("……")
            
        return "".join(parts)
        
    def realize_stream(self, concepts: List[str]) -> str:
        """
        Stream-style output (for printing as thinking).
        
        Example: ["apple", "fruit"] -> "...apple...fruit..."
        """
        if not concepts:
            return ""
            
        return "..." + "...".join(concepts) + "..."

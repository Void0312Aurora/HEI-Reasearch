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
        
    def realize(self, concepts: List[str], input_text: str = None) -> str:
        """
        Convert concept list to sentence.
        
        Args:
            concepts: List of concept words
            input_text: Optional input text for context
            
        Returns:
            Human-readable sentence
        """
        if not concepts:
            return "..."
            
        if self.style == "simple":
            return self._realize_simple(concepts)
        elif self.style == "template":
            return self._realize_template(concepts, input_text)
        else:
            return self._realize_simple(concepts)
            
    def _realize_simple(self, concepts: List[str]) -> str:
        """
        Simple concatenation with ellipsis.
        """
        if len(concepts) == 1:
            return f"……{concepts[0]}……"
            
        return "……" + "……".join(concepts) + "……"
        
    def _realize_template(self, concepts: List[str], input_text: str = None) -> str:
        """
        Template-based realization.
        Structure: [Category/Topic] -> [List] -> [Summary]
        """
        # 1. Identify Topic (First concept or Input)
        topic = input_text if input_text else concepts[0]
        
        # 2. Filter concepts (exclude exact topic match to avoid repetition)
        related = [c for c in concepts if c != topic]
        if not related:
            return f"{topic}……"
            
        # 3. Choose Template (English vs Chinese check)
        # Using a simple heuristic - if topic has Chinese char, use Chinese template
        is_zh = any('\u4e00' <= char <= '\u9fff' for char in topic)
        
        if is_zh:
            # Chinese Template
            # "关于[Topic]：[A]、[B]、[C]……"
            items = "、".join(related[:5])  # Top 5
            return f"关于{topic}，我想到了：{items}。"
        else:
            # English Template
            # "Thinking of [Topic]: [A], [B], [C]..."
            items = ", ".join(related[:5])
            return f"Thinking of {topic}: {items}."

    def realize_stream(self, concepts: List[str]) -> str:
        """Stream-style output."""
        if not concepts:
            return ""
        return "..." + "...".join(concepts) + "..."

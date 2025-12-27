"""
Aurora Agent Tools.
===================

Defines the Tool Interface and standard tools (Calculator, Search).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class AgentTool(ABC):
    def __init__(self, name: str, triggers: List[str]):
        self.name = name
        self.triggers = triggers # Keywords that might trigger this tool
        
    @abstractmethod
    def execute(self, query: str) -> str:
        """Execute the tool on the query string."""
        pass

class CalculatorTool(AgentTool):
    def __init__(self):
        super().__init__("Calculator", ["calculate", "calc", "math", "evaluate"])
        
    def execute(self, query: str) -> str:
        """
        Parses simple math expressions.
        Extracts numbers and operators.
        """
        # Very simple unsafe eval for demo (sandbox in production!)
        # We try to strip non-math chars
        allowed = "0123456789+-*/(). "
        cleaned = "".join([c for c in query if c in allowed])
        try:
            val = eval(cleaned)
            return f"{cleaned} = {val}"
        except:
            return "Error calculating."

class WikiSearchTool(AgentTool):
    def __init__(self):
        super().__init__("WikiSearch", ["search", "who is", "what is", "lookup"])
        
    def execute(self, query: str) -> str:
        """
        Mock Wiki Search.
        """
        # In a real system, this would query a search engine or the Wiki dump index.
        # For prototype, we return canned responses or generic info.
        
        if "elon musk" in query.lower():
            return "Elon Musk: CEO of SpaceX and Tesla."
        if "physics" in query.lower():
            return "Physics: The natural science that studies matter and energy."
            
        return f"Search result for '{query}': [Mock Page Content]"

class ToolRegistry:
    def __init__(self):
        self.tools = [CalculatorTool(), WikiSearchTool()]
        
    def find_tool(self, text: str) -> Optional[AgentTool]:
        """Simple keyword matching."""
        text_lower = text.lower()
        for t in self.tools:
            for trig in t.triggers:
                if trig in text_lower:
                    return t
        return None

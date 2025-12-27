"""
Aurora Agent Base Architecture.
===============================

Defines the abstract interfaces for the four-layer agent:
1. Perception: Text -> Context (Anchors, Intent)
2. Planner: Context -> Path (Geometric Trajectory)
3. Verifier: Path -> VerificationResult (Safe/Unsafe)
4. Actor: Path -> Response (Natural Language)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import torch

@dataclass
class AgentContext:
    """
    The shared state object passed through the pipeline.
    """
    raw_text: str
    anchors: Dict[str, int] = field(default_factory=dict) # {"Word": ID}
    intent: str = "statement" # statement, question, clarification
    focus_J: Optional[torch.Tensor] = None # Focus Vector (J)
    tool_needed: Optional[Any] = None # AgentTool instance if detected
    
@dataclass
class AgentPath:
    """
    The geometric plan produced by the Planner.
    """
    steps: List[int] = field(default_factory=list) # Sequence of Node IDs
    scores: List[float] = field(default_factory=list) # Step-wise scores
    energy_cost: float = 0.0
    holonomy_error: float = 0.0
    
@dataclass
class VerificationResult:
    """
    Output of the Verifier.
    """
    is_safe: bool
    reason: str = ""
    clarification_needed: bool = False

class Perception(ABC):
    @abstractmethod
    def perceive(self, text: str) -> AgentContext:
        """Parse user input into geometric context."""
        pass

class Planner(ABC):
    @abstractmethod
    def plan(self, context: AgentContext) -> AgentPath:
        """Generate a geometric path based on context."""
        pass

class Verifier(ABC):
    @abstractmethod
    def verify(self, path: AgentPath, context: AgentContext) -> VerificationResult:
        """Check if the plan is safe and coherent."""
        pass

class Actor(ABC):
    @abstractmethod
    def act(self, path: AgentPath, context: AgentContext) -> str:
        """Convert the plan into a natural language response."""
        pass

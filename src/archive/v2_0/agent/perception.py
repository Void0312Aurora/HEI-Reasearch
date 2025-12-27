"""
Aurora Agent Perception Layer.
==============================

Maps Natural Language -> Geometric Anchors.
"""

from typing import List, Dict, Optional
import torch
import numpy as np
from .base import Perception, AgentContext
from ..data import AuroraDataset
from ..ingest.wikipedia import WikipediaIngestor

class GeometricPerception(Perception):
    def __init__(self, ds: AuroraDataset, ingestor: WikipediaIngestor, J: torch.Tensor):
        self.ds = ds
        self.ingestor = ingestor
        self.J = J # (N, k)
        
    def perceive(self, text: str) -> AgentContext:
        """
        1. Tokenize text using Ingestor (Jieba).
        2. Resolve tokens to IDs.
        3. Identify Intent (Heuristic).
        4. Construct Context.
        """
        ctx = AgentContext(raw_text=text)
        
        # 1. Tokenize & Resolve
        tokens = self.ingestor.tokenize(text)
        anchors = {}
        
        for t in tokens:
            idx = self.ingestor.resolve_token(t)
            if idx is not None:
                # Store Map: "Original" -> ID
                # Note: One ID might map to multiple original tokens, we keep last or first?
                # Let's keep unique tokens.
                anchors[t] = idx
                
        ctx.anchors = anchors
        
        # 2. Heuristic Intent Detection
        # Enhanced Rules
        lower_text = text.lower()
        if any(w in lower_text for w in ["?", "？", "what", "什么", "who", "谁"]):
             ctx.intent = "question"
        elif any(w in lower_text for w in ["hello", "hi", "你好", "您好", "hey"]):
             ctx.intent = "greeting"
        elif not anchors and not ctx.tool_needed:
             # If no anchors and no tools, likely chitchat or failed parse
             ctx.intent = "unknown"
        else:
             ctx.intent = "statement"
            
        # 3. Determine Focus J
        # Strategy: Mean of Anchors? Or Last Anchor?
        # Dialogue logic usually focuses on the 'New' information, often at the end.
        if anchors:
            # Get last valid ID
            last_id = list(anchors.values())[-1]
            ctx.focus_J = self.J[last_id].clone()
        else:
            # No anchors found?
            pass

        # 4. Tool Detection
        from .tools import ToolRegistry
        registry = ToolRegistry()
        tool = registry.find_tool(text)
        if tool:
            ctx.tool_needed = tool
            
        return ctx

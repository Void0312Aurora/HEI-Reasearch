"""
Aurora Agent Loop.
==================

The "Brain" that coordinates Perception, Planning, Verification, and Acting.
"""

import sys
import os
import torch
import numpy as np
import pickle
import logging

from ..data import AuroraDataset
from ..gauge import GaugeField, NeuralBackend
from ..ingest.wikipedia import WikipediaIngestor
from ..generation.decoder import RiemannianBeamSearchDecoder

from .base import AgentContext, AgentPath, VerificationResult
from .perception import GeometricPerception
from .planner import GeometricPlanner
from .verifier import GeometricVerifier
from .actor import GeometricActor

class AurorAgent:
    def __init__(self, checkpoint_path: str, device='cuda', log_level=logging.INFO):
        self.device = device
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger("AurorAgent")
        
        self.logger.info("Initializing AurorAgent...")
        
        # 1. Load Checkpoint & Physics
        self._load_physics(checkpoint_path)
        
        # 2. Components
        self.ingestor = WikipediaIngestor(self.ds)
        self.ingestor.resolve_token = self._resolve_token_wrapper # Patch ingestor with dataset awareness
        
        self.decoder = RiemannianBeamSearchDecoder(self.gauge_field, self.x, self.J, self.ds, device=device)
        
        # 3. Agent Layers
        self.perception = GeometricPerception(self.ds, self.ingestor, self.J)
        self.planner = GeometricPlanner(self.decoder)
        self.verifier = GeometricVerifier(self.gauge_field, self.x)
        self.actor = GeometricActor(self.ds)
        
        self.logger.info("Agent Ready.")
        
    def _resolve_token_wrapper(self, token):
        # Helper to inject raw_map logic into ingestor if not present
        
        # 1. Exact Match (ID check) -> Not applicable, token is str
        
        # 2. Vocabulary Match (word_to_id)
        if token in self.ds.vocab.word_to_id:
            return self.ds.vocab.word_to_id[token]
            
        # 3. Raw Map Match (raw_map)
        if hasattr(self.ds, 'raw_map') and token in self.ds.raw_map:
             return self.ds.raw_map[token]
             
        # 4. Fuzzy Substring Match (Slow but friendly)
        # Check if token is part of any node name?
        # Only feasible if vocab is small. 14k is small enough.
        # Priority: Exact match of substring
        for word, idx in self.ds.vocab.word_to_id.items():
            # Check if token is in Cilin Code e.g. "C:Plant:01"
            if token in word:
                return idx
                
        return None

    def _load_physics(self, checkpoint_path):
        import io
        target_device = self.device # Capture for closure
        
        class DeviceUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location=target_device)
                return super().find_class(module, name)
                
        with open(checkpoint_path, 'rb') as f:
            ckpt = DeviceUnpickler(f).load()
            
        self.x = torch.tensor(ckpt['x'], device=self.device)
        self.J = torch.tensor(ckpt['J'], device=self.device)
        
        # Dataset
        # Warning: Checkpoint doesn't save ds params efficiently.
        # We assume Cilin for now.
        self.ds = AuroraDataset("cilin", limit=self.x.shape[0]) 
        # Note: limit should match x shape. If x is 14345, limit=None or inferred.
        # AuroraDataset loader might trim. 
        # Ideally we trust ds to load same data.
        
        # Gauge Field - NeuralBackend
        self.gauge_field = GaugeField(torch.zeros((1,2), dtype=torch.long, device=self.device), 3, group='SO', backend_type='neural', input_dim=5).to(self.device)
        self.gauge_field.backend = NeuralBackend(input_dim=5, logical_dim=3, num_relations=200, relation_dim=8).to(self.device)
        
        filtered = {k:v for k,v in ckpt['gauge_field'].items() if not k.startswith(('edges', 'tri'))}
        self.gauge_field.load_state_dict(filtered, strict=False)
        
    def process(self, text: str) -> str:
        """
        Main Loop: Perceive -> Plan -> Verify -> Act.
        """
        self.logger.info(f"User Input: {text}")
        
        # 1. Perception
        context = self.perception.perceive(text)
        self.logger.info(f"Context: Intent={context.intent}, Anchors={list(context.anchors.keys())}")
        
        if not context.anchors and not context.tool_needed:
            # Check for Phatic Intents before giving up
            if context.intent in ["greeting", "unknown"]:
                # Pass to Actor to handle "Hello" or "I don't understand"
                pass
            else:
                return "I couldn't map any words to my geometric concepts."
            
        # 1.5. Tool Execution
        if context.tool_needed:
            tool = context.tool_needed
            self.logger.info(f"Triggering Tool: {tool.name}")
            result = tool.execute(text)
            self.logger.info(f"Tool Result: {result}")
            
            # For Phase 31: Return Tool Result Directly (Agent as Interface)
            # Or feed back into Planner?
            # Geometric Planner can't reason about "2" easily unless "2" is in Cilin.
            return f"[Tool: {tool.name}] {result}"
            
        # 2. Planning
        path = self.planner.plan(context)
        self.logger.info(f"Plan Generated: {len(path.steps)} steps.")
        
        # 3. Verification
        check = self.verifier.verify(path, context)
        if not check.is_safe:
            self.logger.warning(f"Verification Failed: {check.reason}")
            if check.clarification_needed:
                 return f"I'm confused. {check.reason}. Can you clarify?"
            return "I couldn't verify the reasoning path."
            
        # 4. Action
        response = self.actor.act(path, context)
        return response

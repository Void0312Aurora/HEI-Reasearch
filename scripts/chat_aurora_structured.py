"""
Aurora Structured Chat Demo (Phase I).
======================================

Demonstrates the structured dialogue capabilities:
1. Input -> Intention Mapping (ConceptMapper)
2. Skeleton-Based Expansion (InteractionEngine)
3. Template Realization (LanguageRealizer)

Usage:
    python scripts/chat_aurora_structured.py
"""

import sys
import os
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from hei_n.interaction_engine import InteractionEngine
from hei_n.language_realizer import LanguageRealizer

def main():
    print("Initializing Aurora Interaction Engine (Phase I: Structured)...")
    
    checkpoint_path = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    engine = InteractionEngine(
        checkpoint_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        observation_strength=50.0  # Strong anchor for structured output
    )
    
    realizer = LanguageRealizer(style="template")
    
    print("\nAurora Online. (Type 'quit' to exit)")
    print("System: Phase I Structured Dialogue (Skeleton-First + Template)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except EOFError:
            break
            
        if user_input.lower() in ('quit', 'exit'):
            break
            
        if not user_input:
            continue
            
        # Process
        # 1. Engine Thought
        concepts_with_scores = engine.process_input(user_input, steps=200, top_k=5)
        
        # Extract words
        concept_words = [w for w, s in concepts_with_scores]
        
        # 2. Realization
        response = realizer.realize(concept_words, input_text=user_input)
        
        print(f"Aurora: {response}")
        # Debug: Show activated concepts
        # print(f"  [Debug] Activated: {concept_words}")

if __name__ == "__main__":
    main()

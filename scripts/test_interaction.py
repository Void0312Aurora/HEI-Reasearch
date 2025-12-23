#!/usr/bin/env python
"""
Non-interactive test for Aurora Interaction Engine (V2).
Uses concepts that definitely exist in the vocab.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.interaction_engine import InteractionEngine
from hei_n.language_realizer import LanguageRealizer


def main():
    print("=" * 50)
    print("Aurora Interaction Engine - Test V2")
    print("=" * 50)
    
    checkpoint = "checkpoints/aurora_base_gpu_100000.pkl"
    
    print(f"\nLoading checkpoint: {checkpoint}")
    engine = InteractionEngine(
        checkpoint_path=checkpoint,
        device='cuda',
        gamma=0.3,  # Lower damping
        activation_threshold=1.0,  # Looser threshold
        refractory_period=20,
        neighbor_k=512,
    )
    
    realizer = LanguageRealizer(style="simple")
    
    # Test with words that definitely exist
    test_inputs = [
        "food",       # Sememe
        "animal",     # Sememe
        "human",      # Sememe
        "happy",      # Concept
        "water",      # Concept
        "love",       # Concept
    ]
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    for test_input in test_inputs:
        print(f"\n[Input] {test_input}")
        
        engine.reset()
        concepts = engine.process_input(test_input, steps=200)  # More steps
        
        if concepts:
            response = realizer.realize(concepts)
            print(f"[Output] {response}")
            print(f"         (Activated {len(concepts)} concepts: {concepts[:5]}...)")
        else:
            print(f"[Output] (No activation)")
        
    print("\n" + "=" * 50)
    print("Test Complete.")


if __name__ == "__main__":
    main()

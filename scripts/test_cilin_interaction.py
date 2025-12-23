#!/usr/bin/env python
"""
Test Interaction Engine with Cilin Dataset.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.interaction_engine import InteractionEngine
from hei_n.language_realizer import LanguageRealizer


def main():
    print("=" * 60)
    print("Aurora Interaction Engine - Cilin Migration Test")
    print("=" * 60)
    
    checkpoint = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    
    print(f"\nLoading checkpoint: {checkpoint}")
    engine = InteractionEngine(
        checkpoint_path=checkpoint,
        device='cuda',
        gamma=0.8,
        activation_threshold=1.0,
        refractory_period=20,
        neighbor_k=512,
        observation_strength=10.0,
        temperature=0.05,
    )
    
    realizer = LanguageRealizer(style="simple")
    
    # Test cases (likely to exist in Cilin)
    test_inputs = [
        "人",   # Person
        "生物", # Organism
        "动物", # Animal
        "食物", # Food
        "吃",   # Eat
        "love", # Should map to 爱 via lexicon
    ]
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_input in test_inputs:
        print(f"\n[Input] {test_input}")
        
        # Check mapping first
        ids = engine.mapper.text_to_particles(test_input)
        if not ids:
            print(f"  Warning: '{test_input}' not found in vocab.")
            continue
        print(f"  Mapped to: {ids} ({engine.mapper.particles_to_text(ids)})")
            
        engine.reset()
        concepts = engine.process_input(test_input, steps=200, top_k=10)
        
        if concepts:
            response = realizer.realize(concepts[:5])
            print(f"[Output] {response}")
            print(f"         (Top 10: {concepts})")
        else:
            print(f"[Output] (No activation)")
        
    print("\n" + "=" * 60)
    print("Test Complete.")


if __name__ == "__main__":
    main()

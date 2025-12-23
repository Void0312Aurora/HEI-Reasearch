#!/usr/bin/env python
"""
Non-interactive test for Aurora Interaction Engine (V3).
Tests the fixed I/O alignment from temp-08.md.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.interaction_engine import InteractionEngine
from hei_n.language_realizer import LanguageRealizer


def main():
    print("=" * 60)
    print("Aurora Interaction Engine - Test V3 (Fixed I/O Alignment)")
    print("=" * 60)
    
    checkpoint = "checkpoints/aurora_base_gpu_100000.pkl"
    
    print(f"\nLoading checkpoint: {checkpoint}")
    engine = InteractionEngine(
        checkpoint_path=checkpoint,
        device='cuda',
        gamma=0.8,  # Higher damping
        activation_threshold=1.0,
        refractory_period=20,
        neighbor_k=512,
        observation_strength=10.0,  # Strong anchor
        temperature=0.05,  # Low noise
    )
    
    realizer = LanguageRealizer(style="simple")
    
    # Test with both Chinese and English (English will be translated)
    test_inputs = [
        # English (will be translated to Chinese via lexicon)
        "food",
        "animal",
        "love",
        "happy",
        # Chinese (direct)
        "食物",
        "动物",
        "爱",
        "快乐",
    ]
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for test_input in test_inputs:
        print(f"\n[Input] {test_input}")
        
        engine.reset()
        concepts = engine.process_input(test_input, steps=200, top_k=10)
        
        if concepts:
            response = realizer.realize(concepts[:5])  # Top 5
            print(f"[Output] {response}")
            print(f"         (Top 10: {concepts})")
        else:
            print(f"[Output] (No activation)")
        
    print("\n" + "=" * 60)
    
    # Sanity Check: Test anchor neighborhood directly
    print("\n=== Anchor Neighborhood Sanity Check ===")
    test_word = "食物"  # Food in Chinese
    ids = engine.mapper.text_to_particles(test_word)
    if ids:
        print(f"'{test_word}' maps to particle ID: {ids}")
        # Get direct neighbors (without cursor dynamics)
        _, neighbors = engine.force_field.get_neighbors(
            engine.positions_tensor[ids[0]], k=20
        )
        neighbor_words = engine.mapper.particles_to_text(list(_))
        print(f"Direct neighbors: {neighbor_words[:10]}")
    else:
        print(f"'{test_word}' not found in vocab")
        
    print("\n" + "=" * 60)
    print("Test Complete.")


if __name__ == "__main__":
    main()

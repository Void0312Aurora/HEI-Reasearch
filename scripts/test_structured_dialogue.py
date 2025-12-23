"""
Aurora Structured Dialogue Test (Non-Interactive).
==================================================

Tests Phase I structure:
1. "生物" -> Category Expansion
2. "水果" -> Category Expansion
3. "我想要开心" -> Intention Mapping (Gate M) + Template Output

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

def test_query(engine, realizer, text):
    print(f"\n[Test Input] {text}")
    print("-" * 30)
    
    # 1. Process
    concepts_with_scores = engine.process_input(text, steps=200, top_k=5)
    print(f"DEBUG: concepts_with_scores type: {type(concepts_with_scores)}")
    if concepts_with_scores:
        print(f"DEBUG: first item: {concepts_with_scores[0]}")
    
    # 2. Extract words
    concept_words = [item[0] for item in concepts_with_scores]
    print(f"  Activated: {concept_words}")
    
    # 3. Realize
    response = realizer.realize(concept_words, input_text=text)
    print(f"  Aurora: {response}")
    
def main():
    print("Initializing Aurora Interaction Engine (Phase I Test)...")
    
    checkpoint_path = "checkpoints/aurora_base_gpu_cilin_full.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    engine = InteractionEngine(
        checkpoint_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        observation_strength=50.0 
    )
    
    realizer = LanguageRealizer(style="template")
    
    test_query(engine, realizer, "生物")
    test_query(engine, realizer, "水果")
    test_query(engine, realizer, "我想要开心")
    test_query(engine, realizer, "love") # Check English template

if __name__ == "__main__":
    main()

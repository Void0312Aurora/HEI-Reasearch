"""
Prepare Entropy Stats for CCD v3.1.
===================================

Scans CLUECorpusSmall.txt and generating 'entropy_stats.json'.
This defines the radial targets for the Injector.

Ref: Axiom 2.1.2.
"""

import sys
import os
import json
import math
from collections import Counter
from tqdm import tqdm

DATA_PATH = "data/CLUE/CLUECorpusSmall.txt"
OUTPUT_PATH = "data/entropy_stats.json"

def main():
    print(f"Scanning {DATA_PATH}...")
    
    # 1. First Pass: Count
    counts = Counter()
    total_chars = 0
    
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        # Fallback to creating a dummy or simple vocab if file missing?
        # But we saw it in finding.
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        # Limit to first 200k lines for speed if needed
        for i, line in tqdm(enumerate(f)):
            if i > 200000: break
            
            line = line.strip()
            # Simple Tokenization (Char level)
            for char in line:
                if char.strip(): # Skip whitespace
                    counts[char] += 1
                    total_chars += 1
                    
    print(f"Total Chars: {total_chars}")
    print(f"Vocab Size: {len(counts)}")
    
    # 2. Compute Prob and Radius
    # High Freq -> Core -> r=0.
    # Low Freq -> Boundary -> r=1.
    
    stats = {}
    
    if total_chars == 0:
        print("Empty corpus!")
        return

    min_prob = min(counts.values()) / total_chars
    max_prob = max(counts.values()) / total_chars
    
    min_H = -math.log(max_prob + 1e-9)
    max_H = -math.log(min_prob + 1e-9)
    
    print(f"Entropy Range: [{min_H:.2f}, {max_H:.2f}]")
    
    for char, count in counts.items():
        p = count / total_chars
        h = -math.log(p + 1e-9)
        
        # Linear Map H -> r
        denom = max_H - min_H if max_H > min_H else 1.0
        r = (h - min_H) / denom
        r = max(0.0, min(0.99, r)) # Clip to Ball
        
        stats[char] = {
            "count": count,
            "prob": p,
            "entropy": h,
            "r": r
        }
        
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    print(f"Saved stats to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

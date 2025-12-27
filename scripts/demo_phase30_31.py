"""
Phase 30-31: Enhanced Actor & Tool Demo.
========================================

Verifies:
1. Natural Language Generation (Actor Templates).
2. Tool Execution (Calculator, WikiSearch).
"""

import sys
import os
import argparse
import logging
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from aurora.agent.agent_loop import AurorAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()
    
    print("--- Loading Agent ---")
    # Silence init logs
    agent = AurorAgent(args.checkpoint, log_level=logging.WARNING)
    
    tests = [
        # Phase 30: NLG
        "What is 植物 (plant)?", # Should use 'Core concept' template
        "Link C:人:01 (person) and C:吃:07 (eat)", # Should use 'Connected' template
        
        # Phase 31: Tools
        "Calculate 12 * 12",
        "Who is Elon Musk?",
    ]
    
    for t in tests:
        print(f"\nUser: {t}")
        try:
            res = agent.process(t)
            print(f"Agent: {res}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

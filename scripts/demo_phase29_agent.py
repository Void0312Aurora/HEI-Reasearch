"""
Phase 29: Agentic Integration Demo.
===================================

Runs the AurorAgent Loop on test cases.
"""

import sys
import os
import argparse
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aurora.agent.agent_loop import AurorAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    
    agent = AurorAgent(args.checkpoint, log_level=logging.INFO)
    
    if args.interactive:
        print("AurorAgent Ready. Type 'exit' to quit.")
        while True:
            text = input("\nUser: ")
            if text.lower() == 'exit': break
            try:
                response = agent.process(text)
                print(f"Agent: {response}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Hardcoded Tests
        tests = [
            "What is 植物 (plant)?", # Should find C:植物:25
            "Link C:人:01 (person) and C:吃:07 (eat)", # Explicit link
        ]
        
        for t in tests:
            print(f"\n--- Test: '{t}' ---")
            try:
                response = agent.process(t)
                print(f"Agent Response: {response}")
            except Exception as e:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()

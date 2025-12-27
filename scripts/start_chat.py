"""
AurorAgent Interactive Chat.
============================

Run this script to chat with the Geometric Agent.
Features:
- Natural Language Understanding (Perception)
- Geometric Reasoning (Planner)
- Tool Use (Calculator, Search)
- Template-based Response Generation (Actor)
"""

import sys
import os
import argparse
import logging
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from aurora.agent.agent_loop import AurorAgent

def main():
    parser = argparse.ArgumentParser(description="Chat with AurorAgent")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--debug", action="store_true", help="Show debug logs")
    args = parser.parse_args()
    
    level = logging.DEBUG if args.debug else logging.WARNING
    
    print("\nInitializing Aurora Geometric Agent...")
    print(f"Loading Physics from: {args.checkpoint}")
    
    try:
        agent = AurorAgent(args.checkpoint, log_level=level)
    except Exception as e:
        print(f"Error loading agent: {e}")
        return

    print("\n" + "="*50)
    print("AURORA: Online. (Type 'exit' or 'quit' to stop)")
    print("Capabilities: Geometric Reasoning, Math, Fact Search.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("AURORA: Shutting down.")
                break
            
            if not user_input.strip():
                continue
                
            # Process
            response = agent.process(user_input)
            print(f"Aurora: {response}\n")
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"System Error: {e}")

if __name__ == "__main__":
    main()

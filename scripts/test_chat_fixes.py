"""
Verify Chat Fixes (Phase 30b).
==============================

Simulates user inputs that failed previously.
"""
import sys
import os
import io
import argparse
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from aurora.agent.agent_loop import AurorAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()
    
    agent = AurorAgent(args.checkpoint, log_level=logging.WARNING)
    
    # Failure Cases from User Feedback
    cases = [
        "你好",          # Should get greeting
        "Hello?",        # Should get greeting
        "今天晚上去吃烧烤",  # Should map '烧烤' or at least '吃'. 
                         # And Actor should handle single/double concepts gracefully.
    ]
    
    for c in cases:
        print(f"\nUser: {c}")
        res = agent.process(c)
        print(f"Agent: {res}")
        
if __name__ == "__main__":
    main()

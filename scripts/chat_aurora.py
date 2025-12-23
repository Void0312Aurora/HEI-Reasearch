#!/usr/bin/env python
"""
Aurora Chat Demo (CLI).
=======================

Interactive CLI for testing Aurora Interaction Engine.
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hei_n.interaction_engine import InteractionEngine
from hei_n.language_realizer import LanguageRealizer


def main(args):
    print("=" * 50)
    print("Aurora Interaction Engine - Chat Demo")
    print("=" * 50)
    print()
    
    # Initialize engine
    print(f"Loading checkpoint: {args.checkpoint}")
    engine = InteractionEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        gamma=args.gamma,
        activation_threshold=args.threshold,
        refractory_period=args.refractory,
        neighbor_k=args.neighbors,
    )
    
    # Initialize realizer
    realizer = LanguageRealizer(style="simple")
    
    print()
    print("Aurora Base loaded. Enter your message (type 'quit' to exit).")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n[You] > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if user_input.lower() == 'reset':
                engine.reset()
                print("[System] Cursor reset to origin.")
                continue
                
            if user_input.lower() == 'history':
                history = engine.get_activation_history()
                print(f"[History] {' -> '.join(history) if history else '(empty)'}")
                continue
                
            # Process input
            concepts = engine.process_input(user_input, steps=args.steps)
            
            # Realize to language
            response = realizer.realize(concepts)
            
            print(f"[Aurora] {response}")
            
            if args.verbose:
                print(f"         (Activated: {', '.join(concepts)})")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"[Error] {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora Chat Demo")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="checkpoints/aurora_base_gpu_100000.pkl",
        help="Path to Aurora Base checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps per input")
    parser.add_argument("--gamma", type=float, default=0.5, help="Damping coefficient")
    parser.add_argument("--threshold", type=float, default=0.5, help="Activation distance threshold")
    parser.add_argument("--refractory", type=int, default=50, help="Refractory period (steps)")
    parser.add_argument("--neighbors", type=int, default=256, help="KNN neighbors")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    main(args)

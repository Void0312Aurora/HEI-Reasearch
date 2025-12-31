import os
import subprocess
import sys
import argparse

def run_script(path, args=None):
    env = os.environ.copy()
    # Add HEI directory to PYTHONPATH
    hei_path = os.path.abspath("HEI")
    env["PYTHONPATH"] = hei_path + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [sys.executable, path]
    if args:
        cmd.extend(args)
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        print(f"ERRORS:\n{result.stderr}", file=sys.stderr)
    return result

def main():
    print("====================================================")
    print("   HEI Research Audit: A1-A5 Axiom Verification     ")
    print("====================================================")
    
    # Paths
    exp_dir = "HEI/EXP"
    tests_dir = "HEI/tests/verify_generators"
    
    scripts_to_run = [
        ("A2/A3 Audit: Offline Cognition", os.path.join(exp_dir, "verify_offline_cognition.py")),
        ("A3/A4 Audit: Holonomy & Hysteresis", os.path.join(exp_dir, "verify_holonomy.py")),
        ("L2 Audit: Geometric Curvature", os.path.join(exp_dir, "verify_geometric_curvature.py")),
        ("L1 Audit: Semigroup Stability", os.path.join(tests_dir, "verify_semigroup_stability.py")),
    ]
    
    success_count = 0
    total = len(scripts_to_run)
    
    for name, path in scripts_to_run:
        print(f"\n[SECTION: {name}]")
        if not os.path.exists(path):
            print(f"SKIPPING: {path} not found.")
            continue
            
        res = run_script(path)
        if res.returncode == 0:
            success_count += 1
            print(f"✓ {name} completed successfully.")
        else:
            print(f"✗ {name} failed with return code {res.returncode}.")
            
    print("\n" + "="*50)
    print(f"RESEARCH AUDIT SUMMARY: {success_count}/{total} PASSED")
    print("="*50)
    
    if success_count == total:
        print("\nCONCLUSION: The current implementation aligns with Theoretical Foundation-7 (A1-A5).")
    else:
        print("\nCONCLUSION: Some theoretical axioms failed verification. Check logs for details.")

if __name__ == "__main__":
    main()

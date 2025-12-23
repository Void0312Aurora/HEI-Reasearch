#!/usr/bin/env python
"""Minimal test to verify logging works."""
import sys
import time

print("TEST 1: Script started", flush=True)
sys.stdout.write("TEST 2: sys.stdout.write\n")
sys.stdout.flush()
sys.stderr.write("TEST 3: sys.stderr.write\n")
sys.stderr.flush()

time.sleep(2)
print("TEST 4: After sleep", flush=True)
print("TEST COMPLETE")

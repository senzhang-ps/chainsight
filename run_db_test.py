# -*- coding: utf-8 -*-
"""Test script for DB mode simulation"""
import sys
import io
import os

# Set UTF-8 encoding for stdout/stderr
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    from src.core.run import main
    
    # Run with DB mode
    args = [
        "--config", "BC_S5",
        "--start-date", "2025-10-06",
        "--end-date", "2025-10-10",
        "--use-db",
        "--force-restart"
    ]
    
    print("=" * 70)
    print("Starting DB Mode Simulation Test")
    print("=" * 70)
    print(f"Arguments: {args}")
    print()
    
    try:
        exit_code = main(args)
        print(f"\nExit code: {exit_code}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

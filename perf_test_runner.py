"""
Performance Test Runner - Run all three modes and capture timing
OC_Paste_S1_20251224, 76 days (2025-12-15 to 2026-02-28)

This script runs sequentially:
  1. Refactored DB mode (already running separately)
  2. Refactored File mode
  3. Dev File mode
And captures wall-clock timing for each.

Usage: python perf_test_runner.py [--skip-db] [--skip-file] [--skip-dev]
"""
import subprocess, sys, time, json, os
from datetime import datetime
from pathlib import Path

PYTHON = r'D:\PG\Code\chainsight\.venv\Scripts\python.exe'
PROJECT = r'D:\PG\Code\chainsight'
RESULTS_FILE = os.path.join(PROJECT, 'perf_test_results.json')

def run_timed(label, cmd, cwd, timeout_hours=6):
    """Run a command and return (success, elapsed_seconds, output_path)"""
    print(f"\n{'='*70}")
    print(f"[START] {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  CWD: {cwd}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout_hours * 3600,
            capture_output=False,  # let output flow to console
        )
        elapsed = time.time() - start
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        success = False
        print(f"[TIMEOUT] {label} exceeded {timeout_hours} hours!")
    except Exception as e:
        elapsed = time.time() - start
        success = False
        print(f"[ERROR] {label}: {e}")
    
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(mins)}m {secs:.1f}s"
    
    status = 'SUCCESS' if success else 'FAILED'
    print(f"\n[{status}] {label}: {time_str} ({elapsed:.1f}s)")
    
    return {
        'label': label,
        'success': success,
        'elapsed_seconds': round(elapsed, 1),
        'elapsed_human': time_str,
        'start_time': datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-db', action='store_true')
    parser.add_argument('--skip-file', action='store_true')
    parser.add_argument('--skip-dev', action='store_true')
    args = parser.parse_args()
    
    results = []
    
    # ===== 1. Refactored DB mode =====
    if not args.skip_db:
        r = run_timed(
            "Refactored DB Mode (76 days)",
            [PYTHON, '-X', 'utf8', 'run.py',
             '--config', 'OC_Paste_S1_20251224',
             '--start-date', '2025-12-15',
             '--end-date', '2026-02-28',
             '--use-db',
             '--non-interactive'],
            cwd=PROJECT
        )
        results.append(r)
    
    # ===== 2. Refactored File mode =====
    if not args.skip_file:
        r = run_timed(
            "Refactored File Mode (76 days)",
            [PYTHON, '-X', 'utf8', 'run.py',
             '--config', os.path.join('config', 'OC_Paste_S1_20251224.xlsx'),
             '--start-date', '2025-12-15',
             '--end-date', '2026-02-28',
             '--non-interactive'],
            cwd=PROJECT
        )
        results.append(r)
    
    # ===== 3. Dev File mode =====
    if not args.skip_dev:
        r = run_timed(
            "Dev File Mode (76 days)",
            [PYTHON, '-X', 'utf8', 'run.py',
             '--config', os.path.join('..', 'config', 'OC_Paste_S1_20251224.xlsx'),
             '--start-date', '2025-12-15',
             '--end-date', '2026-02-28',
             '--non-interactive'],
            cwd=os.path.join(PROJECT, 'ChainSight_Dev')
        )
        results.append(r)
    
    # ===== Save results =====
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("PERFORMANCE TEST COMPLETE")
    print(f"{'='*70}")
    for r in results:
        status = 'OK' if r['success'] else 'FAIL'
        print(f"  [{status}] {r['label']}: {r['elapsed_human']}")
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == '__main__':
    main()

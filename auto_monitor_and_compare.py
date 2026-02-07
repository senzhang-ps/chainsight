"""
Auto-monitor both runs, then run comparison when both complete.
Run with: python -X utf8 auto_monitor_and_compare.py
"""
import os, sys, time, subprocess, re
from datetime import datetime

DEV_DIR = r'D:\PG\Code\chainsight\config\OC_Paste_S1_20251224\run_20260127_142402'
REF_DIR = r'D:\PG\Code\chainsight\outputs\OC_Paste_S1_20251224\run_20260206_234914'
DB_LOG = r'D:\PG\Code\chainsight\perf_db_run_stdout.log'
FILE_LOG = r'D:\PG\Code\chainsight\outputs\OC_Paste_S1_20251224\run_20260206_234914\simulation_log_20260206_234914.txt'
COMPARE_SCRIPT = r'D:\PG\Code\chainsight\compare_76day_perf.py'
PYTHON = r'D:\PG\Code\chainsight\.venv\Scripts\python.exe'
STATUS_FILE = r'D:\PG\Code\chainsight\monitor_status.txt'
COMPARE_OUTPUT = r'D:\PG\Code\chainsight\comparison_results.txt'

TOTAL_DAYS = 76
CHECK_INTERVAL = 60  # seconds

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(STATUS_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def get_day_progress(log_file):
    """Get latest day number from log file"""
    if not os.path.exists(log_file):
        return 0
    try:
        with open(log_file, encoding='utf-8') as f:
            content = f.read()
        matches = re.findall(r'第 (\d+) 天处理完成', content)
        if matches:
            return int(matches[-1])
    except Exception:
        pass
    return 0

def is_run_complete(log_file):
    """Check if a run has completed (day 76 done)"""
    return get_day_progress(log_file) >= TOTAL_DAYS

def check_file_mode_complete():
    """Also check if the last day file exists"""
    last_file = os.path.join(REF_DIR, 'module1', 'module1_output_20260228.xlsx')
    return os.path.exists(last_file)

def check_python_running():
    """Check if any python processes from our runs are still alive"""
    try:
        result = subprocess.run(
            ['powershell', '-Command', 'Get-Process python -ErrorAction SilentlyContinue | Select-Object Id'],
            capture_output=True, text=True, timeout=10
        )
        return 'python' in result.stdout.lower() or any(c.isdigit() for c in result.stdout)
    except Exception:
        return False

def run_comparison():
    """Run the 76-day comparison script"""
    log("Starting 76-day comparison...")
    cmd = [
        PYTHON, '-X', 'utf8', COMPARE_SCRIPT,
        '--dev-dir', DEV_DIR,
        '--ref-dir', REF_DIR
    ]
    log(f"Command: {' '.join(cmd)}")
    
    with open(COMPARE_OUTPUT, 'w', encoding='utf-8') as outf:
        result = subprocess.run(
            cmd, 
            capture_output=False,
            stdout=outf,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200,  # 2 hour timeout
            cwd=r'D:\PG\Code\chainsight'
        )
    
    log(f"Comparison exit code: {result.returncode}")
    
    # Print last 30 lines of output
    with open(COMPARE_OUTPUT, encoding='utf-8') as f:
        lines = f.readlines()
    log("=== Comparison Results (last 30 lines) ===")
    for line in lines[-30:]:
        log(line.rstrip())
    
    return result.returncode == 0

def main():
    log("=" * 60)
    log("Auto-monitor started")
    log(f"Checking every {CHECK_INTERVAL}s for completion of both runs")
    log("=" * 60)
    
    while True:
        db_day = get_day_progress(DB_LOG)
        file_day = get_day_progress(FILE_LOG)
        db_done = db_day >= TOTAL_DAYS
        file_done = file_day >= TOTAL_DAYS or check_file_mode_complete()
        
        log(f"DB mode: day {db_day}/{TOTAL_DAYS} {'[DONE]' if db_done else ''} | File mode: day {file_day}/{TOTAL_DAYS} {'[DONE]' if file_done else ''}")
        
        if db_done and file_done:
            log("Both runs complete! Starting comparison...")
            # Wait a bit for any final file writes
            time.sleep(10)
            ok = run_comparison()
            if ok:
                log("ALL CHECKS PASSED - ready for report generation")
            else:
                log("DIFFERENCES FOUND - check comparison_results.txt")
            break
        
        time.sleep(CHECK_INTERVAL)
    
    log("Monitor finished")

if __name__ == '__main__':
    main()

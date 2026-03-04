#!/usr/bin/env python3
"""合并分module运行的OC三版本比对结果为完整JSON。

由于module5数据量大，全量运行会超时，因此采用分module运行后合并的方式。
每个module的结果先存为临时文件，然后合并为最终的oc_3way_content_results_full.json。
"""
import json, sys, subprocess, time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = str(PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe")
SCRIPT = str(PROJECT_ROOT / "tools" / "oc_3way_content_compare.py")
FINAL_OUTPUT = PROJECT_ROOT / "tools" / "oc_3way_content_results_full.json"

# module5 需要分段运行
TASKS = [
    {"module": "module1", "days": None},
    {"module": "module3", "days": None},
    {"module": "module4", "days": None},
    {"module": "module5", "days": "1-20"},
    {"module": "module5", "days": "21-40"},
    {"module": "module5", "days": "41-60"},
    {"module": "module5", "days": "61-76"},
    {"module": "module6", "days": None},
]

def run_module(module: str, days: str | None, output_path: str) -> dict:
    cmd = [PYTHON, SCRIPT, "--module", module, "--quiet", "--output", output_path]
    if days:
        cmd.extend(["--days", days])
    print(f"Running: {module}" + (f" days={days}" if days else "") + " ...", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s", flush=True)
    if result.stdout:
        # Print just the summary table
        for line in result.stdout.split("\n"):
            if "PASS" in line or "FAIL" in line:
                print(f"  {line.strip()}", flush=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}", flush=True)
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_module5_results(results_list: list[dict]) -> dict:
    """合并module5分段运行的结果（按sheet合并daily列表）。"""
    merged = {}
    for r in results_list:
        for comp_key, data in r.get("comparisons", {}).items():
            if comp_key not in merged:
                merged[comp_key] = {
                    "daily": [],
                    "dev_vs_src_all_match": True,
                    "dev_vs_db_all_match": True,
                    "src_vs_db_all_match": True,
                }
            merged[comp_key]["daily"].extend(data.get("daily", []))
            if not data.get("dev_vs_src_all_match", True):
                merged[comp_key]["dev_vs_src_all_match"] = False
            if not data.get("dev_vs_db_all_match", True):
                merged[comp_key]["dev_vs_db_all_match"] = False
            if not data.get("src_vs_db_all_match", True):
                merged[comp_key]["src_vs_db_all_match"] = False

    # Recompute totals
    for comp_key, data in merged.items():
        daily = data["daily"]
        data["total_dev_rows"] = sum(d["dev_rows"] for d in daily)
        data["total_src_rows"] = sum(d["src_rows"] for d in daily)
        data["total_db_rows"] = sum(d["db_rows"] for d in daily)
        data["total_dev_src_diffs"] = sum(d["dev_vs_src"]["diff_count"] for d in daily)
        data["total_dev_db_diffs"] = sum(d["dev_vs_db"]["diff_count"] for d in daily)
        data["total_src_db_diffs"] = sum(d["src_vs_db"]["diff_count"] for d in daily)
    return merged


def main():
    t_total = time.time()
    all_comparisons = {}
    module5_parts = []

    for task in TASKS:
        tmp_path = str(PROJECT_ROOT / "tools" / f"_tmp_{task['module']}_{task.get('days', 'all')}.json")
        r = run_module(task["module"], task.get("days"), tmp_path)
        if not r:
            print(f"FAILED: {task}", flush=True)
            sys.exit(1)

        if task["module"] == "module5":
            module5_parts.append(r)
        else:
            all_comparisons.update(r.get("comparisons", {}))

    # Merge module5
    if module5_parts:
        m5_merged = merge_module5_results(module5_parts)
        all_comparisons.update(m5_merged)

    # Build final result
    total_sheets = len(all_comparisons)
    pass_ds = sum(1 for v in all_comparisons.values() if v.get("dev_vs_src_all_match", True))
    pass_dd = sum(1 for v in all_comparisons.values() if v.get("dev_vs_db_all_match", True))
    pass_sd = sum(1 for v in all_comparisons.values() if v.get("src_vs_db_all_match", True))

    final = {
        "run_time": datetime.now().isoformat(),
        "day_range": "1-76",
        "comparisons": all_comparisons,
        "summary": {
            "total_sheets_compared": total_sheets,
            "dev_vs_src_pass": pass_ds,
            "dev_vs_db_pass": pass_dd,
            "src_vs_db_pass": pass_sd,
            "dev_vs_src_fail": total_sheets - pass_ds,
            "dev_vs_db_fail": total_sheets - pass_dd,
            "src_vs_db_fail": total_sheets - pass_sd,
        },
        "elapsed_seconds": round(time.time() - t_total, 1),
    }

    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"FINAL MERGED RESULTS: {FINAL_OUTPUT}")
    print(f"{'='*80}")
    for comp_key, data in all_comparisons.items():
        ds = "PASS" if data.get("dev_vs_src_all_match") else "FAIL"
        dd = "PASS" if data.get("dev_vs_db_all_match") else "FAIL"
        sd = "PASS" if data.get("src_vs_db_all_match") else "FAIL"
        print(f"  {comp_key:<35} Dev/Src={ds}  Dev/DB={dd}  Src/DB={sd}")
    s = final["summary"]
    print(f"\nTotal: {s['total_sheets_compared']} sheets | "
          f"Dev/Src: {s['dev_vs_src_pass']} PASS, {s['dev_vs_src_fail']} FAIL | "
          f"Dev/DB: {s['dev_vs_db_pass']} PASS, {s['dev_vs_db_fail']} FAIL | "
          f"Src/DB: {s['src_vs_db_pass']} PASS, {s['src_vs_db_fail']} FAIL")
    print(f"Total time: {final['elapsed_seconds']}s")


if __name__ == "__main__":
    main()

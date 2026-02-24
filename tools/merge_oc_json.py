#!/usr/bin/env python3
"""合并分模块/分天段的OC对比JSON为完整的 oc_3way_content_results_full.json"""
import json
from pathlib import Path
from datetime import datetime

TOOLS_DIR = Path(__file__).parent

# 需要合并的文件列表（按模块+天段）
PARTS = [
    TOOLS_DIR / "oc_3way_m1.json",   # module1 day1-76
    TOOLS_DIR / "oc_3way_m3.json",   # module3 day1-76
    TOOLS_DIR / "oc_3way_m4.json",   # module4 day1-76
    TOOLS_DIR / "oc_3way_m5a.json",  # module5 day1-38
    TOOLS_DIR / "oc_3way_m5b.json",  # module5 day39-76
    TOOLS_DIR / "oc_3way_m6.json",   # module6 day1-76
]

def merge():
    merged = {
        "run_time": datetime.now().isoformat(),
        "day_range": "1-76",
        "comparisons": {},
        "elapsed_seconds": 0,
    }

    total_elapsed = 0
    total_sheets = 0
    total_pass_dev_src = 0
    total_pass_dev_db = 0

    for fp in PARTS:
        if not fp.exists():
            print(f"WARNING: {fp} not found, skipping")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_elapsed += data.get("elapsed_seconds", 0)

        for comp_key, comp_data in data.get("comparisons", {}).items():
            if comp_key in merged["comparisons"]:
                # 同一个sheet的不同天段需要合并daily列表
                existing = merged["comparisons"][comp_key]
                existing["daily"].extend(comp_data["daily"])
                # 重新计算汇总
                existing["total_dev_rows"] += comp_data["total_dev_rows"]
                existing["total_src_rows"] += comp_data["total_src_rows"]
                existing["total_db_rows"] += comp_data["total_db_rows"]
                existing["total_dev_src_diffs"] += comp_data["total_dev_src_diffs"]
                existing["total_dev_db_diffs"] += comp_data["total_dev_db_diffs"]
                # 重新计算all_match
                existing["dev_vs_src_all_match"] = existing["dev_vs_src_all_match"] and comp_data["dev_vs_src_all_match"]
                existing["dev_vs_db_all_match"] = existing["dev_vs_db_all_match"] and comp_data["dev_vs_db_all_match"]
            else:
                merged["comparisons"][comp_key] = comp_data

    # 对每个sheet的daily按day排序
    for comp_key, comp_data in merged["comparisons"].items():
        comp_data["daily"].sort(key=lambda x: x["day"])

    # 计算summary
    for comp_key, comp_data in merged["comparisons"].items():
        total_sheets += 1
        if comp_data["dev_vs_src_all_match"]:
            total_pass_dev_src += 1
        if comp_data["dev_vs_db_all_match"]:
            total_pass_dev_db += 1

    merged["summary"] = {
        "total_sheets_compared": total_sheets,
        "dev_vs_src_pass": total_pass_dev_src,
        "dev_vs_db_pass": total_pass_dev_db,
        "dev_vs_src_fail": total_sheets - total_pass_dev_src,
        "dev_vs_db_fail": total_sheets - total_pass_dev_db,
    }
    merged["elapsed_seconds"] = round(total_elapsed, 1)

    out_path = TOOLS_DIR / "oc_3way_content_results_full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2, default=str)

    print(f"Merged {total_sheets} sheets from {len(PARTS)} files")
    print(f"Dev/Src: {total_pass_dev_src} PASS, {total_sheets - total_pass_dev_src} FAIL")
    print(f"Dev/DB:  {total_pass_dev_db} PASS, {total_sheets - total_pass_dev_db} FAIL")
    print(f"Total elapsed: {total_elapsed:.1f}s")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    merge()

import json
from pathlib import Path

fp = Path(r"D:\PG\test\chainsight\tools\bc_3way_content_results.json")
d = json.load(open(fp, encoding="utf-8"))

# Find tables where dev_vs_db_all_match or src_vs_db_all_match is False
comps = d["comparisons"]
for table_key, table_data in comps.items():
    if not isinstance(table_data, dict):
        continue
    ds = table_data.get("dev_vs_src_all_match")
    dd = table_data.get("dev_vs_db_all_match")
    sd = table_data.get("src_vs_db_all_match")
    if ds is False or dd is False or sd is False:
        print(f"FAIL table: {table_key}")
        print(f"  Dev vs Src: {ds}, Dev vs DB: {dd}, Src vs DB: {sd}")
        print(f"  total_dev_src_diffs: {table_data.get('total_dev_src_diffs')}")
        print(f"  total_dev_db_diffs: {table_data.get('total_dev_db_diffs')}")
        print(f"  total_src_db_diffs: {table_data.get('total_src_db_diffs')}")
        # Show first few daily failures
        daily = table_data.get("daily", [])
        fail_days = []
        for day_info in daily:
            for pair in ["dev_vs_db", "src_vs_db", "dev_vs_src"]:
                if pair in day_info and isinstance(day_info[pair], dict):
                    if not day_info[pair].get("content_match", True) or not day_info[pair].get("row_match", True):
                        fail_days.append((day_info.get("day"), day_info.get("date"), pair, day_info[pair]))
        print(f"  Failed days: {len(fail_days)}")
        for day_num, date, pair, info in fail_days[:3]:
            print(f"    Day{day_num} ({date}) {pair}: rows_match={info.get('row_match')}, content_match={info.get('content_match')}, diffs={info.get('diff_count')}")
        if len(fail_days) > 3:
            print(f"    ... and {len(fail_days)-3} more")
        print()

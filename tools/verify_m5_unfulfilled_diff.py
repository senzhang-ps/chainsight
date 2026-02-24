# -*- coding: utf-8 -*-
"""
直接验证 module5/UnfulfilledLog 的 Dev vs DB 差异
"""
import pandas as pd
import psycopg2
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEV_DIR = PROJECT_ROOT / "ChainSight_Dev" / "BC_S5" / "run_20260211_181635"
SRC_DIR = PROJECT_ROOT / "outputs" / "BC_S5" / "run_20260211_145215"
DB_CONN = dict(host="localhost", port=5432, dbname="test_db",
               user="postgres", password="123456", client_encoding="utf8")
DB_RUN_ID = "BC_S5_20260211_145215"

START_DATE = pd.Timestamp("2025-10-05")
NUM_DAYS = 87

conn = psycopg2.connect(**DB_CONN)

KEY_COLS = ["date", "sending", "receiving", "demand_element"]
COMPARE_COLS = ["demand_qty", "unfulfilled_qty", "reason"]

total_dev_rows = 0
total_db_rows = 0
total_src_rows = 0
diff_days = []

for day in range(1, NUM_DAYS + 1):
    sim_date = START_DATE + pd.Timedelta(days=day - 1)
    ds = sim_date.strftime("%Y%m%d")
    ds_db = ds  # sim_date in DB is stored as YYYYMMDD string

    # Dev
    dev_file = DEV_DIR / "module5" / f"Module5Output_{ds}.xlsx"
    dev_df = None
    if dev_file.exists():
        try:
            dev_df = pd.read_excel(dev_file, sheet_name="UnfulfilledLog", engine="openpyxl")
        except Exception:
            dev_df = pd.DataFrame()
    else:
        dev_df = pd.DataFrame()

    # Src
    src_file = SRC_DIR / "module5" / f"Module5Output_{ds}.xlsx"
    src_df = None
    if src_file.exists():
        try:
            src_df = pd.read_excel(src_file, sheet_name="UnfulfilledLog", engine="openpyxl")
        except Exception:
            src_df = pd.DataFrame()
    else:
        src_df = pd.DataFrame()

    # DB
    q = "SELECT * FROM module5_output_unfulfilledlog WHERE sim_date = %s AND run_id = %s"
    db_df = pd.read_sql(q, conn, params=[ds_db, DB_RUN_ID])
    drop = [c for c in db_df.columns if c in {"sim_date", "run_id", "config_name", "db_write_time"}]
    db_df = db_df.drop(columns=drop, errors="ignore")

    dev_rows = len(dev_df) if dev_df is not None else 0
    src_rows = len(src_df) if src_df is not None else 0
    db_rows = len(db_df)

    total_dev_rows += dev_rows
    total_src_rows += src_rows
    total_db_rows += db_rows

    row_diff = dev_rows - db_rows

    # Content diff
    content_diff = 0
    if dev_rows > 0 and db_rows > 0:
        avail_keys = [c for c in KEY_COLS if c in dev_df.columns and c in db_df.columns]
        avail_cmp = [c for c in COMPARE_COLS if c in dev_df.columns and c in db_df.columns]
        if avail_keys and avail_cmp:
            dev_m = dev_df[avail_keys + avail_cmp].copy()
            db_m = db_df[avail_keys + avail_cmp].copy()
            for k in avail_keys:
                dev_m[k] = dev_m[k].astype(str).str.strip()
                db_m[k] = db_m[k].astype(str).str.strip()
            dev_m = dev_m.sort_values(avail_keys + avail_cmp).reset_index(drop=True)
            db_m = db_m.sort_values(avail_keys + avail_cmp).reset_index(drop=True)
            dev_m["_idx"] = dev_m.groupby(avail_keys).cumcount()
            db_m["_idx"] = db_m.groupby(avail_keys).cumcount()
            merged = dev_m.merge(db_m, on=avail_keys + ["_idx"], how="outer",
                                 suffixes=("_dev", "_db"), indicator=True)
            only_dev = (merged["_merge"] == "left_only").sum()
            only_db = (merged["_merge"] == "right_only").sum()
            both = merged[merged["_merge"] == "both"]
            content_diff = int(only_dev + only_db)
            for col in avail_cmp:
                c_dev = f"{col}_dev"
                c_db = f"{col}_db"
                if c_dev in both.columns and c_db in both.columns:
                    v_dev = both[c_dev].fillna("").astype(str)
                    v_db = both[c_db].fillna("").astype(str)
                    content_diff += int((v_dev != v_db).sum())

    if row_diff != 0 or content_diff > 0:
        diff_days.append((day, ds, dev_rows, src_rows, db_rows, row_diff, content_diff))

conn.close()

print(f"Total rows: Dev={total_dev_rows}, Src={total_src_rows}, DB={total_db_rows}")
print(f"Row difference (Dev - DB): {total_dev_rows - total_db_rows}")
print(f"\nDays with differences: {len(diff_days)}")
print(f"{'Day':>4s} {'Date':>10s} {'Dev':>7s} {'Src':>7s} {'DB':>7s} {'RowDiff':>8s} {'ContentDiff':>12s}")
print("-" * 60)
for day, ds, dr, sr, dbr, rd, cd in diff_days:
    print(f"{day:4d} {ds:>10s} {dr:7d} {sr:7d} {dbr:7d} {rd:8d} {cd:12d}")

# -*- coding: utf-8 -*-
"""
Verify module5/UnfulfilledLog Dev vs DB using frequency-based (multiset) comparison.
This eliminates row ordering sensitivity and handles int/float type differences.
"""
import pandas as pd
import psycopg2
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DEV_DIR = PROJECT_ROOT / "ChainSight_Dev" / "BC_S5" / "run_20260211_181635"
DB_CONN = dict(host="localhost", port=5432, dbname="test_db",
               user="postgres", password="123456", client_encoding="utf8")
DB_RUN_ID = "BC_S5_20260211_145215"

START_DATE = pd.Timestamp("2025-10-05")
NUM_DAYS = 87

conn = psycopg2.connect(**DB_CONN)

KEY_COLS = ["date", "sending", "receiving", "demand_element"]
COMPARE_COLS = ["demand_qty", "unfulfilled_qty", "reason"]
ALL_COLS = KEY_COLS + COMPARE_COLS


def normalize_row(row, cols):
    """Convert a row to a canonical tuple for frequency comparison."""
    parts = []
    for c in cols:
        v = row.get(c, None)
        if pd.isna(v):
            parts.append(("__NA__",))
        elif isinstance(v, float):
            # Convert float to int if it's a whole number (handles 15.0 -> 15)
            if v == int(v):
                parts.append(int(v))
            else:
                parts.append(round(v, 6))
        elif isinstance(v, str):
            parts.append(v.strip())
        else:
            parts.append(v)
    return tuple(parts)


total_dev_rows = 0
total_db_rows = 0
diff_days_row = []      # Days with row count differences only
diff_days_content = []  # Days with content differences (same row count but different data)
match_days = 0

for day in range(1, NUM_DAYS + 1):
    sim_date = START_DATE + pd.Timedelta(days=day - 1)
    ds = sim_date.strftime("%Y%m%d")

    # Dev
    dev_file = DEV_DIR / "module5" / f"Module5Output_{ds}.xlsx"
    if dev_file.exists():
        try:
            dev_df = pd.read_excel(dev_file, sheet_name="UnfulfilledLog", engine="openpyxl")
        except Exception:
            dev_df = pd.DataFrame()
    else:
        dev_df = pd.DataFrame()

    # DB
    q = "SELECT * FROM module5_output_unfulfilledlog WHERE sim_date = %s AND run_id = %s"
    db_df = pd.read_sql(q, conn, params=[ds, DB_RUN_ID])
    drop = [c for c in db_df.columns if c in {"sim_date", "run_id", "config_name", "db_write_time"}]
    db_df = db_df.drop(columns=drop, errors="ignore")

    dev_rows = len(dev_df)
    db_rows = len(db_df)
    total_dev_rows += dev_rows
    total_db_rows += db_rows

    # Determine available columns
    avail_cols = [c for c in ALL_COLS if c in dev_df.columns and c in db_df.columns]
    if not avail_cols or dev_rows == 0 or db_rows == 0:
        if dev_rows != db_rows:
            diff_days_row.append((day, ds, dev_rows, db_rows, dev_rows - db_rows, "N/A"))
        else:
            match_days += 1
        continue

    # Build frequency counters (multiset comparison)
    dev_counter = Counter()
    for _, row in dev_df.iterrows():
        dev_counter[normalize_row(row, avail_cols)] += 1

    db_counter = Counter()
    for _, row in db_df.iterrows():
        db_counter[normalize_row(row, avail_cols)] += 1

    # Compare multisets
    only_in_dev = dev_counter - db_counter  # rows in Dev but not in DB (or more copies in Dev)
    only_in_db = db_counter - dev_counter   # rows in DB but not in Dev

    extra_dev = sum(only_in_dev.values())
    extra_db = sum(only_in_db.values())

    if extra_dev == 0 and extra_db == 0:
        match_days += 1
    elif dev_rows != db_rows:
        diff_days_row.append((day, ds, dev_rows, db_rows, dev_rows - db_rows,
                              f"+Dev={extra_dev}, +DB={extra_db}"))
    else:
        diff_days_content.append((day, ds, dev_rows, db_rows, extra_dev, extra_db))

conn.close()

print(f"Total rows: Dev={total_dev_rows}, DB={total_db_rows}")
print(f"Row difference (Dev - DB): {total_dev_rows - total_db_rows}")
print(f"Perfect match days: {match_days}")
print()

if diff_days_row:
    print(f"=== Days with ROW COUNT differences: {len(diff_days_row)} ===")
    print(f"{'Day':>4s} {'Date':>10s} {'Dev':>7s} {'DB':>7s} {'RowDiff':>8s} {'Detail'}")
    print("-" * 70)
    for day, ds, dr, dbr, rd, detail in diff_days_row:
        print(f"{day:4d} {ds:>10s} {dr:7d} {dbr:7d} {rd:8d} {detail}")
    print()

if diff_days_content:
    print(f"=== Days with CONTENT differences (same row count): {len(diff_days_content)} ===")
    print(f"{'Day':>4s} {'Date':>10s} {'Dev':>7s} {'DB':>7s} {'ExtraDev':>9s} {'ExtraDB':>9s}")
    print("-" * 60)
    for day, ds, dr, dbr, ed, edb in diff_days_content:
        print(f"{day:4d} {ds:>10s} {dr:7d} {dbr:7d} {ed:9d} {edb:9d}")
    print()

if not diff_days_row and not diff_days_content:
    print("ALL 87 DAYS MATCH PERFECTLY (Dev == DB)")
else:
    print(f"Summary: {match_days} match, {len(diff_days_row)} row-diff, {len(diff_days_content)} content-diff")

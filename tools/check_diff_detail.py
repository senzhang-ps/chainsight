"""
检查 Dev (Excel) vs DB (PostgreSQL) 的具体数值差异
针对 module5/DeploymentPlan 和 module5/UnfulfilledLog 的 FAIL 天
"""
import pandas as pd
import numpy as np
import psycopg2

DEV_DIR = r"D:\PG\test\chainsight\ChainSight_Dev\BC_S5\run_20260211_181635"
DB_CONN = dict(host="localhost", port=5432, dbname="test_db", user="postgres", password="123456")
RUN_ID = "config/BC_S5_20260212_171802"

# Affected days from JSON analysis
FAIL_DAYS = {
    12: "20251016",
    17: "20251021",
    22: "20251026",
    82: "20251225",
}


def compare_deployment_plan(day_num, sim_date):
    """Compare DeploymentPlan for a specific day."""
    print(f"\n{'='*80}")
    print(f"DeploymentPlan — Day {day_num} (sim_date={sim_date})")
    print(f"{'='*80}")

    # Read Dev Excel
    excel_path = f"{DEV_DIR}/module5/Module5Output_{sim_date}.xlsx"
    dev = pd.read_excel(excel_path, sheet_name="DeploymentPlan")
    dev.columns = [c.lower() for c in dev.columns]
    # Normalize date column to string
    dev["date"] = dev["date"].astype(str).str.replace("-", "")

    # Read DB
    conn = psycopg2.connect(**DB_CONN)
    db = pd.read_sql(
        f"SELECT * FROM module5_output_deploymentplan WHERE sim_date='{sim_date}' AND run_id='{RUN_ID}'",
        conn,
    )
    conn.close()
    # Drop DB metadata columns
    db = db.drop(columns=["sim_date", "run_id", "config_name", "db_write_time"], errors="ignore")

    print(f"  Dev rows: {len(dev)}, DB rows: {len(db)}")

    # The diff columns of interest
    diff_cols = ["deploy_qty_with_plan_order", "deploy_from_future_production"]

    # Build a key for merging — use all columns except the diff columns and non-key columns
    # Key candidates: date, material, sending, receiving, demand_element, planned_delivery_date, orig_location
    # But rows may not be unique on these. Let's use row index (positional) matching
    # since the comparison tool likely did positional match (same row count).
    # Actually, let's sort both DataFrames the same way and compare positionally.

    # Common columns (non-diff)
    all_dev_cols = set(dev.columns)
    all_db_cols = set(db.columns)
    common_cols = sorted(all_dev_cols & all_db_cols)
    print(f"  Common columns: {len(common_cols)}")
    print(f"  Dev-only columns: {all_dev_cols - all_db_cols}")
    print(f"  DB-only columns: {all_db_cols - all_dev_cols}")

    # Sort both by a stable set of columns for positional comparison
    sort_cols = [c for c in ["date", "material", "sending", "receiving", "demand_element",
                             "planned_delivery_date", "orig_location", "demand_qty", "planned_qty"] 
                 if c in common_cols]
    dev_sorted = dev[common_cols].sort_values(sort_cols).reset_index(drop=True)
    db_sorted = db[common_cols].sort_values(sort_cols).reset_index(drop=True)

    # Compare the diff columns
    for col in diff_cols:
        if col not in common_cols:
            print(f"  Column '{col}' not in common columns, skipping")
            continue

        dev_vals = dev_sorted[col].fillna(0).astype(float)
        db_vals = db_sorted[col].fillna(0).astype(float)

        # Find rows where values differ (with tolerance)
        mask = ~np.isclose(dev_vals.values, db_vals.values, rtol=1e-9, atol=1e-9, equal_nan=True)
        diff_idx = np.where(mask)[0].tolist()
        print(f"\n  Column '{col}': {len(diff_idx)} rows differ")

        if diff_idx:
            # Show first 10 examples with context
            print(f"  {'Row':>6}  {'Material':>12}  {'Sending':>8}  {'Receiving':>10}  {'Dev_Value':>12}  {'DB_Value':>12}  {'Diff':>12}")
            print(f"  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
            for i, idx in enumerate(diff_idx[:10]):
                mat = dev_sorted.loc[idx, "material"] if "material" in dev_sorted.columns else "?"
                snd = dev_sorted.loc[idx, "sending"] if "sending" in dev_sorted.columns else "?"
                rcv = dev_sorted.loc[idx, "receiving"] if "receiving" in dev_sorted.columns else "?"
                dv = dev_vals.iloc[idx]
                dbv = db_vals.iloc[idx]
                print(f"  {idx:>6}  {str(mat):>12}  {str(snd):>8}  {str(rcv):>10}  {dv:>12.4f}  {dbv:>12.4f}  {dv-dbv:>12.4f}")

            # Summary stats
            dev_diff_vals = dev_vals.values[mask]
            db_diff_vals = db_vals.values[mask]
            abs_diffs = np.abs(dev_diff_vals - db_diff_vals)
            print(f"\n  Diff statistics:")
            print(f"    Mean absolute diff: {abs_diffs.mean():.6f}")
            print(f"    Max absolute diff:  {abs_diffs.max():.6f}")
            print(f"    Min absolute diff:  {abs_diffs.min():.6f}")
            print(f"    Median abs diff:    {np.median(abs_diffs):.6f}")


def compare_unfulfilled_log(day_num, sim_date):
    """Compare UnfulfilledLog for a specific day."""
    print(f"\n{'='*80}")
    print(f"UnfulfilledLog — Day {day_num} (sim_date={sim_date})")
    print(f"{'='*80}")

    # Read Dev Excel
    excel_path = f"{DEV_DIR}/module5/Module5Output_{sim_date}.xlsx"
    dev = pd.read_excel(excel_path, sheet_name="UnfulfilledLog")
    dev.columns = [c.lower() for c in dev.columns]
    dev["date"] = dev["date"].astype(str).str.replace("-", "")

    # Read DB
    conn = psycopg2.connect(**DB_CONN)
    db = pd.read_sql(
        f"SELECT * FROM module5_output_unfulfilledlog WHERE sim_date='{sim_date}' AND run_id='{RUN_ID}'",
        conn,
    )
    conn.close()
    db = db.drop(columns=["sim_date", "run_id", "config_name", "db_write_time"], errors="ignore")

    print(f"  Dev rows: {len(dev)}, DB rows: {len(db)}")
    print(f"  Row difference: {len(dev) - len(db)} (Dev has {'more' if len(dev) > len(db) else 'fewer'})")

    diff_cols = ["demand_qty", "unfulfilled_qty"]
    common_cols = sorted(set(dev.columns) & set(db.columns))

    sort_cols = [c for c in ["date", "sending", "receiving", "demand_element", "reason", "demand_qty"]
                 if c in common_cols]

    # Since row counts differ, we need to do a merge-based comparison
    # Add a row number within each group for matching
    dev_sorted = dev.sort_values(sort_cols).reset_index(drop=True)
    db_sorted = db.sort_values(sort_cols).reset_index(drop=True)

    # Key columns for merge (everything except diff cols)
    key_cols = [c for c in common_cols if c not in diff_cols]

    # Method 1: positional comparison up to min length
    min_len = min(len(dev_sorted), len(db_sorted))
    print(f"  Comparing first {min_len} rows (positional match after sort)")

    for col in diff_cols:
        dev_vals = dev_sorted[col].iloc[:min_len].fillna(0).astype(float)
        db_vals = db_sorted[col].iloc[:min_len].fillna(0).astype(float)

        mask = ~np.isclose(dev_vals.values, db_vals.values, rtol=1e-9, atol=1e-9, equal_nan=True)
        diff_idx = np.where(mask)[0].tolist()
        print(f"\n  Column '{col}': {len(diff_idx)} rows differ (out of {min_len} comparable rows)")

        if diff_idx:
            print(f"  {'Row':>6}  {'Sending':>8}  {'Receiving':>10}  {'Demand_Elem':>12}  {'Dev_Value':>12}  {'DB_Value':>12}  {'Diff':>12}")
            print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
            for idx in diff_idx[:10]:
                snd = dev_sorted.loc[idx, "sending"] if "sending" in dev_sorted.columns else "?"
                rcv = dev_sorted.loc[idx, "receiving"] if "receiving" in dev_sorted.columns else "?"
                de = dev_sorted.loc[idx, "demand_element"] if "demand_element" in dev_sorted.columns else "?"
                dv = dev_vals.iloc[idx]
                dbv = db_vals.iloc[idx]
                print(f"  {idx:>6}  {str(snd):>8}  {str(rcv):>10}  {str(de):>12}  {dv:>12.4f}  {dbv:>12.4f}  {dv-dbv:>12.4f}")

            dev_diff_vals = dev_vals.values[mask]
            db_diff_vals = db_vals.values[mask]
            abs_diffs = np.abs(dev_diff_vals - db_diff_vals)
            print(f"\n  Diff statistics:")
            print(f"    Mean absolute diff: {abs_diffs.mean():.6f}")
            print(f"    Max absolute diff:  {abs_diffs.max():.6f}")
            print(f"    Min absolute diff:  {abs_diffs.min():.6f}")

    # Show the extra rows in Dev (if Dev has more)
    if len(dev_sorted) > len(db_sorted):
        extra = dev_sorted.iloc[len(db_sorted):]
        print(f"\n  Extra rows in Dev (not in DB): {len(extra)}")
        print(f"  Sample extra rows:")
        print(extra.head(5).to_string(index=False))


if __name__ == "__main__":
    # Process only Day 12 (first FAIL day) for initial validation
    # Then optionally expand to all FAIL days
    for day_num, sim_date in sorted(FAIL_DAYS.items()):
        compare_deployment_plan(day_num, sim_date)
        compare_unfulfilled_log(day_num, sim_date)
        # Only do first day in detail; others just summary
        if day_num == 12:
            print("\n\n>>> Detailed output shown for Day 12. Remaining days shown as summary. <<<\n")

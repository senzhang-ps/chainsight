"""
Dev vs Src daily comparison — sampled every 10 days.
Uses vectorized hash comparison for speed (no iterrows).
87 days sampled at days 1,11,21,...,81,87 → ~10 sample points.
"""
import os, sys, json, time, warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

DEV_BASE = r"D:\PG\test\chainsight\ChainSight_Dev\BC_S5\run_20260211_181635"
SRC_BASE = r"D:\PG\test\chainsight\outputs\BC_S5\run_20260211_145215"

START_DATE = datetime(2025, 10, 5)
END_DATE   = datetime(2025, 12, 30)
SAMPLE_EVERY = 10  # every 10 days

MODULES = {
    "module1": {
        "pattern": "module1_output_{date}.xlsx",
        "sheets": ["OrderLog", "ShipmentLog", "CutLog", "SupplyDemandLog", "Summary"],
    },
    "module3": {
        "pattern": "Module3Output_{date}.xlsx",
        "sheets": ["NetDemand"],
    },
    "module4": {
        "pattern": "Module4Output_{date}.xlsx",
        "sheets": ["ProductionPlan", "CapacityExceed", "Validation", "ChangeoverLog"],
    },
    "module5": {
        "pattern": "Module5Output_{date}.xlsx",
        "sheets": ["DeploymentPlan", "UnfulfilledLog", "StockOnHandLog", "Validation"],
    },
    "module6": {
        "pattern": "Module6Output_{date}.xlsx",
        "sheets": ["DeliveryPlan", "VehicleLog", "TruckUsageLog",
                    "UnsatisfiedMDQLog", "ValidationLog", "BypassRuleHitLog"],
    },
}


def normalize_df(df):
    """Normalize a DataFrame for comparison: unify types, fill NaN."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(6)
        else:
            df[col] = df[col].fillna("").astype(str).str.strip()
            # try convert string-numbers to float
            numeric = pd.to_numeric(df[col], errors='coerce')
            mask = numeric.notna()
            if mask.any():
                df.loc[mask, col] = numeric[mask].round(6).astype(str)
    return df


def fast_frequency_compare(df1, df2):
    """
    Fast frequency-based comparison using row hashing.
    Returns (match: bool, detail: str).
    """
    # Column check
    cols1, cols2 = set(df1.columns), set(df2.columns)
    if cols1 != cols2:
        common = sorted(cols1 & cols2)
        if not common:
            return False, f"No common columns. Dev={cols1}, Src={cols2}"
        extra_info = ""
        if cols1 - cols2:
            extra_info += f" only_dev={cols1-cols2}"
        if cols2 - cols1:
            extra_info += f" only_src={cols2-cols1}"
        df1 = df1[common]
        df2 = df2[common]
    else:
        common = sorted(cols1)
        extra_info = ""
        df1 = df1[common]
        df2 = df2[common]

    if df1.empty and df2.empty:
        return True, "both empty"

    # Normalize
    n1 = normalize_df(df1)
    n2 = normalize_df(df2)

    # Convert each row to a hashable string and use Counter
    from collections import Counter
    
    def rows_to_counter(df):
        # Vectorized: join all columns as strings per row
        str_df = df.astype(str)
        keys = str_df.apply(lambda r: '|'.join(r), axis=1)
        return Counter(keys)

    c1 = rows_to_counter(n1)
    c2 = rows_to_counter(n2)

    if c1 == c2:
        return True, f"MATCH ({len(df1)} rows)" + extra_info

    only_dev = sum((c1 - c2).values())
    only_src = sum((c2 - c1).values())
    return False, f"DIFF: {only_dev} rows only-dev, {only_src} rows only-src (dev={len(df1)}, src={len(df2)}){extra_info}"


def sample_dates(start, end, every):
    """Generate sampled dates: day 1, 1+every, 1+2*every, ..., last day."""
    dates = []
    d = start
    idx = 0
    while d <= end:
        if idx % every == 0:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
        idx += 1
    # Always include last day
    last = end.strftime("%Y%m%d")
    if last not in dates:
        dates.append(last)
    return dates


def main():
    t0 = time.time()
    dates = sample_dates(START_DATE, END_DATE, SAMPLE_EVERY)
    total_days = (END_DATE - START_DATE).days + 1
    print(f"Sampling {len(dates)} days out of {total_days} (every {SAMPLE_EVERY} days)")
    print(f"Sample dates: {dates}")
    print(f"Dev: {DEV_BASE}")
    print(f"Src: {SRC_BASE}")
    print("=" * 90)

    total_cmp = 0
    total_pass = 0
    total_fail = 0
    failures = []
    module_summary = {}

    for mod_name, mod_info in MODULES.items():
        pattern = mod_info["pattern"]
        sheets  = mod_info["sheets"]
        m_pass = 0
        m_fail = 0
        m_t0 = time.time()

        for date_str in dates:
            fname = pattern.format(date=date_str)
            dev_path = os.path.join(DEV_BASE, mod_name, fname)
            src_path = os.path.join(SRC_BASE, mod_name, fname)

            if not os.path.exists(dev_path) and not os.path.exists(src_path):
                for s in sheets:
                    total_cmp += 1; total_pass += 1; m_pass += 1
                continue
            if not os.path.exists(dev_path) or not os.path.exists(src_path):
                missing = "Dev" if not os.path.exists(dev_path) else "Src"
                for s in sheets:
                    total_cmp += 1; total_fail += 1; m_fail += 1
                    failures.append(f"{mod_name}/{s} @ {date_str}: {missing} file missing")
                continue

            try:
                dev_xl = pd.ExcelFile(dev_path)
                src_xl = pd.ExcelFile(src_path)
            except Exception as e:
                for s in sheets:
                    total_cmp += 1; total_fail += 1; m_fail += 1
                    failures.append(f"{mod_name}/{s} @ {date_str}: read error {e}")
                continue

            for sheet in sheets:
                total_cmp += 1
                dev_has = sheet in dev_xl.sheet_names
                src_has = sheet in src_xl.sheet_names

                if not dev_has and not src_has:
                    total_pass += 1; m_pass += 1; continue
                if not dev_has or not src_has:
                    total_fail += 1; m_fail += 1
                    missing = "Dev" if not dev_has else "Src"
                    failures.append(f"{mod_name}/{sheet} @ {date_str}: sheet missing in {missing}")
                    continue

                try:
                    df_dev = pd.read_excel(dev_xl, sheet_name=sheet)
                    df_src = pd.read_excel(src_xl, sheet_name=sheet)
                    match, detail = fast_frequency_compare(df_dev, df_src)
                    if match:
                        total_pass += 1; m_pass += 1
                    else:
                        total_fail += 1; m_fail += 1
                        failures.append(f"{mod_name}/{sheet} @ {date_str}: {detail}")
                except Exception as e:
                    total_fail += 1; m_fail += 1
                    failures.append(f"{mod_name}/{sheet} @ {date_str}: error {e}")

            dev_xl.close()
            src_xl.close()

        m_elapsed = time.time() - m_t0
        status = "PASS" if m_fail == 0 else "FAIL"
        module_summary[mod_name] = {"pass": m_pass, "fail": m_fail, "time": round(m_elapsed, 1)}
        print(f"[{status}] {mod_name}: {m_pass}/{m_pass+m_fail} pass  ({m_elapsed:.1f}s)")

    elapsed = time.time() - t0
    print("\n" + "=" * 90)
    print(f"TOTAL: {total_pass}/{total_cmp} PASS, {total_fail} FAIL  ({elapsed:.1f}s)")

    if failures:
        print(f"\n--- FAILURES ({len(failures)}) ---")
        for f in failures:
            print(f"  {f}")

    # Save JSON
    out = {
        "meta": {
            "sample_every": SAMPLE_EVERY,
            "sample_dates": dates,
            "total_days": total_days,
            "total_comparisons": total_cmp,
            "total_pass": total_pass,
            "total_fail": total_fail,
            "elapsed_s": round(elapsed, 1),
        },
        "modules": module_summary,
        "failures": failures,
    }
    out_path = os.path.join(os.path.dirname(__file__), "bc_daily_dev_vs_src_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

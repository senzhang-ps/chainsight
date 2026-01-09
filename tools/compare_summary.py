import sys
from typing import Tuple, List
import pandas as pd


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['date', 'material', 'location'])
    for c in ['date', 'material', 'location']:
        if c in df.columns:
            df[c] = df[c].astype(str)
    num_cols: List[str] = [c for c in df.columns if c not in ['date', 'material', 'location']]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    df = df.sort_values(['date', 'material', 'location']).reset_index(drop=True)
    return df


def _apply_date_filter(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    if df is None or df.empty or 'date' not in df.columns:
        return df
    s = pd.to_datetime(start) if start else None
    e = pd.to_datetime(end) if end else None
    d = pd.to_datetime(df['date'], errors='coerce')
    mask = pd.Series([True] * len(df))
    if s is not None:
        mask = mask & (d >= s)
    if e is not None:
        mask = mask & (d <= e)
    return df[mask].copy()


def compare_csv(base_path: str, new_path: str, start: str = None, end: str = None) -> Tuple[bool, str]:
    base = pd.read_csv(base_path)
    new = pd.read_csv(new_path)
    if start or end:
        base = _apply_date_filter(base, start, end)
        new = _apply_date_filter(new, start, end)
    base = _normalize(base)
    new = _normalize(new)
    if list(base.columns) != list(new.columns):
        return False, f"column_mismatch: base={list(base.columns)} new={list(new.columns)}"
    if base.shape != new.shape:
        return False, f"shape_mismatch: base={base.shape} new={new.shape}"
    if base.equals(new):
        return True, "equal"
    # brief diff summary
    num_cols = [c for c in base.columns if c not in ['date', 'material', 'location']]
    merged = base.merge(new, on=['date', 'material', 'location'], how='outer', suffixes=('_base', '_new'), indicator=True)
    diffs = {}
    for col in num_cols:
        cb, cn = f"{col}_base", f"{col}_new"
        if cb in merged.columns and cn in merged.columns:
            d = merged[(merged[cb].fillna(0) != merged[cn].fillna(0))]
            if not d.empty:
                diffs[col] = int(len(d))
    return False, f"value_diff: {diffs}"


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python tools/compare_summary.py <base_csv> <new_csv> [start_date] [end_date]")
        return 2
    base, new = sys.argv[1], sys.argv[2]
    start = sys.argv[3] if len(sys.argv) >= 4 else None
    end = sys.argv[4] if len(sys.argv) >= 5 else None
    try:
        ok, msg = compare_csv(base, new, start, end)
        print({"equal": ok, "detail": msg})
        return 0 if ok else 1
    except Exception as e:
        print({"equal": False, "error": str(e)})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

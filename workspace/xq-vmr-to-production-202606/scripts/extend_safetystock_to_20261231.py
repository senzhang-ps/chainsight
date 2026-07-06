"""Extend M3_SafetyStock.csv forward to 2026-12-31 by copying each
(material, location) pair's last-available-day safety_stock_qty forward,
one row per day. No new data is pulled.

Preserves original column order, date string format (e.g. 2026/12/9), and
per-pair appearance order. Writes <dir>/M3_SafetyStock_extended.csv for review.
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

TARGETS = [
    Path("input/xq-vmr-to-production-202606/scenarios/baseline/config/M3_SafetyStock.csv"),
    Path("workspace/xq-vmr-to-production-202606/scenarios/baseline/config/M3_SafetyStock.csv"),
]
END = date(2026, 12, 31)


def fmt(d: date) -> str:
    return f"{d.year}/{d.month}/{d.day}"


for p in TARGETS:
    df = pd.read_csv(p, dtype={"location": str})
    df["_d"] = pd.to_datetime(df["date"])

    new_rows = []
    # preserve pair appearance order
    for (mat, loc), g in df.groupby(["material", "location"], sort=False):
        last = g.loc[g["_d"].idxmax()]
        last_d = last["_d"].date()
        qty = last["safety_stock_qty"]
        d = last_d + timedelta(days=1)
        while d <= END:
            new_rows.append({"material": mat, "location": loc,
                             "date": fmt(d), "safety_stock_qty": qty})
            d += timedelta(days=1)

    ext = pd.DataFrame(new_rows)
    ext["_d"] = pd.to_datetime(ext["date"])

    combined = pd.concat([df, ext], ignore_index=True)
    # stable sort by pair (appearance order) then date
    order = {pr: i for i, pr in enumerate(
        df[["material", "location"]].drop_duplicates().itertuples(index=False, name=None))}
    combined["_po"] = combined.apply(lambda r: order[(r["material"], r["location"])], axis=1)
    combined = combined.sort_values(["_po", "_d"], kind="stable").reset_index(drop=True)
    out = combined[["material", "location", "date", "safety_stock_qty"]]

    dest = p.with_name("M3_SafetyStock_extended.csv")
    out.to_csv(dest, index=False)
    print(f"{p.parent.parts[0]}/.../{p.name}")
    print(f"  orig rows={len(df)} added={len(ext)} -> {dest.name} rows={len(out)} "
          f"max_date={pd.to_datetime(out['date']).max().date()} "
          f"pairs={out[['material','location']].drop_duplicates().shape[0]}")

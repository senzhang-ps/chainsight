"""诊断两个 run 的 module1 shipment log 字段级差异。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pgsql_db.settings import resolve_database_config
from src.core.db.pgsql.db import DB

META_COLUMNS = {"run_id", "config_name", "db_write_time"}
# ``order_id`` 在每次重启后会依订单生成顺序重新编号，不能作为 resume
# 业务一致性关联键。发货业务身份以日、物料、地点及需求类型为准。
DEFAULT_KEYS = ["sim_date", "date", "material", "location", "demand_type"]


def _db() -> DB:
    cfg = resolve_database_config()
    db = DB(
        host=cfg["host"], port=cfg["port"], database=cfg["database"],
        user=cfg["user"], password=cfg["password"], schema="test",
        auto_create_schema=False,
    )
    db.connect()
    return db


def _canonical_value(value) -> str:
    if pd.isna(value):
        return "<NULL>"
    return str(value)


def _multiset(rows: pd.DataFrame, columns: list[str]) -> Counter:
    return Counter(tuple(_canonical_value(row[column]) for column in columns) for _, row in rows.iterrows())


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("用法: python tests/_diagnose_resume_shipmentlog_diff.py <连续run_id> <续跑run_id>")
    continuous_id, resumed_id = sys.argv[1:]
    db = _db()
    table = "module1_output_shipmentlog"
    continuous = db.read(table, run_id=continuous_id).drop(columns=META_COLUMNS, errors="ignore")
    resumed = db.read(table, run_id=resumed_id).drop(columns=META_COLUMNS, errors="ignore")
    columns = sorted(set(continuous.columns) | set(resumed.columns))
    continuous = continuous.reindex(columns=columns)
    resumed = resumed.reindex(columns=columns)

    available_keys = [key for key in DEFAULT_KEYS if key in columns]
    left_multiset = _multiset(continuous, columns)
    right_multiset = _multiset(resumed, columns)
    only_continuous = left_multiset - right_multiset
    only_resumed = right_multiset - left_multiset

    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "table": table,
        "continuous_run_id": continuous_id,
        "resumed_run_id": resumed_id,
        "rows": {"continuous": len(continuous), "resumed": len(resumed)},
        "columns": columns,
        "key_columns": available_keys,
        "exact_multiset": {
            "continuous_only": sum(only_continuous.values()),
            "resumed_only": sum(only_resumed.values()),
        },
    }

    if available_keys:
        left = continuous.copy()
        right = resumed.copy()
        left["_ordinal"] = left.groupby(available_keys, dropna=False).cumcount()
        right["_ordinal"] = right.groupby(available_keys, dropna=False).cumcount()
        merge_keys = available_keys + ["_ordinal"]
        payload = [column for column in columns if column not in available_keys]
        merged = left.merge(right, on=merge_keys, how="outer", suffixes=("_continuous", "_resumed"), indicator=True)
        field_counts: dict[str, int] = {}
        samples: list[dict] = []
        for _, row in merged.iterrows():
            changed = {}
            for column in payload:
                left_value = _canonical_value(row.get(f"{column}_continuous"))
                right_value = _canonical_value(row.get(f"{column}_resumed"))
                if left_value != right_value:
                    changed[column] = {"continuous": left_value, "resumed": right_value}
                    field_counts[column] = field_counts.get(column, 0) + 1
            if changed and len(samples) < 30:
                sample = {key: _canonical_value(row[key]) for key in merge_keys}
                sample["row_presence"] = row["_merge"]
                sample["differences"] = changed
                samples.append(sample)
        report["keyed_comparison"] = {
            "matched_or_joined_rows": len(merged),
            "row_presence": {str(key): int(value) for key, value in merged["_merge"].value_counts().items()},
            "field_difference_counts": field_counts,
            "samples": samples,
        }

    aggregate_keys = [key for key in ("sim_date", "date", "material", "location", "demand_type") if key in columns]
    if aggregate_keys:
        def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
            numeric = pd.to_numeric(frame["quantity"], errors="coerce").fillna(0)
            return (
                frame.assign(_quantity=numeric)
                .groupby(aggregate_keys, dropna=False, as_index=False)["_quantity"]
                .sum()
                .sort_values(aggregate_keys, kind="mergesort")
                .reset_index(drop=True)
            )

        aggregate_continuous = _aggregate(continuous)
        aggregate_resumed = _aggregate(resumed)
        aggregate_merged = aggregate_continuous.merge(
            aggregate_resumed,
            on=aggregate_keys,
            how="outer",
            suffixes=("_continuous", "_resumed"),
            indicator=True,
        ).fillna({"_quantity_continuous": 0, "_quantity_resumed": 0})
        aggregate_merged["quantity_delta"] = (
            aggregate_merged["_quantity_continuous"] - aggregate_merged["_quantity_resumed"]
        )
        aggregate_differences = aggregate_merged[
            aggregate_merged["quantity_delta"].ne(0)
        ]
        report["business_aggregate_comparison"] = {
            "keys": aggregate_keys,
            "equal": aggregate_differences.empty,
            "different_groups": len(aggregate_differences),
            "total_quantity_continuous": float(aggregate_continuous["_quantity"].sum()),
            "total_quantity_resumed": float(aggregate_resumed["_quantity"].sum()),
            "samples": aggregate_differences.head(30).to_dict("records"),
        }

    output = PROJECT_ROOT / "outputs" / "resume_parity_compare" / f"shipmentlog_detail_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    print(f"报告: {output}")
    print(json.dumps(report["exact_multiset"], ensure_ascii=False))
    if "keyed_comparison" in report:
        print(json.dumps(report["keyed_comparison"]["field_difference_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

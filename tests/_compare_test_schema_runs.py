"""比较 test schema 中两个完整 run 的模块、状态和汇总输出。"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pgsql_db.settings import resolve_database_config
from src.core.db.pgsql.db import DB

IGNORED_COLUMNS = {"run_id", "config_name", "db_write_time"}


def _db() -> DB:
    cfg = resolve_database_config()
    db = DB(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        schema="test",
        auto_create_schema=False,
    )
    db.connect()
    return db


def _tables(db: DB) -> list[str]:
    rows = db.execute_query(
        "SELECT DISTINCT table_name FROM information_schema.columns "
        "WHERE table_schema = %s AND column_name = 'run_id' "
        "AND (table_name LIKE 'module%%' OR table_name LIKE 'viewcontext%%' "
        "OR table_name LIKE 'summary%%') ORDER BY table_name",
        ("test",),
    )
    return [row[0] for row in rows]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    result = df.drop(columns=IGNORED_COLUMNS, errors="ignore").copy()
    result = result.reindex(sorted(result.columns), axis=1)
    if result.empty:
        return result.reset_index(drop=True)
    for column in result.columns:
        result[column] = result[column].map(
            lambda value: "<NULL>" if pd.isna(value) else str(value)
        )
    return result.sort_values(list(result.columns), kind="mergesort").reset_index(drop=True)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("用法: python tests/_compare_test_schema_runs.py <连续run_id> <续跑run_id>")

    continuous_run_id, resumed_run_id = sys.argv[1:]
    db = _db()
    comparisons: list[dict] = []
    for table in _tables(db):
        continuous = _normalise(db.read(table, run_id=continuous_run_id))
        resumed = _normalise(db.read(table, run_id=resumed_run_id))
        equal = continuous.equals(resumed)
        comparisons.append(
            {
                "table": table,
                "equal": equal,
                "continuous_rows": len(continuous),
                "resumed_rows": len(resumed),
                "columns": continuous.columns.tolist(),
            }
        )

    run_statuses = db.execute_query(
        f"SELECT runid, status, progress_date, finished_at FROM "
        f"{db.qualified_name('orch_run_event')} WHERE runid IN (%s, %s) ORDER BY runid",
        (continuous_run_id, resumed_run_id),
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema": "test",
        "continuous_run_id": continuous_run_id,
        "resumed_run_id": resumed_run_id,
        "run_statuses": [list(row) for row in run_statuses],
        "comparisons": comparisons,
        "different_tables": [item["table"] for item in comparisons if not item["equal"]],
    }
    output = PROJECT_ROOT / "outputs" / "resume_parity_compare" / f"compare_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2), encoding="utf-8")

    print(f"报告: {output}")
    print(f"比较表数: {len(comparisons)}")
    print(f"差异表数: {len(report['different_tables'])}")
    for item in comparisons:
        marker = "OK" if item["equal"] else "DIFF"
        print(f"[{marker}] {item['table']}: 连续={item['continuous_rows']}, 续跑={item['resumed_rows']}")
    return 0 if not report["different_tables"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

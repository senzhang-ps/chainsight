"""只读比较两个 PostgreSQL schema 的 ``cfg_*`` 配置。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgsql_db.settings import resolve_database_config
from src.core.db.pgsql.db import DB

META_COLUMNS = {"config_name", "run_id", "db_write_time", "file_date", "id", "created_at", "updated_at"}


def _db(schema: str) -> DB:
    cfg = resolve_database_config()
    db = DB(
        host=cfg["host"], port=cfg["port"], database=cfg["database"],
        user=cfg["user"], password=cfg["password"], schema=schema,
        auto_create_schema=False,
    )
    db.connect()
    return db


def _tables(db: DB, schema: str) -> list[str]:
    rows = db.execute_query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = %s AND table_name LIKE 'cfg\\_%%' ESCAPE '\\' "
        "ORDER BY table_name",
        (schema,),
    )
    return [str(row[0]) for row in rows]


def _columns(db: DB, schema: str, table: str) -> list[str]:
    rows = db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (schema, table),
    )
    return [str(row[0]) for row in rows]


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float):
        return format(value, ".15g")
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return str(value).strip()


def _read_canonical(db: DB, schema: str, table: str, config_name: str | None) -> tuple[Counter, list[str]]:
    columns = [column for column in _columns(db, schema, table) if column.casefold() not in META_COLUMNS]
    if not columns:
        return Counter(), columns
    quote = lambda identifier: '"' + identifier.replace('"', '""') + '"'
    sql = f"SELECT {', '.join(quote(column) for column in columns)} FROM {db.qualified_name(table)}"
    params: tuple[str, ...] = ()
    if config_name and "config_name" in _columns(db, schema, table):
        sql += " WHERE config_name = %s"
        params = (config_name,)
    rows = db.execute_query(sql, params)
    return Counter(tuple(_canonical_value(value) for value in row) for row in rows), columns


def main() -> int:
    parser = argparse.ArgumentParser(description="比较两个 schema 的 cfg_* 配置业务行")
    parser.add_argument("--left-schema", default="input")
    parser.add_argument("--right-schema", default="legacy")
    parser.add_argument("--config-name", default="OC_Paste_S1_20251224_repare")
    args = parser.parse_args()

    left_db, right_db = _db(args.left_schema), _db(args.right_schema)
    try:
        left_tables, right_tables = set(_tables(left_db, args.left_schema)), set(_tables(right_db, args.right_schema))
        results = []
        for table in sorted(left_tables | right_tables):
            if table not in left_tables or table not in right_tables:
                results.append({"table": table, "status": "table_missing", "left_rows": None, "right_rows": None})
                continue
            left, left_columns = _read_canonical(left_db, args.left_schema, table, args.config_name)
            right, right_columns = _read_canonical(right_db, args.right_schema, table, args.config_name)
            left_only, right_only = left - right, right - left
            results.append({
                "table": table,
                "status": "equal" if left == right and set(left_columns) == set(right_columns) else "different",
                "left_rows": sum(left.values()), "right_rows": sum(right.values()),
                "left_only_rows": sum(left_only.values()), "right_only_rows": sum(right_only.values()),
                "left_only_columns": sorted(set(left_columns) - set(right_columns)),
                "right_only_columns": sorted(set(right_columns) - set(left_columns)),
                "left_signature": hashlib.sha256(repr(sorted(left.items())).encode()).hexdigest(),
                "right_signature": hashlib.sha256(repr(sorted(right.items())).encode()).hexdigest(),
                "left_only_samples": [list(row) for row in list(left_only)[:3]],
                "right_only_samples": [list(row) for row in list(right_only)[:3]],
            })
        report = {
            "left_schema": args.left_schema, "right_schema": args.right_schema,
            "config_name": args.config_name, "tables": results,
            "difference_count": sum(item["status"] != "equal" for item in results),
        }
        output = PROJECT_ROOT / "outputs" / "config_schema_compare"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{args.left_schema}_vs_{args.right_schema}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告: {path}")
        print(f"配置表: {len(results)}；差异: {report['difference_count']}")
        for item in results:
            marker = "OK" if item["status"] == "equal" else "DIFF"
            print(f"[{marker}] {item['table']}: {item.get('left_rows')} vs {item.get('right_rows')}")
        return 0
    finally:
        left_db.close()
        right_db.close()


if __name__ == "__main__":
    raise SystemExit(main())

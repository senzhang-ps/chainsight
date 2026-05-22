"""清空 Summary 汇总表与 Orchestrator 状态表（一次性维护脚本）。

直接 ``python tests/clear_summary_and_orchestrator_tables.py`` 即可。

- 默认 TRUNCATE：把表清空到 0 行，但保留表结构与索引；跨 run_id 全部清掉。
- 表不存在时安静跳过，不报错。
- 仅清这两组表，**不会动 module1..6_output_* 输出表**。

如果只想清当前 run_id 的行（保留其他历史 run）：把 ``MODE`` 改为 ``"by_run_id"``
并填好 ``RUN_ID``。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 把仓库根加入 sys.path，便于直接运行而非 -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psycopg import sql

from pgsql_db.db_connection import DatabaseConnection


# ==================== 配置 ====================
MODE = "truncate"              # "truncate"（全清） 或 "by_run_id"（按 run_id 删）
RUN_ID = ""                    # MODE=="by_run_id" 时使用
# ==============================================


SUMMARY_TABLES = [
    "summary_output_ordershipmentcutsummary",
    "summary_output_fullchangeoverlog",
    "summary_output_fullcapacityexceed",
    "summary_output_fullproductionplan",
    "summary_output_fulldeploymentplan",
    "summary_output_fulldeliveryplan",
    "summary_output_fulltruckusage",
]

ORCHESTRATOR_TABLES = [
    "orchestrator_unrestricted_inventory",
    "orchestrator_open_deployment",
    "orchestrator_open_deployment_pastdue_cleanup",
    "orchestrator_planning_intransit",
    "orchestrator_space_quota",
    "orchestrator_delivery_gr",
    "orchestrator_production_gr",
    "orchestrator_production_plan_backlog",
    "orchestrator_shipment_log",
    "orchestrator_delivery_shipment_log",
    "orchestrator_inventory_change_log",
    "orchestrator_daily_logs",
]


def _table_exists(db: DatabaseConnection, table_name: str) -> bool:
    with db.get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            (table_name,),
        )
        return bool(cur.fetchone()[0])


def _count_rows(db: DatabaseConnection, table_name: str) -> int:
    with db.get_cursor(commit=False) as cur:
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
        return int(cur.fetchone()[0])


def _truncate(db: DatabaseConnection, table_name: str) -> tuple[bool, int]:
    """TRUNCATE 单表。返回 (是否执行, 清前行数)。"""
    if not _table_exists(db, table_name):
        return False, 0
    before = _count_rows(db, table_name)
    with db.get_cursor() as cur:
        cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY").format(
            sql.Identifier(table_name)
        ))
    return True, before


def _delete_by_run_id(
    db: DatabaseConnection, table_name: str, run_id: str
) -> tuple[bool, int]:
    """按 run_id DELETE 单表。返回 (是否执行, 删除行数)。"""
    if not _table_exists(db, table_name):
        return False, 0
    with db.get_cursor() as cur:
        cur.execute(
            sql.SQL("DELETE FROM {} WHERE run_id = %s").format(sql.Identifier(table_name)),
            (run_id,),
        )
        deleted = cur.rowcount or 0
    return True, deleted


def main() -> int:
    if MODE not in {"truncate", "by_run_id"}:
        print(f"[ERROR] MODE 必须是 'truncate' 或 'by_run_id'，当前: {MODE!r}")
        return 2
    if MODE == "by_run_id" and not RUN_ID:
        print("[ERROR] MODE='by_run_id' 时必须设置 RUN_ID")
        return 2

    db = DatabaseConnection()
    db.connect()

    groups = [("Summary 汇总表", SUMMARY_TABLES), ("Orchestrator 状态表", ORCHESTRATOR_TABLES)]
    total_tables = 0
    total_rows = 0

    print(f"[模式] {MODE}" + (f"  (run_id={RUN_ID})" if MODE == "by_run_id" else ""))
    print()

    try:
        for group_name, tables in groups:
            print(f"=== {group_name} ===")
            for tbl in tables:
                if MODE == "truncate":
                    ran, n = _truncate(db, tbl)
                else:
                    ran, n = _delete_by_run_id(db, tbl, RUN_ID)
                if not ran:
                    print(f"  [SKIP] {tbl}  (表不存在)")
                    continue
                action = "TRUNCATE" if MODE == "truncate" else "DELETE"
                print(f"  [{action}] {tbl}  ← {n} 行")
                total_tables += 1
                total_rows += n
            print()

        print(f"[OK] 共清理 {total_tables} 张表，累计 {total_rows} 行")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())

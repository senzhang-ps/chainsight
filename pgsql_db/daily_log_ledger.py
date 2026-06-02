"""orchestrator daily logs 累计快照的 DB-only 恢复工具。"""
from __future__ import annotations  # 延迟解析类型注解，降低运行期导入依赖。

import logging  # 用于记录兼容恢复路径的 warning。
from typing import TYPE_CHECKING  # 用于仅在类型检查阶段导入数据库连接类型。

import pandas as pd  # 用于识别并转换 pd.Timestamp。

if TYPE_CHECKING:  # 避免运行期导入导致循环依赖。
    from pgsql_db.db_connection import DatabaseConnection  # schema-aware 数据库连接类型。

logger = logging.getLogger("SupplyChainSimulation")  # 复用仿真主日志。

DAILY_LOG_TABLE = "orchestrator_daily_logs"  # daily logs 在 PostgreSQL 中的表名。
DAILY_LOG_MODE = "cumulative_snapshot"  # 当前 daily logs 存储口径：累计快照。
DAILY_LOG_COLUMNS = ("timestamp", "date", "event_type", "message")  # 单条日志事件的核心字段。


def ensure_daily_log_ledger_schema(db: "DatabaseConnection") -> None:
    """创建或补齐 daily logs 累计快照表。

    参数：
        db: 带 schema 上下文的数据库连接。
    """
    qualified = db.qualified_name(DAILY_LOG_TABLE)  # 生成当前 schema 下的 daily logs 表名。
    db.execute_non_query(  # 首次运行时创建 daily logs 表。
        f"""
        CREATE TABLE IF NOT EXISTS {qualified} (
            "run_id" TEXT,
            "sim_date" TEXT,
            "config_name" TEXT,
            "db_write_time" TIMESTAMP,
            "file_date" TEXT,
            "timestamp" TEXT,
            "date" TEXT,
            "event_type" TEXT,
            "message" TEXT
        )
        """
    )

    # 兼容旧库：表已存在但缺少这些列时逐列补齐。
    for ddl in (  # 每个 DDL 只补一个列，便于幂等执行。
        'ADD COLUMN IF NOT EXISTS "file_date" TEXT',
        'ADD COLUMN IF NOT EXISTS "timestamp" TEXT',
        'ADD COLUMN IF NOT EXISTS "date" TEXT',
        'ADD COLUMN IF NOT EXISTS "event_type" TEXT',
        'ADD COLUMN IF NOT EXISTS "message" TEXT',
    ):
        db.execute_non_query(f"ALTER TABLE {qualified} {ddl}")  # 幂等补齐列。


def initialize_daily_log_ledger_from_db(
    db: "DatabaseConnection",
    orch,
    run_id: str,
    last_batch_end: str | None = None,
) -> int:
    """从 PostgreSQL 恢复累计 daily logs。

    参数：
        db: 带 schema 上下文的数据库连接。
        orch: 待恢复 ``daily_logs`` 字段的 orchestrator 实例。
        run_id: 需要恢复的运行标识。
        last_batch_end: 最后成功提交的 checkpoint 日期；提供时只恢复到该日期。

    返回：
        写入 ``orch.daily_logs`` 的日志记录数量。
    """
    ensure_daily_log_ledger_schema(db)  # 确保表结构存在后再查询。

    orch.daily_log_mode = DAILY_LOG_MODE  # 将 orchestrator 标记为累计快照口径。
    orch.daily_log_pending_events = []  # 清空早期事件流水实验遗留的 pending 队列。
    orch.daily_log_next_seq = 1  # 重置事件流水序号，避免旧状态污染。
    orch.daily_log_last_committed_seq = 0  # 重置事件流水提交游标。
    orch.daily_logs = []  # 清空内存日志，后续只从 DB 恢复。

    if not run_id:  # 没有 run_id 无法定位历史数据。
        return 0  # 视为没有可恢复日志。

    if last_batch_end:  # 有 checkpoint 边界时必须按边界日期恢复。
        if _run_has_event_seq(db, run_id):  # 兼容早期事件流水实验数据。
            restored = _restore_event_rows_through_date(db, orch, run_id, last_batch_end)  # 恢复边界日前事件行。
            if restored:  # 成功恢复实验数据时直接返回。
                logger.warning(  # 告警说明当前 run 使用了兼容恢复路径。
                    "  daily_logs restored from event rows for run_id=%s through %s",
                    run_id,
                    last_batch_end,
                )
                return restored  # 返回恢复数量。

        restored = _restore_snapshot_for_date(db, orch, run_id, last_batch_end)  # 正常路径：恢复边界日累计快照。
        if restored:  # 找到快照时返回恢复数量。
            return restored
        return 0  # 边界日没有快照时返回 0。

    return _restore_latest_snapshot(db, orch, run_id)  # 没有 checkpoint 边界时恢复该 run 最新快照。


def _restore_snapshot_for_date(
    db: "DatabaseConnection",
    orch,
    run_id: str,
    sim_date: str,
) -> int:
    """恢复指定仿真日的累计快照。

    参数：
        db: 带 schema 上下文的数据库连接。
        orch: 需要更新的 orchestrator 实例。
        run_id: 需要恢复的运行标识。
        sim_date: 兼容 ``YYYY-MM-DD`` 的快照日期。

    返回：
        恢复的日志记录数量。
    """
    # 转成 text 并截取日期部分，兼容 DATE、TIMESTAMP、TEXT 三类存储。
    query = (  # 查询指定 run_id 和 sim_date 的累计快照行。
        f'SELECT "timestamp", "date", "event_type", "message" '
        f"FROM {db.qualified_name(DAILY_LOG_TABLE)} "
        "WHERE run_id = %s AND LEFT(sim_date::text, 10) = %s "
        'ORDER BY COALESCE("timestamp"::text, \'\') ASC'
    )
    rows = db.execute_query(query, (run_id, str(sim_date)[:10]))  # 参数化执行查询。
    return _assign_rows(orch, rows)  # 将查询结果写回 orchestrator。


def _run_has_event_seq(db: "DatabaseConnection", run_id: str) -> bool:
    """判断指定 run 是否存在旧事件流水 daily-log 行。

    参数：
        db: 带 schema 上下文的数据库连接。
        run_id: 待检查的运行标识。

    返回：
        至少存在一行 ``event_seq`` 时返回 True，否则返回 False。
    """
    try:  # event_seq 列可能不存在，因此需要捕获异常。
        rows = db.execute_query(  # 检查是否存在事件流水实验行。
            f"SELECT 1 FROM {db.qualified_name(DAILY_LOG_TABLE)} "
            "WHERE run_id = %s AND event_seq IS NOT NULL LIMIT 1",
            (run_id,),
        )
        return bool(rows)  # 查询到记录则表示存在旧事件流水数据。
    except Exception:  # 表无 event_seq 列或查询失败时按不存在处理。
        return False


def _restore_latest_snapshot(db: "DatabaseConnection", orch, run_id: str) -> int:
    """恢复指定 run 最新可用的 daily-log 快照。

    参数：
        db: 带 schema 上下文的数据库连接。
        orch: 需要更新的 orchestrator 实例。
        run_id: 需要恢复的运行标识。

    返回：
        恢复的日志记录数量。
    """
    rows = db.execute_query(  # 查询该 run 最大 sim_date。
        f"SELECT LEFT(MAX(sim_date::text), 10) "
        f"FROM {db.qualified_name(DAILY_LOG_TABLE)} WHERE run_id = %s",
        (run_id,),
    )
    if not rows or not rows[0][0]:  # 没有历史快照时返回 0。
        return 0
    return _restore_snapshot_for_date(db, orch, run_id, str(rows[0][0]))  # 恢复最新日期快照。


def _restore_event_rows_through_date(
    db: "DatabaseConnection",
    orch,
    run_id: str,
    sim_date: str,
) -> int:
    """恢复 checkpoint 日期之前的旧事件流水行。

    参数：
        db: 带 schema 上下文的数据库连接。
        orch: 需要更新的 orchestrator 实例。
        run_id: 需要恢复的运行标识。
        sim_date: 兼容 ``YYYY-MM-DD`` 的闭区间恢复边界。

    返回：
        去重后的恢复日志记录数量。
    """
    query = (  # 查询截至边界日的事件流水实验行。
        f'SELECT "timestamp", "date", "event_type", "message" '
        f"FROM {db.qualified_name(DAILY_LOG_TABLE)} "
        "WHERE run_id = %s AND LEFT(sim_date::text, 10) <= %s "
        'ORDER BY COALESCE("event_seq"::text, \'\') ASC, COALESCE("timestamp"::text, \'\') ASC'
    )
    try:  # event_seq 列不存在时查询会失败。
        rows = db.execute_query(query, (run_id, str(sim_date)[:10]))  # 参数化查询事件行。
    except Exception:  # 查询失败时交给上层走正常快照恢复。
        return 0

    seen = set()  # 记录已恢复事件的内容 key。
    unique_rows = []  # 保存去重后的事件行。
    for row in rows or []:  # 遍历查询结果；rows 为空时不处理。
        key = tuple("" if value is None else str(value) for value in row)  # 用四个日志字段构造去重 key。
        if key in seen:  # 完全相同事件已存在时跳过。
            continue
        seen.add(key)  # 记录新事件 key。
        unique_rows.append(row)  # 保留新事件行。

    return _assign_rows(orch, unique_rows)  # 将去重后的事件写回 orchestrator。


def _assign_rows(orch, rows) -> int:
    """将查询结果写入 ``orch.daily_logs``。

    参数：
        orch: 需要更新的 orchestrator 实例。
        rows: 由 ``timestamp/date/event_type/message`` 组成的查询结果行。

    返回：
        写入的日志记录数量。
    """
    records = []  # 保存转换后的日志 dict。
    for row in rows or []:  # 遍历 DB 查询行。
        record = {  # 转换为 orchestrator.daily_logs 使用的字段结构。
            "timestamp": row[0],
            "date": row[1],
            "event_type": row[2],
            "message": row[3],
        }
        for key, value in list(record.items()):  # 遍历字段值，处理 pandas 时间类型。
            if isinstance(value, pd.Timestamp):  # DB 返回 pandas Timestamp 时需要序列化为字符串。
                record[key] = value.isoformat()  # 转为 ISO 字符串，保持 JSON 兼容。
        records.append(record)  # 加入恢复结果列表。

    orch.daily_logs = records  # 覆盖 orchestrator 内存日志为 DB 恢复结果。
    return len(records)  # 返回恢复条数。


__all__ = [  # 明确模块公开 API。
    "DAILY_LOG_MODE",
    "DAILY_LOG_TABLE",
    "ensure_daily_log_ledger_schema",
    "initialize_daily_log_ledger_from_db",
]

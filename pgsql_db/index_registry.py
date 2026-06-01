"""PostgreSQL schema 内表的索引注册器。"""
from __future__ import annotations  # 延迟解析类型注解。

import logging  # 用于记录索引创建失败 warning。
from typing import TYPE_CHECKING, Iterable  # TYPE_CHECKING 避免运行期导入；Iterable 用于列集合参数。

if TYPE_CHECKING:  # 仅类型检查阶段导入，避免运行时循环依赖。
    from .db_connection import DatabaseConnection  # schema-aware 数据库连接类型。

logger = logging.getLogger("SupplyChainSimulation")  # 复用仿真主日志。


# 写入路径索引保持窄列集合，降低 COPY 写入维护成本。
_OUTPUT_LIGHTWEIGHT_INDEXES: dict[str, tuple[str, ...]] = {
    # 模块 1 输出表。
    "module1_output_orderlog": ("run_id", "sim_date"),  # 订单日志按 run 和仿真日查询/删除。
    "module1_output_shipmentlog": ("run_id", "sim_date"),  # 发运日志按 run 和仿真日查询/删除。
    "module1_output_cutlog": ("run_id", "sim_date"),  # cut 日志按 run 和仿真日查询/删除。
    "module1_output_supplydemandlog": ("run_id", "sim_date"),  # 供需日志按 run 和仿真日查询/删除。
    "module1_output_summary": ("run_id", "sim_date"),  # M1 summary 按 run 和仿真日查询。
    # 模块 3 输出表。
    "module3_output_netdemand": ("run_id", "sim_date"),  # 净需求输出按 run 和仿真日查询。
    # 模块 4 输出表。
    "module4_output_productionplan": ("run_id", "sim_date"),  # 生产计划按 run 和仿真日查询。
    "module4_output_capacityexceed": ("run_id", "sim_date"),  # 产能超限日志按 run 和仿真日查询。
    "module4_output_validation": ("run_id", "sim_date"),  # M4 校验日志按 run 和仿真日查询。
    "module4_output_changeoverlog": ("run_id", "sim_date"),  # changeover 日志按 run 和仿真日查询。
    # 模块 5 输出表。
    "module5_output_deploymentplan": ("run_id", "sim_date"),  # 调拨计划按 run 和仿真日查询。
    "module5_output_unfulfilledlog": ("run_id", "sim_date"),  # 未满足日志按 run 和仿真日查询。
    "module5_output_stockonhandlog": ("run_id", "sim_date"),  # 库存日志按 run 和仿真日查询。
    "module5_output_validation": ("run_id", "sim_date"),  # M5 校验日志按 run 和仿真日查询。
    # 模块 6 输出表。
    "module6_output_deliveryplan": ("run_id", "sim_date"),  # 配送计划按 run 和仿真日查询。
    "module6_output_vehiclelog": ("run_id", "sim_date"),  # 车辆日志按 run 和仿真日查询。
    "module6_output_truckusagelog": ("run_id", "sim_date"),  # 车辆使用日志按 run 和仿真日查询。
    "module6_output_unsatisfiedmdqlog": ("run_id", "sim_date"),  # 未满足 MDQ 日志按 run 和仿真日查询。
    "module6_output_validationlog": ("run_id", "sim_date"),  # M6 校验日志按 run 和仿真日查询。
    "module6_output_bypassrulehitlog": ("run_id", "sim_date"),  # bypass 规则命中日志按 run 和仿真日查询。
    # Orchestrator 状态快照表。
    "orchestrator_unrestricted_inventory": ("run_id", "sim_date"),  # unrestricted inventory 快照按 run 和仿真日查询。
    "orchestrator_open_deployment": ("run_id", "sim_date"),  # open deployment 快照按 run 和仿真日查询。
    "orchestrator_open_deployment_pastdue_cleanup": ("run_id", "sim_date"),  # 过期调拨清理审计按 run 和仿真日查询。
    "orchestrator_planning_intransit": ("run_id", "sim_date"),  # planning in-transit 快照按 run 和仿真日查询。
    "orchestrator_space_quota": ("run_id", "sim_date"),  # space quota 快照按 run 和仿真日查询。
    "orchestrator_delivery_gr": ("run_id", "sim_date"),  # delivery GR 快照按 run 和仿真日查询。
    "orchestrator_production_gr": ("run_id", "sim_date"),  # production GR 快照按 run 和仿真日查询。
    "orchestrator_production_plan_backlog": ("run_id", "sim_date"),  # 生产计划 backlog 按 run 和仿真日查询。
    "orchestrator_shipment_log": ("run_id", "sim_date"),  # orchestrator shipment 日志按 run 和仿真日查询。
    "orchestrator_delivery_shipment_log": ("run_id", "sim_date"),  # delivery shipment 日志按 run 和仿真日查询。
    "orchestrator_inventory_change_log": ("run_id", "sim_date"),  # 库存变化日志按 run 和仿真日查询。
    "orchestrator_daily_logs": ("run_id", "sim_date"),  # daily logs 累计快照按 run 和仿真日查询。
    # Summary 输出表生成后主要按 run_id 查询。
    "summary_output_ordershipmentcutsummary": ("run_id",),  # 订单/发运/cut 汇总按 run 查询。
    "summary_output_fullchangeoverlog": ("run_id",),  # 全量 changeover 汇总按 run 查询。
    "summary_output_fullcapacityexceed": ("run_id",),  # 全量产能超限汇总按 run 查询。
    "summary_output_fullproductionplan": ("run_id",),  # 全量生产计划汇总按 run 查询。
    "summary_output_fulldeploymentplan": ("run_id",),  # 全量调拨计划汇总按 run 查询。
    "summary_output_fulldeliveryplan": ("run_id",),  # 全量配送计划汇总按 run 查询。
    "summary_output_fulltruckusage": ("run_id",),  # 全量车辆使用汇总按 run 查询。
    "summary_historical_inventory_record": ("run_id",),  # 历史库存汇总按 run 查询。
}


# Summary 源表索引仅在 summary 聚合前创建。
_SUMMARY_WIDE_INDEXES: dict[str, tuple[str, ...]] = {
    "module1_output_orderlog": ("run_id", "date", "material", "location"),  # M1 订单聚合按 run/date/material/location。
    "module1_output_shipmentlog": ("run_id", "date", "material", "location"),  # M1 发运聚合按 run/date/material/location。
    "module1_output_cutlog": ("run_id", "date", "material", "location"),  # M1 cut 聚合按 run/date/material/location。
    "module4_output_changeoverlog": ("run_id", "changeover_end_date"),  # changeover 汇总按 run 和结束日期。
    "module4_output_capacityexceed": ("run_id", "date"),  # 产能超限汇总按 run 和日期。
    "module4_output_productionplan": ("run_id", "available_date"),  # 生产计划汇总按 run 和可用日期。
    "module5_output_deploymentplan": ("run_id", "date"),  # 调拨汇总按 run 和日期。
    "module6_output_deliveryplan": ("run_id", "actual_ship_date"),  # 配送汇总按 run 和实际发运日期。
    "module6_output_truckusagelog": ("run_id", "date"),  # 车辆使用汇总按 run 和日期。
}


def _table_has_all_columns(
    db: "DatabaseConnection",
    table_name: str,
    columns: Iterable[str],
) -> bool:
    """检查当前 schema 中的表是否包含全部目标列。

    参数：
        db: 带 schema 上下文的数据库连接。
        table_name: 未带 schema 的 PostgreSQL 表名。
        columns: 目标索引依赖的列名集合。

    返回：
        目标列全部存在时返回 True，否则返回 False。
    """
    try:  # information_schema 查询可能因连接或权限问题失败。
        rows = db.execute_query(  # 查询当前 schema 下目标表的全部列名。
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (db.schema, table_name),
        )
    except Exception:  # 查询失败时不创建索引，避免影响主流程。
        return False

    existing = {row[0] for row in rows or []}  # 转成集合，便于快速判断列是否存在。
    return all(col in existing for col in columns)  # 只有目标列全部存在时才允许建索引。


def _index_name(table_name: str, columns: tuple[str, ...], suffix: str) -> str:
    """生成稳定的索引名。

    参数：
        table_name: 未带 schema 的 PostgreSQL 表名。
        columns: 按索引顺序排列的列名。
        suffix: 索引用途后缀，例如 ``cfg``、``rw``、``sm``。

    返回：
        可重复生成的 PostgreSQL 索引名。
    """
    col_part = "_".join(columns)  # 将索引列名拼成索引名片段。
    return f"idx_{table_name}_{col_part}_{suffix}"  # 表名、列名、用途共同组成索引名，避免冲突。


def _create_index_if_not_exists(
    db: "DatabaseConnection",
    table_name: str,
    columns: tuple[str, ...],
    *,
    suffix: str,
    unique: bool = False,
    where: str | None = None,
) -> bool:
    """在表和列均存在时创建索引。

    参数：
        db: 带 schema 上下文的数据库连接。
        table_name: 未带 schema 的 PostgreSQL 表名。
        columns: 按索引顺序排列的列名。
        suffix: 索引用途后缀，用于生成索引名。
        unique: 为 True 时创建唯一索引。
        where: 可选的部分索引过滤条件。

    返回：
        索引已存在或创建成功时返回 True，否则返回 False。
    """
    if not _table_has_all_columns(db, table_name, columns):  # 缺表或缺列时跳过索引创建。
        return False

    idx_name = _index_name(table_name, columns, suffix)  # 生成稳定索引名。
    qualified_tbl = db.qualified_name(table_name)  # 生成带 schema 的目标表名。
    col_list = ", ".join(f'"{c}"' for c in columns)  # 拼接已加引号的索引列列表。
    unique_sql = "UNIQUE " if unique else ""  # 根据参数决定是否创建唯一索引。
    where_sql = f" WHERE {where}" if where else ""  # 根据参数决定是否创建部分索引。
    ddl = f'CREATE {unique_sql}INDEX IF NOT EXISTS "{idx_name}" ON {qualified_tbl} ({col_list}){where_sql};'  # 构造 DDL。

    try:  # 索引创建失败只影响性能，不应破坏仿真主流程。
        db.execute_non_query(ddl)  # 执行幂等索引创建语句。
        return True  # 执行成功或索引已存在均视为成功。
    except Exception as e:  # 捕获并记录失败原因。
        logger.warning(f"[index_registry] {idx_name} 创建失败：{e}")  # 输出索引创建 warning。
        return False  # 返回失败，让调用方统计时不计入成功数量。


def ensure_config_indexes(db: "DatabaseConnection") -> int:
    """确保配置表存在 ``config_name`` 查询索引。

    参数：
        db: 带 schema 上下文的数据库连接。

    返回：
        已存在或成功创建的索引数量。
    """
    from .config_manifest import MANIFEST_TABLE  # 延迟导入，避免模块初始化循环依赖。

    created = 0  # 记录成功创建或已存在的索引数量。
    try:  # 列表查询失败时直接返回 0。
        all_tables = db.get_all_tables()  # 获取当前 schema 下全部表名。
    except Exception as e:  # 连接或权限异常时进入兜底。
        logger.warning(f"[index_registry] 列举 schema {db.schema!r} 内表失败：{e}")  # 记录失败原因。
        return 0

    for tbl in all_tables:  # 遍历当前 schema 下所有表。
        if not (tbl.startswith("cfg_") or tbl == MANIFEST_TABLE):  # 只处理配置表和 manifest 表。
            continue
        if _create_index_if_not_exists(db, tbl, ("config_name",), suffix="cfg"):  # 创建 config_name 索引。
            created += 1  # 成功则累计数量。

    return created  # 返回成功索引数量。


def ensure_output_lightweight_indexes(db: "DatabaseConnection") -> int:
    """确保模块与 orchestrator 输出表存在写入路径轻量索引。

    参数：
        db: 带 schema 上下文的数据库连接。

    返回：
        已存在或成功创建的索引数量。
    """
    created = 0  # 记录成功创建或已存在的索引数量。
    for tbl, cols in _OUTPUT_LIGHTWEIGHT_INDEXES.items():  # 遍历写入路径轻量索引注册表。
        if _create_index_if_not_exists(db, tbl, cols, suffix="rw"):  # 创建 run/write 路径索引。
            created += 1  # 成功则累计数量。

    return created  # 返回成功索引数量。


def ensure_daily_log_ledger_indexes(db: "DatabaseConnection") -> int:
    """保留 daily logs 专用索引初始化的历史公开接口。

    参数：
        db: 带 schema 上下文的数据库连接；当前保留参数但不使用。

    返回：
        固定返回 0；daily logs 使用通用输出表索引注册逻辑。
    """
    return 0  # daily logs 已纳入 _OUTPUT_LIGHTWEIGHT_INDEXES，不再单独建索引。


def ensure_summary_wide_indexes(db: "DatabaseConnection") -> int:
    """确保 summary 聚合查询所需的宽索引存在。

    参数：
        db: 带 schema 上下文的数据库连接。

    返回：
        已存在或成功创建的索引数量。
    """
    created = 0  # 记录成功创建或已存在的索引数量。
    for tbl, cols in _SUMMARY_WIDE_INDEXES.items():  # 遍历 summary 宽索引注册表。
        if _create_index_if_not_exists(db, tbl, cols, suffix="sm"):  # 创建 summary 聚合索引。
            created += 1  # 成功则累计数量。

    return created  # 返回成功索引数量。


__all__ = [  # 明确模块公开 API。
    "ensure_config_indexes",
    "ensure_daily_log_ledger_indexes",
    "ensure_output_lightweight_indexes",
    "ensure_summary_wide_indexes",
]

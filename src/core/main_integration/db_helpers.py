"""
db_helpers.py

数据库辅助函数模块，用于 checkpoint 写入与批次数据管理。
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from psycopg import sql as psql
import pandas as pd

from ...utils.normalization import normalize_identifiers
from ...utils.numeric_safe import coerce_db_float, coerce_db_int

# 复用 src/utils/logger_config.py::DualLogger 创建的同名 logger，
# 这样消息既能进控制台又能进 simulation_log_*.txt。
logger = logging.getLogger("SupplyChainSimulation")


def _flush_batch_to_db(
    writer,
    batch_results: dict,
    run_id: str,
    db,
    run_key: str,
    config_name: str,
    start_date: str,
    end_date: str,
    batch_start_date: str,
    batch_end_date: str,
    orch,
    m1_previous_orders=None,
    runtime_state=None,
) -> None:
    """
    原子写入一个批次的模块结果到数据库，成功后更新 sim_checkpoint。

    真正的原子性保障：
    所有 DELETE + COPY + UPSERT 在同一个 PostgreSQL 事务中执行，
    任何步骤异常都会整体 ROLLBACK，确保数据一致性。

    Args:
        runtime_state: DbRuntimeState 实例。若提供，其序列化形式将写入
                       checkpoint JSON，以支持断点续跑。
    """
    from pgsql_db.checkpoint import serialize_orchestrator_state, _json_serializer

    logger.info(f"\n💾 批次写入: {batch_start_date} ~ {batch_end_date}")

    # 步骤 0：在事务外先序列化编排器状态（纯 CPU 操作，无 DB 交互）
    orch_state = serialize_orchestrator_state(orch)
    # m1_previous_orders is rebuilt from DB orderlog on resume to keep checkpoint small.
    # [DB-MEM] 将 DbRuntimeState 序列化进 checkpoint JSON（M4 line states, capacity, M3 result）
    if runtime_state is not None:
        orch_state['db_runtime_state'] = runtime_state.to_dict()
    orch_json = json.dumps(orch_state, default=_json_serializer, ensure_ascii=False)
    orch_json_bytes = len(orch_json.encode('utf-8'))
    if orch_json_bytes >= 16 * 1024 * 1024:
        logger.warning(f"  [WARN] checkpoint JSON size={orch_json_bytes / 1024 / 1024:.2f} MB")

    # 步骤 0b：在事务外预处理批次数据为可写入的 DataFrame
    prepared_tables = writer.prepare_batch_dataframes(batch_results, run_id=run_id)

    # 步骤 0b-2：预处理当日编排器数据为可写入的 DataFrame
    # [DB-MEM] 优先使用直接从 orchestrator 对象读取的视图（无需 CSV 中间文件），
    # 回退到基于 CSV 文件的兼容路径，以保持向后兼容。
    orchestrator_tables = prepare_orchestrator_day_dataframes_from_orch(
        orch=orch,
        run_id=run_id,
        sim_date=batch_end_date,
        config_name=config_name,
        cleanup_audit_df=getattr(runtime_state, 'cleanup_audit_df', None) if runtime_state else None,
    )
    if not orchestrator_tables:
        # 回退：尝试基于 CSV 文件的兼容路径（适用于仍会写出 CSV 的运行）
        orchestrator_tables = writer.prepare_orchestrator_day_dataframes(
            orchestrator_dir=str(orch.output_dir),
            run_id=run_id,
            sim_date=batch_end_date,
        )
    if orchestrator_tables:
        orch_total_rows = sum(df.shape[0] for df, _ in orchestrator_tables.values())
        logger.info(f"  📋 Orchestrator 当日数据: {len(orchestrator_tables)} 张表, 共 {orch_total_rows} 行")
        for tbl_name, (df, _) in orchestrator_tables.items():
            logger.info(f"    - {tbl_name}: {len(df)} 行")
    prepared_tables.update(orchestrator_tables)

    # 获取底层连接
    conn = db.connect()

    # 步骤 0c：在事务外预查表元数据，减少事务内 information_schema 查询
    # autocommit=True 模式下，SELECT 语句自动提交，不会开启隐式事务，
    # 因此后续 conn.transaction() 始终创建顶层 BEGIN...COMMIT 事务。
    table_meta = _precheck_table_metadata(conn)

    try:
        with conn.transaction():
            with conn.cursor() as cur:
                # 步骤 1：幂等清理（适用于断点续跑重试场景）
                _atomic_delete_batch(cur, run_id, batch_start_date, table_meta=table_meta)

                # 步骤 2：通过 COPY 写入模块结果
                _atomic_copy_batch(conn, cur, prepared_tables, db)

                # 步骤 3：UPSERT checkpoint
                cur.execute(
                    """
                    INSERT INTO sim_checkpoint
                        (run_key, run_id, config_name, start_date, end_date,
                         last_batch_end, orch_state_json, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, 'running')
                    ON CONFLICT (run_id) DO UPDATE SET
                        last_batch_end  = EXCLUDED.last_batch_end,
                        orch_state_json = EXCLUDED.orch_state_json,
                        status          = 'running',
                        updated_at      = NOW()
                    """,
                    (run_key, run_id, config_name, start_date,
                     end_date, batch_end_date, orch_json),
                )
        # 事务成功提交
        logger.info(f"  ✅ checkpoint 更新至 {batch_end_date}")
    except Exception:
        # conn.transaction() 退出时已自动 ROLLBACK
        logger.error(f"  ❌ 批次 {batch_start_date}~{batch_end_date} 写入失败，已回滚")
        raise


def _precheck_table_metadata(conn) -> dict:
    """在事务外一次性查询所有输出表的元数据（表存在性、列存在性）。

    返回：{表名: {'has_sim_date': bool, 'has_run_id': bool}}，不存在的表不出现在 dict 中
    """
    with conn.cursor() as cur:
        # 一次查询获取所有 public 表及其列
        cur.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND column_name IN ('sim_date', 'run_id')
        """)
        meta = {}
        for row in cur.fetchall():
            tname, cname = row[0], row[1]
            if tname not in meta:
                meta[tname] = {'has_sim_date': False, 'has_run_id': False}
            if cname == 'sim_date':
                meta[tname]['has_sim_date'] = True
            elif cname == 'run_id':
                meta[tname]['has_run_id'] = True
    return meta


def _atomic_delete_batch(cur, run_id: str, batch_start_date: str, table_meta: dict = None) -> None:
    """在已有事务内删除批次数据（含模块输出 + Summary + Orchestrator 表）。

    当 table_meta 已提供时，直接使用缓存的表元数据，
    避免在事务内执行大量 information_schema 查询（减少锁持有时间）。
    """
    # 完整表列表，与 truncate_output_tables 保持一致；按 sim_date + run_id 过滤
    output_tables = [
        # Module 输出表
        'module1_output_orderlog',
        'module1_output_shipmentlog',
        'module1_output_cutlog',
        'module1_output_supplydemandlog',
        'module1_output_summary',
        'module3_output_netdemand',
        'module4_output_productionplan',
        'module4_output_capacityexceed',
        'module4_output_validation',
        'module4_output_changeoverlog',
        'module5_output_deploymentplan',
        'module5_output_unfulfilledlog',
        'module5_output_stockonhandlog',
        'module5_output_validation',
        'module6_output_deliveryplan',
        'module6_output_vehiclelog',
        'module6_output_truckusagelog',
        'module6_output_unsatisfiedmdqlog',
        'module6_output_validationlog',
        'module6_output_bypassrulehitlog',
        # Summary 输出表
        'summary_output_ordershipmentcutsummary',
        'summary_output_fullchangeoverlog',
        'summary_output_fullcapacityexceed',
        'summary_output_fullproductionplan',
        'summary_output_fulldeploymentplan',
        'summary_output_fulldeliveryplan',
        'summary_output_fulltruckusage',
        # Orchestrator 状态表
        'orchestrator_unrestricted_inventory',
        'orchestrator_open_deployment',
        'orchestrator_open_deployment_pastdue_cleanup',
        'orchestrator_planning_intransit',
        'orchestrator_space_quota',
        'orchestrator_delivery_gr',
        'orchestrator_production_gr',
        'orchestrator_production_plan_backlog',
        'orchestrator_shipment_log',
        'orchestrator_delivery_shipment_log',
        'orchestrator_inventory_change_log',
        'orchestrator_daily_logs',
    ]
    deleted_total = 0
    for table_name in output_tables:
        # 使用缓存的元数据（如果可用），否则回退到事务内查询
        if table_meta is not None:
            meta = table_meta.get(table_name)
            if meta is None:
                continue  # 表不存在
            has_sim_date = meta['has_sim_date']
            has_run_id = meta['has_run_id']
        else:
            # 回退：事务内查询 information_schema
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = %s)",
                (table_name,)
            )
            if not cur.fetchone()[0]:
                continue
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = 'sim_date')",
                (table_name,)
            )
            has_sim_date = cur.fetchone()[0]
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = 'run_id')",
                (table_name,)
            )
            has_run_id = cur.fetchone()[0]

        if not has_sim_date:
            continue
        # 按 sim_date + run_id 删除
        if has_run_id and run_id:
            cur.execute(
                psql.SQL('DELETE FROM {} WHERE sim_date >= %s AND run_id = %s').format(
                    psql.Identifier(table_name)
                ),
                (batch_start_date, run_id)
            )
        else:
            cur.execute(
                psql.SQL('DELETE FROM {} WHERE sim_date >= %s').format(
                    psql.Identifier(table_name)
                ),
                (batch_start_date,)
            )
        deleted_total += 1
    if deleted_total > 0:
        logger.info(f"  🗑️ 已清理批次 {batch_start_date} 起的旧数据（{deleted_total} 张表）")


def _atomic_copy_batch(conn, cur, prepared_tables: dict, db) -> None:
    """在已有事务内使用 COPY 批量写入预处理好的数据。"""
    for table_name, (df, clean_columns) in prepared_tables.items():
        if df.empty:
            continue
        # 确保表存在（在事务内创建）
        if not _table_exists_in_txn(cur, table_name):
            columns = []
            for col_name, dtype in df.dtypes.items():
                clean_col = db._clean_name(str(col_name))
                pg_type = db._pandas_to_pg_type(dtype, col_name=clean_col)
                columns.append(f'"{clean_col}" {pg_type}')
            cur.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns)})')

        # 准备数据
        col_types = {}
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s",
            (table_name,)
        )
        for row in cur.fetchall():
            col_types[row[0]] = row[1]

        # 自动添加表中不存在的列（不同 config 的模块输出列可能不同）
        if col_types:  # 表已存在时才需要检查
            for col_name in clean_columns:
                if col_name not in col_types:
                    # 根据 DataFrame 中该列的 dtype 推断 PG 类型
                    col_idx = list(clean_columns).index(col_name)
                    df_dtype = df.iloc[:, col_idx].dtype
                    pg_type = db._pandas_to_pg_type(df_dtype, col_name=col_name)
                    cur.execute(f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS "{col_name}" {pg_type}')
                    col_types[col_name] = pg_type
                    logger.info(f'    [ALTER] 为 {table_name} 添加新列: {col_name} ({pg_type})')
        import numpy as np
        records = df.values.tolist()
        col_name_to_idx = {col: idx for idx, col in enumerate(clean_columns)}

        # 列类型升级：BIGINT → DOUBLE PRECISION（防止浮点截断）；
        # 以及 TEXT → BOOLEAN/DOUBLE/BIGINT（修复历史上被误建为 TEXT 的 bool/数值列）。
        _INT_TYPES = {'BIGINT', 'INTEGER', 'SMALLINT', 'INT', 'INT4', 'INT8', 'INT2'}
        _TEXT_TYPES = {'TEXT', 'VARCHAR', 'CHARACTER VARYING', 'CHAR', 'CHARACTER'}
        _TEXT_UPGRADE_CAST = {
            'BOOLEAN': '::BOOLEAN',
            'DOUBLE PRECISION': '::DOUBLE PRECISION',
            'BIGINT': '::DOUBLE PRECISION::BIGINT',  # 经 DOUBLE 中转，兼容 '2' 与 '2.0'
        }
        for cn, ct in list(col_types.items()):
            if cn in col_name_to_idx:
                col_idx = col_name_to_idx[cn]
                if col_idx < len(df.columns):
                    df_dtype = df.iloc[:, col_idx].dtype
                    expected_pg = db._pandas_to_pg_type(df_dtype, col_name=cn)
                    ctu = ct.upper()
                    if expected_pg == 'DOUBLE PRECISION' and ctu in _INT_TYPES:
                        cur.execute(f'ALTER TABLE "{table_name}" ALTER COLUMN "{cn}" TYPE DOUBLE PRECISION USING "{cn}"::DOUBLE PRECISION')
                        col_types[cn] = 'DOUBLE PRECISION'
                    elif ctu in _TEXT_TYPES and expected_pg in _TEXT_UPGRADE_CAST:
                        # 共享事务内用 SAVEPOINT 保护：cast 失败仅回滚本列并保持 TEXT，不废整批
                        cast = _TEXT_UPGRADE_CAST[expected_pg]
                        try:
                            cur.execute('SAVEPOINT sp_text_upg')
                            cur.execute(f'ALTER TABLE "{table_name}" ALTER COLUMN "{cn}" TYPE {expected_pg} USING "{cn}"{cast}')
                            cur.execute('RELEASE SAVEPOINT sp_text_upg')
                            col_types[cn] = expected_pg
                            logger.info(f'    [UPGRADE] {table_name}.{cn} TEXT→{expected_pg}')
                        except Exception as exc:
                            cur.execute('ROLLBACK TO SAVEPOINT sp_text_upg')
                            logger.warning(f'    [UPGRADE-SKIP] {table_name}.{cn} 保持 TEXT（cast 失败: {exc}）')

        int_col_indices = set()
        float_col_indices = set()
        text_col_indices = set()
        bool_col_indices = set()
        for cn, ct in col_types.items():
            if cn in col_name_to_idx:
                idx = col_name_to_idx[cn]
                ctu = ct.upper()
                if ctu in ('BIGINT', 'INTEGER', 'SMALLINT', 'INT', 'INT4', 'INT8', 'INT2'):
                    int_col_indices.add(idx)
                elif ctu in ('DOUBLE PRECISION', 'REAL', 'NUMERIC', 'FLOAT4', 'FLOAT8'):
                    float_col_indices.add(idx)
                elif ctu in ('TEXT', 'VARCHAR', 'CHARACTER VARYING', 'CHAR', 'CHARACTER'):
                    text_col_indices.add(idx)
                elif ctu in ('BOOLEAN', 'BOOL'):
                    bool_col_indices.add(idx)

        for i, row in enumerate(records):
            new_row = []
            for j, val in enumerate(row):
                if j in bool_col_indices:
                    # BOOLEAN 列：兼容 np.bool_/Python bool、历史 'True'/'False' 文本、0/1 数值
                    if isinstance(val, (bool, np.bool_)):
                        new_row.append(bool(val))
                    elif val is None:
                        new_row.append(None)
                    elif isinstance(val, str):
                        s = val.strip().lower()
                        new_row.append(
                            True if s in ('true', 't', '1', 'yes', 'y')
                            else False if s in ('false', 'f', '0', 'no', 'n')
                            else None
                        )
                    else:
                        try:
                            new_row.append(None if pd.isna(val) else bool(val))
                        except (TypeError, ValueError):
                            new_row.append(None)
                elif j in int_col_indices:
                    new_row.append(coerce_db_int(val))
                elif j in float_col_indices:
                    new_row.append(coerce_db_float(val))
                elif pd.isna(val) if not isinstance(val, str) else False:
                    new_row.append(None)
                elif j in text_col_indices:
                    # 布尔值转为 "True"/"False" 字符串，与 Dev/Src xlsx 输出格式保持一致
                    # （避免 PG bool 转 text 时产生 "t"/"f"）
                    if isinstance(val, (bool, np.bool_)):
                        new_row.append(str(val))
                    else:
                        new_row.append(str(val) if val is not None else None)
                else:
                    new_row.append(val)
            records[i] = tuple(new_row)

        cols_str = ", ".join([f'"{col}"' for col in clean_columns])
        copy_sql = f'COPY "{table_name}" ({cols_str}) FROM STDIN'
        with cur.copy(copy_sql) as copy:
            for record in records:
                copy.write_row(record)


# ---------------------------------------------------------------------------
# [DB-MEM] 直接从 orchestrator 视图构建 DB DataFrames（无需 CSV 中间文件）
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """列名清洗函数 — 与 ModuleDataWriter._clean_name() 实现一致。"""
    clean = name.replace(" ", "_").replace("-", "_").replace(".", "_")
    clean = "".join(c for c in clean if c.isalnum() or c == "_")
    if clean and clean[0].isdigit():
        clean = "_" + clean
    return clean.lower()


def prepare_orchestrator_day_dataframes_from_orch(
    orch,
    run_id: Optional[str],
    sim_date: str,
    config_name: str = "",
    cleanup_audit_df: Optional[pd.DataFrame] = None,
) -> Dict[str, tuple]:
    """直接从 orchestrator 视图构建当日 DataFrame（无需 CSV 中间文件）。

    替代基于 CSV 的 ``ModuleDataWriter.prepare_orchestrator_day_dataframes()``，
    在数据库模式下消除批次写入数据库前调用 ``orch.save_daily_state()`` 写出 CSV 中间文件的依赖。

    输出结构与 CSV 兼容路径保持相同的表命名和元数据列约定：
        ``{表名: (DataFrame, 清洗后列名列表)}``

    列处理规则：
    - 除 daily_logs 外，所有视图均应用 ``normalize_identifiers()``
      （与 ``persistence.py:save_daily_state`` 保持一致）。
    - 追加 5 列元数据：file_date、sim_date、run_id、config_name、db_write_time。
    - 所有列名均应用 ``_clean_name()`` 清洗。
    """
    from ...utils.normalization import normalize_identifiers

    date_key = pd.to_datetime(sim_date).strftime("%Y%m%d")
    sim_date_str = pd.to_datetime(sim_date).strftime("%Y-%m-%d")

    # ---- 构建 (基础名, DataFrame, 是否规范化) 元组列表 ----
    date_arg = sim_date_str  # 大多数视图方法接受 YYYY-MM-DD 格式

    raw_views: List[tuple] = []

    # 1. unrestricted_inventory
    raw_views.append((
        "unrestricted_inventory",
        _safe_view(orch, "get_unrestricted_inventory_view", date_arg),
        True,
    ))
    # 2. open_deployment
    raw_views.append((
        "open_deployment",
        _safe_view(orch, "get_open_deployment_view", date_arg),
        True,
    ))
    # 3. planning_intransit
    raw_views.append((
        "planning_intransit",
        _safe_view(orch, "get_planning_intransit_view", date_arg),
        True,
    ))
    # 4. space_quota
    raw_views.append((
        "space_quota",
        _safe_view(orch, "get_space_quota_view", date_arg),
        True,
    ))
    # 5. production_plan_backlog
    raw_views.append((
        "production_plan_backlog",
        _safe_view(orch, "get_production_plan_backlog_view", date_arg),
        True,
    ))
    # 6. delivery_gr
    raw_views.append((
        "delivery_gr",
        _safe_view(orch, "get_delivery_gr_view", date_arg),
        True,
    ))
    # 7. production_gr
    raw_views.append((
        "production_gr",
        _safe_view(orch, "get_production_gr_view", date_arg),
        True,
    ))
    # 8. shipment_log（从字典构建，与 persistence.py 相同）
    _daily_shipments = getattr(orch, 'shipment_log_by_date', {}).get(sim_date_str, [])
    _shipment_df = pd.DataFrame(_daily_shipments)
    if _shipment_df.empty:
        _shipment_df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
    raw_views.append(("shipment_log", _shipment_df, True))

    # 9. delivery_shipment_log（从字典构建，与 persistence.py 相同）
    _daily_dlv_shipments = getattr(orch, 'delivery_shipment_log_by_date', {}).get(sim_date_str, [])
    _dlv_ship_df = pd.DataFrame(_daily_dlv_shipments)
    if _dlv_ship_df.empty:
        _dlv_ship_df = pd.DataFrame(columns=[
            'date', 'material', 'sending', 'receiving', 'quantity',
            'ori_deployment_uid', 'actual_ship_date', 'actual_delivery_date', 'type',
        ])
    raw_views.append(("delivery_shipment_log", _dlv_ship_df, True))

    # 10. inventory_change_log
    raw_views.append((
        "inventory_change_log",
        _safe_view(orch, "generate_inventory_change_log", date_arg),
        True,
    ))
    # 11. daily_logs（不做规范化 — 与 persistence.py 一致）
    _daily_logs = getattr(orch, 'daily_logs', [])
    _logs_df = pd.DataFrame(_daily_logs) if _daily_logs else pd.DataFrame(
        columns=['timestamp', 'date', 'event_type', 'message']
    )
    raw_views.append(("daily_logs", _logs_df, False))

    # 12. open_deployment_pastdue_cleanup (written separately by cleanup_past_due_open_deployments)
    if cleanup_audit_df is not None and not cleanup_audit_df.empty:
        raw_views.append(("open_deployment_pastdue_cleanup", cleanup_audit_df.copy(), True))

    # ---- 应用规范化 + 元数据 + 列名清洗 ----
    prepared: Dict[str, tuple] = {}
    for base_name, df, apply_norm in raw_views:
        if df is None or df.empty:
            continue

        df = df.copy()
        if apply_norm:
            df = normalize_identifiers(df)

        # 追加元数据列（与 ModuleDataWriter.prepare_orchestrator_day_dataframes 相同）
        df["file_date"] = date_key
        df["sim_date"] = sim_date_str
        if run_id:
            df["run_id"] = run_id
        if config_name:
            df["config_name"] = config_name
        df["db_write_time"] = datetime.now()

        clean_columns = [_clean_name(str(col)) for col in df.columns]
        df.columns = clean_columns
        prepared[f"orchestrator_{base_name}"] = (df, clean_columns)

    return prepared


def _safe_view(orch, method_name: str, date_arg: str) -> pd.DataFrame:
    """安全调用 orchestrator 视图方法，出错时返回空 DataFrame。"""
    method = getattr(orch, method_name, None)
    if method is None:
        return pd.DataFrame()
    try:
        result = method(date_arg)
        if result is None:
            return pd.DataFrame()
        return result
    except Exception as e:
        logger.warning(f"  [警告] {method_name}({date_arg}) 调用失败: {e}")
        raise


def _table_exists_in_txn(cur, table_name: str) -> bool:
    """在事务内检查表是否存在。"""
    cur.execute(
        "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s)",
        (table_name,)
    )
    return cur.fetchone()[0]

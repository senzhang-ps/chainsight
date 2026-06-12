"""
db_runner.py

数据库模式运行器
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .db_config import _load_config_from_database
from .local_writer import _write_results_to_local_dev_format
from .utils import resolve_excel_path, _PROJECT_ROOT


def _normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """配置表比较前标准化：列排序、缺失统一、转字符串并排序行。"""
    if df is None:
        return pd.DataFrame()

    norm = df.copy()
    norm.columns = [str(c) for c in norm.columns]
    norm = norm.reindex(sorted(norm.columns), axis=1)
    norm = norm.where(pd.notna(norm), None)
    norm = norm.astype(str)
    if len(norm.columns) > 0 and not norm.empty:
        norm = norm.sort_values(by=list(norm.columns), kind="mergesort").reset_index(drop=True)
    else:
        norm = norm.reset_index(drop=True)
    return norm


def _find_local_config_file(config_arg: str) -> Path | None:
    """按用户输入的 ``--config`` 参数定位本地 Excel 配置文件。

    支持名字、Excel 路径、以及包含唯一 Excel 的目录路径。
    """
    raw = Path(config_arg).expanduser()
    if raw.is_dir():
        from .config_dir import ConfigDir

        return ConfigDir.from_path(raw.resolve()).excel_path
    return resolve_excel_path(config_arg)


def _resolve_db_config_input(config_arg: str) -> tuple[str, Path | None]:
    """解析数据库模式 ``--config`` 输入。

    返回 ``(config_name, local_config_file)``：
    - 目录路径：使用目录内唯一 Excel，config_name 取 Excel stem。
    - Excel 路径或可解析名字：使用解析出的 Excel，config_name 取 Excel stem。
    - 纯数据库配置名：保留原 stem，local_config_file 为 None。
    """
    local_config_file = _find_local_config_file(config_arg)
    if local_config_file is not None:
        return local_config_file.stem, local_config_file
    return Path(config_arg).stem, None


def _build_expected_local_config(
    config_arg: str,
    *,
    config_name: str | None = None,
    report_dir: str | Path | None = None,
    logger=None,
    input_func=input,
) -> dict:
    """构建本地期望配置（Excel + 同目录 CSV 无条件优先）。

    ``config_arg`` 是用户原始输入（名字 / 相对路径 / 绝对路径）。
    流程：先完整执行输入数据质量检测并输出全部问题记录与报告；存在
    ERROR 时由操作员确认是否继续入库（N 则停止运行）；收到入库确认后
    才对 M1_DemandForecast 按主键合并 quantity，最后以 schema 表名去
    ``cfg_`` 前缀为 key 返回数据。列名/类型到数据库契约的转换留待后续。
    """
    config_file = _find_local_config_file(config_arg)
    if config_file is None:
        return {}

    from pgsql_db.config_table_schema import get_config_table_schema_by_sheet
    from pgsql_db.excel_importer import (
        ExcelImporter,
        merge_m1_demandforecast_quantity,
    )
    from src.utils.data_quality import ConfigInputDataQualityChecker

    sheet_data = ExcelImporter.load_excel_file_with_csv_priority(str(config_file))
    checker = ConfigInputDataQualityChecker()
    dq_result = checker.validate(
        sheet_data,
        config_name=config_name or config_file.stem,
        report_dir=report_dir,
    )

    # 检测全部完成、错误已统一输出后，由操作员确认是否继续入库。
    if not dq_result["passed"] and not _confirm_import_after_dq(
        dq_result,
        report_dir=report_dir,
        logger=logger,
        input_func=input_func,
    ):
        raise RuntimeError(
            "[DQ] 操作员未确认入库，运行已停止；问题明细见 input_quality.xlsx"
        )

    # 收到入库确认后才执行 M1_DemandForecast 主键合并（quantity 求和）。
    if "M1_DemandForecast" in sheet_data:
        sheet_data["M1_DemandForecast"] = merge_m1_demandforecast_quantity(
            sheet_data["M1_DemandForecast"]
        )

    expected_config: dict[str, pd.DataFrame] = {}
    for sheet_name, df in sheet_data.items():
        # schema 未声明的 Sheet 不参与配置比对与入库。
        table_schema = get_config_table_schema_by_sheet(sheet_name)
        if table_schema is None:
            continue
        expected_config[str(table_schema["db_table"]).removeprefix("cfg_")] = df
    return expected_config


def _log_or_print(logger, level: str, message: str) -> None:
    """有 logger 时按级别记录，否则直接打印（交互确认场景兜底）。"""
    if logger is not None:
        getattr(logger, level)(message)
    else:
        print(message)


def _confirm_import_after_dq(
    dq_result: dict,
    *,
    report_dir: str | Path | None,
    logger=None,
    input_func=input,
) -> bool:
    """检测发现 ERROR 后，要求操作员确认是否继续入库。

    Args:
        dq_result: ``ConfigInputDataQualityChecker.validate`` 的检测结果。
        report_dir: 质量报告目录，用于在提示中给出报告路径。
        logger: 可选日志对象；为空时直接打印。
        input_func: 交互输入函数，测试时可注入。

    Returns:
        True 表示操作员确认继续入库；N 或未收到输入返回 False。
    """
    summary = dq_result.get("summary", {})
    _log_or_print(logger, "error", "[DQ] 输入配置表数据质量检测发现问题。")
    _log_or_print(
        logger,
        "error",
        "[DQ] summary: issues={issues}, errors={errors}, warnings={warnings}; "
        "按检测项: {by_check}; 按Sheet: {by_sheet}".format(
            issues=summary.get("issues", 0),
            errors=summary.get("errors", 0),
            warnings=summary.get("warnings", 0),
            by_check=summary.get("by_check", {}),
            by_sheet=summary.get("by_sheet", {}),
        ),
    )
    if report_dir is not None:
        report_xlsx = Path(report_dir) / "input_quality.xlsx"
        if report_xlsx.exists():
            _log_or_print(logger, "error", f"[DQ] 数据质量检测报告已输出: {report_xlsx}")
        else:
            _log_or_print(logger, "error", f"[DQ] 数据质量检测报告尚未生成: {report_xlsx}")
    _log_or_print(
        logger,
        "warning",
        "[DQ] 确认入库后将合并 M1_DemandForecast 重复主键的 quantity 并继续同步数据库。",
    )

    prompt = "是否继续入库并往下执行？请输入 Y/N: "
    while True:
        try:
            answer = input_func(prompt)
        except (EOFError, KeyboardInterrupt):
            _log_or_print(logger, "error", "[DQ] 未收到操作员确认，已停止执行。")
            return False
        normalized = str(answer).strip()
        if normalized in {"Y", "y"}:
            _log_or_print(logger, "warning", "[DQ] 操作员确认继续入库。")
            return True
        if normalized in {"N", "n"}:
            _log_or_print(logger, "error", "[DQ] 操作员选择停止执行。")
            return False
        _log_or_print(logger, "warning", "[DQ] 输入无效，请输入 Y/y 或 N/n。")


def _resolve_local_config_source(config_arg: str) -> Optional[Path]:
    """复用 ``_find_local_config_file`` 拿到本地配置文件路径（便于附加 mtime/source）。"""
    return _find_local_config_file(config_arg)


def _diff_local_vs_db_config(expected_config: dict, db_config: dict) -> dict:
    """比较本地期望配置与DB配置，返回差异分类。"""
    expected_keys = set(expected_config.keys())
    db_keys = set(db_config.keys())

    new_tables = sorted(expected_keys - db_keys)
    deleted_tables = sorted(db_keys - expected_keys)

    unchanged_tables = []
    changed_tables = []

    for key in sorted(expected_keys & db_keys):
        left = _normalize_for_compare(expected_config.get(key, pd.DataFrame()))
        right = _normalize_for_compare(db_config.get(key, pd.DataFrame()))
        if left.equals(right):
            unchanged_tables.append(key)
        else:
            changed_tables.append(key)

    return {
        "new_tables": new_tables,
        "deleted_tables": deleted_tables,
        "changed_tables": changed_tables,
        "unchanged_tables": sorted(unchanged_tables),
    }


def _diff_local_vs_db_via_manifest(
    expected_config: dict,
    expected_hashes: dict[str, str],
    manifest: dict[str, dict],
    db,
    config_name: str,
    logger,
) -> dict:
    """基于 manifest hash 快速判定差异；仅对疑似变更表读取 DB 明细做兜底校验。

    **键约定**：
    - ``expected_config`` / ``expected_hashes`` 用 *db_key*（无 ``cfg_`` 前缀，如
      ``global_seed``）作 key——这是 :func:`_build_expected_local_config` 已剥
      前缀后的产物，也是下游 :func:`_sync_config_by_diff` 期望接收的键形态。
    - ``manifest`` 用 *table_name*（带 ``cfg_`` 前缀，如 ``cfg_global_seed``）作
      key——这是物理表名，与 PG 中真实存在的表对得上。

    本函数内部把两者统一转换为 db_key 形态做集合运算，输出依旧是 db_key 形态，
    避免 :func:`_sync_config_by_diff` 中 ``f"cfg_{db_key}"`` 再次拼接形成
    ``cfg_cfg_xxx`` 的双前缀，也避免每次运行把所有表错判成新增/删除。

    流程：
      1. 用预先算好的 ``expected_hashes`` 与 manifest 中的 ``content_hash`` 比较。
      2. hash 一致 -> ``unchanged_tables``，不读 DB。
      3. hash 不一致 -> 只读取该 ``cfg_*`` 表 ``WHERE config_name = %s`` 的明细，
         经 :func:`_normalize_for_compare` 标准化后再判定 changed / unchanged。
      4. ``expected - manifest`` -> ``new_tables``；``manifest - expected`` -> ``deleted_tables``。
    """

    # 把 manifest 的物理 table_name 还原回 db_key（去掉 cfg_ 前缀），统一基线。
    # 非 cfg_ 前缀的 manifest 行视为遗留数据，跳过——它们不来自正规配置导入路径。
    manifest_by_dbkey: dict[str, dict] = {}
    for table_name, entry in manifest.items():
        if not isinstance(table_name, str) or not table_name.startswith("cfg_"):
            continue
        manifest_by_dbkey[table_name[4:]] = entry

    expected_keys = set(expected_config.keys())
    manifest_keys = set(manifest_by_dbkey.keys())

    new_tables = sorted(expected_keys - manifest_keys)
    deleted_tables = sorted(manifest_keys - expected_keys)
    changed_tables: list[str] = []
    unchanged_tables: list[str] = []

    fallback_read_count = 0
    for db_key in sorted(expected_keys & manifest_keys):
        local_hash = expected_hashes.get(db_key)
        remote_hash = manifest_by_dbkey[db_key].get("content_hash")
        if local_hash and remote_hash and local_hash == remote_hash:
            unchanged_tables.append(db_key)
            continue

        # hash 不一致：只读取该 cfg_* 表中当前 config_name 的明细做兜底比对，
        # 防止 manifest 落后于实际数据（例如手工写库）时误判为变更。
        table_name = f"cfg_{db_key}"
        fallback_read_count += 1
        try:
            df = db.read_table(table_name, filters={"config_name": config_name})
        except Exception as e:
            logger.warning(f"  [manifest] 读取 {table_name} 兜底比对失败：{e}")
            changed_tables.append(db_key)
            continue

        drop_cols = [
            c for c in df.columns
            if c in ("config_name", "config_type", "db_write_time")
            or str(c).startswith("unnamed") or str(c).startswith("Unnamed")
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        left = _normalize_for_compare(expected_config.get(db_key, pd.DataFrame()))
        right = _normalize_for_compare(df)
        if left.equals(right):
            unchanged_tables.append(db_key)
        else:
            changed_tables.append(db_key)

    if fallback_read_count:
        logger.info(
            f"  [manifest] hash 不一致触发兜底读取 {fallback_read_count} 张 cfg_* 明细"
        )

    return {
        "new_tables": new_tables,
        "deleted_tables": deleted_tables,
        "changed_tables": changed_tables,
        "unchanged_tables": sorted(unchanged_tables),
    }


def _verify_cfg_table_persisted(db, table_name: str, config_name: str, expect_rows: bool) -> bool:
    """确认某个 ``cfg_*`` 表对当前 config_name 已成功持久化到 DB。

    用于 manifest 写入前的把关，避免配置表部分写入失败后仍把 manifest 标成成功，
    导致 manifest 与 DB 状态分裂。

    判定规则：
    - 表不存在 -> ``False``。
    - 本地 DataFrame 为空 (``expect_rows=False``)：表存在即视为成功（首次导入
      会为空 sheet 也建立 header-only 表）。
    - 本地 DataFrame 非空：必须能在 DB 中找到至少一行 ``config_name = %s``。
    """
    table_exists = getattr(db, "table_exists", None)
    if not callable(table_exists) or not table_exists(table_name):
        return False
    if not expect_rows:
        return True
    try:
        qualified = db.qualified_name(table_name)
        rows = db.execute_query(
            f"SELECT 1 FROM {qualified} WHERE config_name = %s LIMIT 1",
            (config_name,),
        )
        return bool(rows)
    except Exception:
        return False


def _precheck_config_table_write(db, table_name: str, df: pd.DataFrame, config_name: str) -> None:
    """Fail before deleting old config rows when an existing table is incompatible."""
    table_exists = getattr(db, "table_exists", None)
    check_compatible = getattr(db, "_check_table_compatible", None)
    if not callable(table_exists) or not callable(check_compatible):
        return
    if not table_exists(table_name):
        return

    candidate = df.copy()
    if candidate.empty:
        candidate["config_name"] = pd.Series(dtype="object")
        candidate["db_write_time"] = pd.Series(dtype="datetime64[ns]")
    else:
        candidate["config_name"] = config_name
        candidate["db_write_time"] = datetime.now()

    if not check_compatible(table_name, candidate):
        raise RuntimeError(
            f"Config table {table_name} is incompatible; old rows were not deleted."
        )


def _sync_config_by_diff(
    db,
    config_name: str,
    expected_config: dict,
    diff_result: dict,
    logger,
    *,
    expected_hashes: dict[str, str] | None = None,
    source_file: Path | None = None,
    table_comment_map: dict[str, dict] | None = None,
) -> None:
    """按差异最小化同步：仅更新新增/变更表，并删除本地已不存在的表数据。

    若提供 ``expected_hashes`` / ``source_file``，会同步更新
    ``cfg_import_manifest``，让下次比对直接命中 hash 快速路径。
    """
    from pgsql_db.config_manifest import (
        ensure_manifest_table,
        upsert_manifest_entry,
        delete_manifest_entry,
    )

    config_basename = Path(config_name).stem

    to_upsert = sorted(set(diff_result["new_tables"] + diff_result["changed_tables"]))
    to_delete = diff_result["deleted_tables"]

    manifest_enabled = expected_hashes is not None
    if manifest_enabled:
        # 幂等创建 manifest 表，确保 schema 切换后仍可用。
        ensure_manifest_table(db)

    source_mtime: Optional[datetime] = None
    source_path_str: Optional[str] = None
    if source_file is not None:
        try:
            source_mtime = datetime.fromtimestamp(source_file.stat().st_mtime)
            source_path_str = str(source_file)
        except OSError:
            source_mtime = None
            source_path_str = str(source_file)

    # 单表失败只记录不中断，所有表都尝试完成后统一报错；
    # 写入异常时 db_connection 的 conn.transaction() 已自动回滚，连接仍可用。
    failures: list[tuple[str, str]] = []

    for db_key in to_upsert:
        table_name = f"cfg_{db_key}"
        df = expected_config.get(db_key, pd.DataFrame())
        comment_meta = (table_comment_map or {}).get(table_name, {})
        try:
            _precheck_config_table_write(db, table_name, df, config_basename)
            deleted = db.delete_config_data(table_name, config_basename)
            write_ok = db.create_table_from_df(
                df,
                table_name,
                if_exists="append",
                config_name=config_basename,
                table_comment=comment_meta.get("table_comment"),
                column_comments=comment_meta.get("column_comments"),
                column_types=comment_meta.get("column_types"),
                primary_key_columns=comment_meta.get("primary_key_columns"),
                index_columns=comment_meta.get("index_columns"),
            )
            if not write_ok:
                raise RuntimeError("create_table_from_df returned False")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ❌ 配置表 {table_name} 同步失败（继续同步其余表）: {exc}")
            failures.append((table_name, str(exc)))
            continue
        logger.info(f"  🔄 同步配置表 {table_name}: 删除 {deleted} 行，写入 {len(df)} 行")
        if manifest_enabled:
            upsert_manifest_entry(
                db,
                config_name=config_basename,
                table_name=table_name,
                row_count=len(df),
                content_hash=expected_hashes.get(db_key, ""),
                source_file=source_path_str,
                source_mtime=source_mtime,
            )

    for db_key in to_delete:
        table_name = f"cfg_{db_key}"
        try:
            deleted = db.delete_config_data(table_name, config_basename)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  ❌ 配置表 {table_name} 数据删除失败（继续处理其余表）: {exc}")
            failures.append((table_name, str(exc)))
            continue
        logger.info(f"  🗑️ 删除本地已不存在配置表数据 {table_name}: 删除 {deleted} 行")
        if manifest_enabled:
            delete_manifest_entry(
                db,
                config_name=config_basename,
                table_name=table_name,
            )

    # 全部表尝试完成后统一报错：配置未完整入库时停止后续流程（不跑模拟）。
    if failures:
        logger.error(
            "[SYNC] 配置同步完成，但有 %s 张表失败：%s",
            len(failures),
            "；".join(f"{name}: {reason}" for name, reason in failures),
        )
        raise RuntimeError(
            f"Config sync failed for {len(failures)} table(s): "
            + ", ".join(name for name, _ in failures)
            + "; simulation is aborted."
        )


def _refresh_config_table_comments(
    db,
    expected_config: dict,
    table_comment_map: dict[str, dict],
    logger,
) -> None:
    """刷新所有期望配置表的 PostgreSQL 表/字段注释。"""
    if not table_comment_map or not hasattr(db, "apply_table_comments"):
        return

    refreshed = 0
    for db_key in sorted(expected_config):
        table_name = f"cfg_{db_key}"
        comment_meta = table_comment_map.get(table_name)
        if not comment_meta:
            continue
        try:
            db.apply_table_comments(
                table_name,
                table_comment=comment_meta.get("table_comment"),
                column_comments=comment_meta.get("column_comments"),
            )
            refreshed += 1
        except Exception as exc:
            logger.warning(f"  [COMMENT] 刷新配置表注释失败 {table_name}: {exc}")

    if refreshed:
        logger.info(f"  [COMMENT] 已刷新配置表注释 {refreshed} 张")


def _drop_legacy_config_type_columns(db, expected_config: dict, logger) -> None:
    """删除旧版配置表中冗余的 config_type 列。"""
    drop_column = getattr(db, "drop_column_if_exists", None)
    if not callable(drop_column):
        return

    dropped = 0
    for db_key in sorted(expected_config):
        table_name = f"cfg_{db_key}"
        try:
            drop_column(table_name, "config_type")
            dropped += 1
        except Exception as exc:
            logger.warning(f"  [SCHEMA] 删除旧字段 {table_name}.config_type 失败: {exc}")

    if dropped:
        logger.info(f"  [SCHEMA] 已检查并清理旧字段 config_type: {dropped} 张配置表")


def _apply_csv_overrides_for_db(config_data: dict, config_name: str, logger, db=None) -> None:
    """数据库模式禁用 CSV 覆盖，保持与 Dev 文件模式一致。"""
    return


def _build_db_log_dir(
    config_name: str,
    timestamp: str,
    project_root: Path | None = None,
    output_subpath: Path | None = None,
) -> Path:
    """返回数据库模式日志目录路径（位于项目 outputs 根目录下）。

    路径形态：``<root>/outputs/<output_subpath>/db_run_<timestamp>``。
    ``output_subpath`` 缺省时回退到 ``Path(config_name) / config_name``，
    确保纯 DB 名（无本地 Excel）仍能落到二级结构下。
    """
    root = project_root if project_root is not None else _PROJECT_ROOT
    sub = output_subpath if output_subpath is not None else Path(config_name) / config_name
    return root / "outputs" / sub / f"db_run_{timestamp}"


def _run_with_database(ns: argparse.Namespace) -> int:
    """
    数据库模式运行
    
    - 从数据库读取配置（使用配置名称如 BC_S5）
    - 默认将输出写入数据库
    - 本地默认仅保存日志；启用 `--local` 时额外导出 Dev 格式结果
    """
    from datetime import datetime
    import time
    import shutil
    
    # 规范化配置名称：去掉路径前缀和 .xlsx 后缀，确保纯名称如 BC_S5
    # 用户可能传入 "config/BC_S5"、"config/BC_S5.xlsx" 或 "BC_S5"；
    # 也支持新入口 --config-dir <绝对路径或短格式>。
    _raw_config_dir = getattr(ns, "config_dir", None)
    _raw_config = ns.config
    cfg = None  # ConfigDir 实例，若可构造则用于派生 output_subpath
    try:
        if _raw_config_dir:
            from .config_dir import ConfigDir
            from .utils import expand_config_dir_arg
            cfg = ConfigDir.from_path(expand_config_dir_arg(_raw_config_dir))
            local_config_file = cfg.excel_path
            config_name = cfg.excel_path.stem
            _raw_config = str(cfg.excel_path)  # 供下游同名配置比对函数使用
        else:
            config_name, local_config_file = _resolve_db_config_input(_raw_config)
            if local_config_file is not None:
                from .config_dir import ConfigDir
                cfg = ConfigDir.from_excel_path(local_config_file)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ConfigError] {e}", file=sys.stderr)
        return 2

    # 输出二级子路径（DB 模式与文件模式共用此规则）
    _output_subpath = cfg.output_subpath if cfg is not None else Path(config_name) / config_name

    # P1 schema 隔离：从 ConfigDir.project 推导 schema，纯 --config 模式回退 default_schema。
    # schema 解析失败（非法字符 / 缺失）以 ValueError 终止，对应退出码 2。
    try:
        from .schema_resolver import resolve_project_schema
        from pgsql_db.settings import resolve_database_config as _resolve_db_cfg
        _db_cfg = _resolve_db_cfg()
        _project = cfg.project if cfg is not None else None
        _scenario = cfg.scenario if cfg is not None else None
        db_schema = resolve_project_schema(
            _project,
            default_schema=_db_cfg.get("default_schema"),
        )
    except ValueError as e:
        print(f"[ConfigError] {e}", file=sys.stderr)
        return 2

    start_date = ns.start_date
    end_date = ns.end_date
    
    if not start_date:
        raise ValueError("数据库模式必须提供 --start-date 参数")
    
    
    # 导入数据库模块
    try:
        from pgsql_db.module_data_writer import ModuleDataWriter
        from pgsql_db.db_initializer import DatabaseInitializer
    except ImportError as e:
        return 1
    
    # ========== 使用 DatabaseInitializer 自动检测和初始化 ==========
    initializer = DatabaseInitializer(
        host=ns.db_host,
        port=ns.db_port,
        database=ns.db_name,
        user=ns.db_user,
        password=ns.db_password,
        schema=db_schema,
    )
    
    # 执行初始化（检测数据库、创建数据库、检测配置表、导入配置）
    init_result = initializer.initialize(
        config_name=config_name,
        auto_import_config=(local_config_file is None),
        verbose=True
    )
    
    if not init_result["success"]:
        return 1
    
    # 获取数据库连接供后续使用
    db = initializer.db
    
    # 创建本地日志目录（默认仅保存日志文件）
    # 新路径形态：outputs/<project>/<scenario>/db_run_<ts>/
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = _build_db_log_dir(
        config_name=config_name,
        timestamp=ts,
        project_root=_PROJECT_ROOT,
        output_subpath=_output_subpath,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志系统
    from ...utils.logger_config import setup_logging
    logger, redirector = setup_logging(str(log_dir), log_level="INFO", redirect_print=True)
    
    total_start = time.time()
    program_start_datetime = datetime.now()
    _run_completed = False  # 用于 finally 中判断是否异常退出
    temp_dir = None          # 运行结束后统一清理
    
    logger.info("\n" + "=" * 60)
    logger.info("🕐 程序时间信息")
    logger.info("=" * 60)
    logger.info(f"📅 程序开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(
        "[DB] project={project}, scenario={scenario}, db_schema={schema}".format(
            project=_project or "(none)",
            scenario=_scenario or "(none)",
            schema=db_schema,
        )
    )
    
    try:
        # ========== 步骤0.5: 同名配置比对并按需同步 ==========
        logger.info("\n" + "=" * 60)
        logger.info("🔄 步骤0.5: 检查同名配置并按差异同步数据库")
        logger.info("=" * 60)

        from pgsql_db.config_manifest import (
            ensure_manifest_table,
            load_manifest,
            upsert_manifest_entry,
            compute_table_hash,
        )

        # 构建本地期望配置（Excel + CSV覆盖）—— 用原始 --config 输入（支持任意路径/子目录）
        expected_config = _build_expected_local_config(
            _raw_config,
            config_name=config_name,
            report_dir=log_dir / "input_quality",
            logger=logger,
        )
        if not expected_config:
            logger.info(f"  ℹ️ 未找到本地配置文件 {config_name}.xlsx（输入：{_raw_config}），直接使用数据库配置")
        else:
            # 预计算每个 cfg_* 的稳定 content hash（不依赖 DB），用于 manifest 快速比对。
            expected_hashes = {
                key: compute_table_hash(df) for key, df in expected_config.items()
            }
            from pgsql_db.config_comments import load_config_table_comment_map

            table_comment_map = load_config_table_comment_map()
            local_source = _resolve_local_config_source(_raw_config)
            config_basename = Path(config_name).stem

            ensure_manifest_table(db)

            exists, _ = initializer.check_config_data_exists(config_name)
            if not exists:
                logger.info("  ℹ️ 数据库中不存在同名配置，执行首次导入")
                local_source = _resolve_local_config_source(_raw_config)
                if local_source is None:
                    logger.error(f"[ERROR] 未找到本地配置文件: {config_name}.xlsx（输入：{_raw_config}）")
                    return 1
                first_import_diff = {
                    "new_tables": sorted(expected_config.keys()),
                    "changed_tables": [],
                    "deleted_tables": [],
                    "unchanged_tables": [],
                }
                _sync_config_by_diff(
                    db,
                    config_name,
                    expected_config,
                    first_import_diff,
                    logger,
                    expected_hashes=expected_hashes,
                    source_file=local_source,
                    table_comment_map=table_comment_map,
                )
                imported_count = len(expected_config)
                logger.info(f"  ✅ 首次导入完成: {imported_count} 个配置表")
                # 首次导入后逐表验证 DB 状态。任一 expected 表未通过验证即视为部分失败，
                # 整体退出，避免把"部分导入失败"固化成"已成功导入"。
                missing_tables: list[str] = []
                for db_key, df in expected_config.items():
                    table_name = f"cfg_{db_key}"
                    if not _verify_cfg_table_persisted(
                        db, table_name, config_basename, expect_rows=not df.empty
                    ):
                        missing_tables.append(table_name)
                if missing_tables:
                    logger.error(
                        "[ERROR] 首次导入存在部分失败：以下配置表未在 DB 验证到本 config 的数据，"
                        " 已拒绝写入 manifest 并终止运行 -> "
                        + ", ".join(missing_tables)
                    )
                    return 1
            else:
                logger.info("  ℹ️ 检测到数据库存在同名配置，开始比对本地与数据库")
                manifest = load_manifest(db, config_basename)

                if manifest:
                    # manifest 命中：走 hash 快速路径，只在 hash 不一致时读取 cfg_* 明细。
                    diff_result = _diff_local_vs_db_via_manifest(
                        expected_config,
                        expected_hashes,
                        manifest,
                        db,
                        config_basename,
                        logger,
                    )
                else:
                    # 无 manifest：回退到原有的"整表读取 + pandas 比对"。
                    logger.info("  ℹ️ 未找到 manifest，回退到整表读取比对（首次升级路径）")
                    db_snapshot = _load_config_from_database(db, config_name) or {}
                    diff_result = _diff_local_vs_db_config(expected_config, db_snapshot)

                changed_count = len(diff_result["changed_tables"])
                new_count = len(diff_result["new_tables"])
                deleted_count = len(diff_result["deleted_tables"])

                if changed_count == 0 and new_count == 0 and deleted_count == 0:
                    logger.info("  ✅ 本地配置与数据库无差异，直接读取数据库配置")
                    # 即便无差异，也对照 manifest 兜底刷新（缺 manifest 行的补齐）。
                    # 同样使用 DB 状态验证：本地非空但 DB 未发现该 config 数据时跳过，
                    # 避免遗留数据库 + 缺 manifest 场景下把空表当作"已导入"记进 manifest。
                    if not manifest:
                        try:
                            src_mtime = (
                                datetime.fromtimestamp(local_source.stat().st_mtime)
                                if local_source is not None else None
                            )
                        except OSError:
                            src_mtime = None
                        skipped_backfill: list[str] = []
                        for db_key, df in expected_config.items():
                            table_name = f"cfg_{db_key}"
                            if not _verify_cfg_table_persisted(
                                db, table_name, config_basename, expect_rows=not df.empty
                            ):
                                skipped_backfill.append(table_name)
                                continue
                            upsert_manifest_entry(
                                db,
                                config_name=config_basename,
                                table_name=table_name,
                                row_count=len(df),
                                content_hash=expected_hashes.get(db_key, ""),
                                source_file=str(local_source) if local_source else None,
                                source_mtime=src_mtime,
                            )
                        if skipped_backfill:
                            logger.warning(
                                "  [manifest] 兜底回填跳过未验证表："
                                + ", ".join(skipped_backfill)
                            )
                else:
                    logger.info(
                        f"  🔍 差异检测结果: 变更 {changed_count}，新增 {new_count}，删除 {deleted_count}"
                    )
                    _sync_config_by_diff(
                        db, config_name, expected_config, diff_result, logger,
                        expected_hashes=expected_hashes,
                        source_file=local_source,
                        table_comment_map=table_comment_map,
                    )
                    logger.info("  ✅ 差异同步完成")

            _refresh_config_table_comments(
                db,
                expected_config,
                table_comment_map,
                logger,
            )
            _drop_legacy_config_type_columns(db, expected_config, logger)

        # ========== 步骤1: 从数据库获取配置 ==========
        logger.info("\n" + "=" * 60)
        logger.info("📥 步骤1: 从数据库加载配置")
        logger.info("=" * 60)

        config_data = _load_config_from_database(db, config_name)
        if not config_data:
            logger.error(f"[ERROR] 未在数据库中找到配置: {config_name}")
            return 1

        logger.info(f"[OK] 已加载 {len(config_data)} 个配置表")
        
        # ========== 步骤2: 使用DuckDB处理并运行仿真 ==========
        logger.info("\n" + "=" * 60)
        logger.info("[DUCK] 步骤2: DuckDB处理配置并运行仿真")
        logger.info("=" * 60)
        
        # 预建所有输出表（确保中断时表结构已存在）
        pre_writer = ModuleDataWriter(db, config_name=config_name)
        if hasattr(pre_writer, "ensure_output_tables_exist"):
            pre_writer.ensure_output_tables_exist()
        else:
            logger.warning(
                "[WARN] 当前 ModuleDataWriter 不包含 ensure_output_tables_exist，"
                "已跳过预建输出表；请同步 `pgsql_db/module_data_writer.py` 与当前运行代码版本。"
            )

        # P1 索引注册器：cfg_* (config_name) + 输出表轻量 (run_id, sim_date)。
        # 配置阶段已 ensure_manifest_table，再在此一次性把所有 cfg_* 索引补齐。
        try:
            from pgsql_db.index_registry import (
                ensure_config_indexes,
                ensure_output_lightweight_indexes,
            )
            cfg_idx = ensure_config_indexes(db)
            out_idx = ensure_output_lightweight_indexes(db)
            logger.info(
                f"[INDEX] 注册 cfg_* 索引 {cfg_idx} 个，输出表轻量索引 {out_idx} 个"
            )
        except Exception as _ix_err:
            logger.warning(f"[INDEX] 注册索引时发生异常（不阻塞主流程）：{_ix_err}")

        # 并发保护（最佳努力）：尝试避免同一 config_name 被多个进程同时运行
        from pgsql_db.checkpoint import try_acquire_run_lock, release_run_lock
        _lock_key = config_name
        if not try_acquire_run_lock(db, _lock_key):
            logger.error(
                f"[ERROR] 已有另一个进程正在运行 config_name='{config_name}'。"
                " 如需强制接管，请先终止已有进程或等待其完成后重试。"
            )
            return 1
        logger.info(f"[LOCK] 获取并发锁成功（config_name={config_name}）")
        
        # 创建临时输出目录
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_output = temp_dir / "output"
        temp_output.mkdir(exist_ok=True)
        
        # 使用与文件模式一致的标准仿真引擎。
        from ..main_integration import run_integrated_simulation_from_dict
        logger.info("[RUN] 使用标准仿真引擎运行（与文件模式一致）...")
        # 读取resume标志
        resume_flag = getattr(ns, 'resume', False)
        
        # 断点续跑时复用未完成的 run_id，避免每次生成新 run_id 导致数据碎片化
        # 1. --force-restart 时：始终生成新 run_id（全新开始）
        # 2. 其他情况：自动检测；若存在未完成(running/failed)运行则复用 run_id 并自动进入断点续跑，否则全新开始
        effective_run_id = f"db_{config_name}_{ts}"  # 默认：新 run_id
        if not getattr(ns, 'force_restart', False):
            try:
                from pgsql_db.checkpoint import load_checkpoint as _lc
                _existing_cp = _lc(db, config_name, start_date=start_date, end_date=end_date)
                if _existing_cp:
                    effective_run_id = _existing_cp['run_id']
                    resume_flag = True  # 自动启用断点续跑模式
                    logger.info(f"[RESUME] 检测到未完成运行，自动进入断点续跑 run_id={effective_run_id}, last_batch_end={_existing_cp['last_batch_end']}")
                else:
                    logger.info(f"[NEW] 未检测到未完成运行，全新开始 run_id={effective_run_id}")
            except Exception as _ce:
                logger.warning(f"[WARN] 断点续跑检测异常（将全新开始）: {_ce}")

        # 落盘 effective_run_id：续跑场景下这里已经是复用的既有 run_id，与 DB 对齐
        from .output_dir import _write_run_id_file
        _write_run_id_file(
            log_dir,
            effective_run_id,
            "db_run_id.txt",
            logger,
            db_schema=db_schema,
        )

        try:
            result = run_integrated_simulation_from_dict(
                config_data=config_data,
                config_name=config_name,
                start_date=start_date,
                end_date=end_date,
                output_base_dir=str(temp_output),
                skip_validation=True,
                resume=resume_flag,
                db=db,
                run_key=config_name,
                batch_size=1,
                run_id=effective_run_id
            )
        except Exception as sim_err:
            # 仿真异常时更新 checkpoint 状态为 failed
            logger.error(f"[ERROR] 仿真过程中发生异常: {sim_err}")
            try:
                from pgsql_db.checkpoint import update_checkpoint_status
                update_checkpoint_status(db, effective_run_id, 'failed', error_message=str(sim_err)[:500])
                logger.info(f"❌ 运行状态已更新为 failed (run_id={effective_run_id})")
            except Exception:
                pass  # checkpoint 状态更新失败不影响异常传播
            raise
        
        if not result or not result.get('simulation_completed'):
            logger.error("[ERROR] 仿真运行失败")
            return 1
        
        # ========== 步骤3: 后处理（Summary + Orchestrator + 状态更新）==========
        logger.info("\n" + "=" * 60)
        logger.info("[OUT] 步骤3: 后处理（Summary汇总、Orchestrator写入、状态更新）")
        logger.info("=" * 60)
        
        output_dir = result.get('output_directory')
        # 关键修复：使用仿真过程中实际使用的 run_id，而非重新生成
        # 仿真过程中 _flush_batch_to_db 使用 effective_run_id 写入模块数据，
        # 后续 Summary 和 Orchestrator 必须使用相同的 run_id 才能正确关联和查询数据
        run_id = result.get('run_id', effective_run_id)
        writer = ModuleDataWriter(db, config_name=config_name)
        
        db_write_start = time.time()
        
        all_results = result.get('results')
        
        if all_results:
            logger.info(f"[OUT] 模块数据已在仿真过程中批量写入（run_id={run_id}），跳过全量重写")
            
            # ========== 本地文件输出（默认禁用） ==========
            if getattr(ns, 'local', False):
                logger.info("\n" + "=" * 60)
                logger.info("[DIR] 输出本地文件（Dev格式）")
                logger.info("=" * 60)
                # 与 log_dir 共用同一 <project>/<scenario>/db_run_<ts> 目录，
                # 让 DB 模式所有产物（日志 + Dev 格式导出）聚合在一处。
                local_output_dir = log_dir
                local_output_dir.mkdir(parents=True, exist_ok=True)
                
                orch_output_dir = str(Path(output_dir) / "orchestrator") if output_dir else None
                
                _write_results_to_local_dev_format(
                    all_results=all_results,
                    output_dir=local_output_dir,
                    logger=logger,
                    config_dict=result.get('config_dict'),
                    start_date=start_date,
                    end_date=end_date,
                    orchestrator_output_dir=orch_output_dir
                )
                logger.info(f"[OK] 本地输出目录: {local_output_dir}")
                
        elif output_dir and Path(output_dir).exists():
            logger.info("[OUT] 从文件读取并写入模块输出到数据库...")
            writer.write_all_modules(str(output_dir), run_id=run_id, if_exists='replace')
        else:
            logger.warning("[WARN] 无可用的模块输出数据写入数据库")
        
        # Orchestrator 状态数据写入
        # 注意：模块每日输出已在 _flush_batch_to_db 中原子写入，
        # 但 Orchestrator CSV 文件是仿真过程中实时写到临时目录的，需要单独写入DB
        if output_dir and not all_results:
            orch_dir = Path(output_dir) / "orchestrator"
            if orch_dir.exists():
                logger.info("[OUT] 写入 Orchestrator 状态数据到数据库...")
                writer.write_orchestrator_data(str(orch_dir), run_id=run_id, if_exists='replace')
            else:
                logger.warning("[WARN] Orchestrator 目录不存在，跳过写入")
        elif output_dir:
            logger.info("[OUT] Orchestrator 状态数据已在仿真过程中按天写入数据库，跳过最终整段重写")
        
        # Summary 汇总报告生成
        # 必须使用与模块数据相同的 run_id 进行过滤，否则查不到数据
        logger.info(f"[DATA] 从数据库生成 Summary 汇总报告（run_id={run_id}）...")
        # P1 索引注册器：Summary 前建立宽 BTREE 索引，配合 group by / order by 命中。
        try:
            from pgsql_db.index_registry import ensure_summary_wide_indexes
            wide_idx = ensure_summary_wide_indexes(db)
            logger.info(f"[INDEX] Summary 前注册宽索引 {wide_idx} 个")
        except Exception as _ix_err:
            logger.warning(f"[INDEX] Summary 宽索引注册失败（不阻塞主流程）：{_ix_err}")
        summary_success = False
        summary_results = {}
        try:
            summary_results = writer.generate_summary_reports_from_db(
                run_id=run_id,
                start_date=start_date,
                end_date=end_date,
                if_exists='replace'
            )
            failed_summary_tables = [
                name for name, rows in summary_results.items()
                if isinstance(rows, int) and rows < 0
            ]
            if failed_summary_tables:
                raise RuntimeError(
                    "Summary 表生成失败: " + ", ".join(failed_summary_tables)
                )
            summary_success = True
        except Exception as summary_err:
            logger.error(f"[ERROR] Summary 汇总报告生成失败: {summary_err}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 状态更新：仅在所有 Summary 表生成完成后才将 checkpoint 标记为 completed
        # 之前的 bug：simulation_db.py 在仿真循环结束后就标记 completed，
        # 但此时 Summary 表尚未生成，导致状态不一致
        if db is not None:
            from pgsql_db.checkpoint import update_checkpoint_status
            def _update_checkpoint_status_with_retry(status: str, error_message: str | None = None) -> None:
                last_err = None
                for attempt in range(3):
                    try:
                        update_checkpoint_status(db, run_id, status, error_message=error_message)
                        return
                    except Exception as cp_err:
                        last_err = cp_err
                        try:
                            db.close()
                        except Exception:
                            pass
                        if attempt < 2:
                            wait_seconds = 5 * (2 ** attempt)
                            logger.warning(
                                f"[WARN] 更新 checkpoint 状态失败，{wait_seconds}s 后重试 "
                                f"({attempt + 1}/3, run_id={run_id}): {cp_err}"
                            )
                            time.sleep(wait_seconds)
                raise last_err

            if summary_success:
                _update_checkpoint_status_with_retry('completed')
                logger.info(f"[OK] 运行状态已更新为 completed（所有 Summary 已生成，run_id={run_id}）")
            else:
                _update_checkpoint_status_with_retry(
                    'failed',
                    error_message='Summary 汇总报告生成失败'
                )
                logger.warning(f"[WARN] 运行状态已更新为 failed（Summary 生成失败，run_id={run_id}）")
                _run_completed = True
                return 1
        
        db_write_time = time.time() - db_write_start
        logger.info(f"[TIME]  数据库写入耗时: {db_write_time:.2f}秒")
        
        total_time = time.time() - total_start
        program_end_datetime = datetime.now()
        
        # 格式化运行时间
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours >= 1:
            runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
        elif minutes >= 1:
            runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
        else:
            runtime_str = f"{seconds:.2f}秒"
        
        logger.info("\n" + "=" * 60)
        logger.info("[OK] 数据库模式运行完成 (DuckDB)")
        logger.info("=" * 60)
        logger.info("🕐 程序时间统计:")
        logger.info(f"   📅 开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   📅 结束时间: {program_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   [TIME]  总运行时间: {runtime_str}")
        logger.info(f"   [TIME]  其中DB写入: {db_write_time:.2f}秒")
        logger.info(f"   [DIR] 日志目录: {log_dir}")
        logger.info("[DATA] 数据存储:")
        logger.info(f"   [SAVE] 所有输出数据已写入 PostgreSQL 数据库")
        logger.info(f"   🔗 连接: postgresql://localhost:5432/test_db")
        logger.info(f"   [INFO] 表前缀: {run_id}")
        logger.info("=" * 60)

        _run_completed = True
        return 0
        
    except KeyboardInterrupt:
        # Ctrl+C / SIGINT：不打印堆栈，仅记录中断
        # finally 块会将 checkpoint 标记为 interrupted
        logger.warning("\n[INTERRUPTED] 用户中断 (Ctrl+C)，正在保存状态...")
        return 130  # Unix 惯例: 128 + SIGINT(2)

    except Exception as e:
        logger.error(f"[ERROR] 执行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # 非正常退出时将 checkpoint 状态标记为 interrupted
        # 这样断点续跑时能区分“正在运行”与“被中断”，也方便运维排查
        if not _run_completed:
            try:
                from pgsql_db.checkpoint import update_checkpoint_status
                _target_run_id = locals().get('effective_run_id')
                if _target_run_id:
                    _status_rows = db.execute_query(
                        f"SELECT COALESCE(status, 'running') FROM {db.qualified_name('sim_checkpoint')} WHERE run_id = %s",
                        (_target_run_id,),
                    )
                    if _status_rows and _status_rows[0][0] == 'running':
                        update_checkpoint_status(
                            db,
                            _target_run_id,
                            'interrupted',
                            error_message='进程异常退出（kill / crash / KeyboardInterrupt）',
                        )
                        logger.info(f"checkpoint 状态已更新为 interrupted（run_id={_target_run_id}）")
            except Exception:
                pass  # 连接已关闭或 DB 不可用时静默忽略
        # 正常或异常退出时显式释放并发锁
        try:
            from pgsql_db.checkpoint import release_run_lock
            release_run_lock(db, config_name)
            logger.info(f"[LOCK] 并发锁已释放（config_name={config_name}）")
        except Exception:
            pass
        # 关闭数据库连接
        db.close()
        logger.info("🔌 数据库连接已关闭")
        if redirector:
            redirector.stop_redirect()
        # 清理临时输出目录（M4 文件已全部存入 DB，temp 目录无需保留）
        if temp_dir is not None and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"[CLEAN] 临时目录已清理: {temp_dir}")
            except Exception:
                pass  # 清理失败不影响主流程


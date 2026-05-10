"""
db_runner.py

数据库模式运行器
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from pgsql_db import table_mapping
from .db_config import _load_config_from_database
from .local_writer import _write_results_to_local_dev_format


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


def _find_local_config_file(config_name: str) -> Path | None:
    """按配置名查找本地 Excel 配置文件。"""
    config_basename = Path(config_name).stem
    project_root = Path(__file__).parent.parent.parent.parent
    search_dirs = [project_root / "config", project_root / "test_files", project_root]

    for d in search_dirs:
        candidate = d / f"{config_basename}.xlsx"
        if candidate.exists():
            return candidate
    return None


def _build_expected_local_config(config_name: str) -> dict:
    """构建本地期望配置（Excel + CSV 覆盖空 sheet，与文件模式语义一致）。"""
    config_file = _find_local_config_file(config_name)
    if config_file is None:
        return {}

    try:
        xl = pd.ExcelFile(str(config_file))
    except ValueError:
        # 与 ExcelImporter 保持一致：兼容 openpyxl 对部分字体 family 的严格校验
        from pgsql_db.excel_importer import ExcelImporter
        ExcelImporter._patch_openpyxl_font_family()
        xl = pd.ExcelFile(str(config_file))

    sheet_data: dict[str, pd.DataFrame] = {}
    for sheet_name in xl.sheet_names:
        sheet_data[sheet_name] = xl.parse(sheet_name)

    # CSV 覆盖：仅当 Excel 中对应 sheet 为空时使用同名 CSV，保持与 load_configuration 一致
    from src.core.main_integration.config_loader import load_csv_overrides
    csv_overrides = load_csv_overrides(str(config_file))
    for sheet_name, csv_df in csv_overrides.items():
        existing = sheet_data.get(sheet_name)
        if existing is None or existing.empty:
            sheet_data[sheet_name] = csv_df

    expected = {}
    for sheet_name, df in sheet_data.items():
        table_name = table_mapping.get_config_table_name(sheet_name)
        db_key = table_name[4:] if table_name.startswith("cfg_") else table_name
        expected[db_key] = df

    return expected


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


def _sync_config_by_diff(db, config_name: str, expected_config: dict, diff_result: dict, logger) -> None:
    """按差异最小化同步：仅更新新增/变更表，并删除本地已不存在的表数据。"""
    config_basename = Path(config_name).stem

    to_upsert = sorted(set(diff_result["new_tables"] + diff_result["changed_tables"]))
    to_delete = diff_result["deleted_tables"]

    for db_key in to_upsert:
        table_name = f"cfg_{db_key}"
        df = expected_config.get(db_key, pd.DataFrame())
        deleted = db.delete_config_data(table_name, config_basename)
        db.create_table_from_df(
            df,
            table_name,
            if_exists="append",
            config_name=config_basename,
            config_type=config_basename,
        )
        logger.info(f"  🔄 同步配置表 {table_name}: 删除 {deleted} 行，写入 {len(df)} 行")

    for db_key in to_delete:
        table_name = f"cfg_{db_key}"
        deleted = db.delete_config_data(table_name, config_basename)
        logger.info(f"  🗑️ 删除本地已不存在配置表数据 {table_name}: 删除 {deleted} 行")


def _apply_csv_overrides_for_db(config_data: dict, config_name: str, logger, db=None) -> None:
    """数据库模式禁用 CSV 覆盖，保持与 Dev 文件模式一致。"""
    return


def _build_db_log_dir(
    config_name: str,
    timestamp: str,
    project_root: Path | None = None,
) -> Path:
    """返回数据库模式日志目录路径（位于项目 outputs 根目录下）。"""
    root = project_root if project_root is not None else Path.cwd()
    return root / "outputs" / f"db_{config_name}_{timestamp}"


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
    # 用户可能传入 "config/BC_S5"、"config/BC_S5.xlsx" 或 "BC_S5"
    _raw_config = ns.config
    config_name = Path(_raw_config).stem  # 去掉目录和后缀
    start_date = ns.start_date
    end_date = ns.end_date
    
    if not start_date:
        raise ValueError("数据库模式必须提供 --start-date 参数")
    
    
    # 导入数据库模块
    try:
        from pgsql_db.db_connection import DatabaseConnection
        from pgsql_db.excel_importer import ExcelImporter
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
        password=ns.db_password
    )
    
    # 执行初始化（检测数据库、创建数据库、检测配置表、导入配置）
    init_result = initializer.initialize(
        config_name=config_name,
        auto_import_config=True,
        verbose=True
    )
    
    if not init_result["success"]:
        return 1
    
    # 获取数据库连接供后续使用
    db = initializer.db
    
    # 创建本地日志目录（默认仅保存日志文件）
    project_root = Path.cwd()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = _build_db_log_dir(config_name=config_name, timestamp=ts)
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
    
    try:
        # ========== 步骤0.5: 同名配置比对并按需同步 ==========
        logger.info("\n" + "=" * 60)
        logger.info("🔄 步骤0.5: 检查同名配置并按差异同步数据库")
        logger.info("=" * 60)

        # 构建本地期望配置（Excel + CSV覆盖）
        expected_config = _build_expected_local_config(config_name)
        if not expected_config:
            logger.info(f"  ℹ️ 未找到本地配置文件 {Path(config_name).stem}.xlsx，直接使用数据库配置")
        else:
            exists, _ = initializer.check_config_data_exists(config_name)
            if not exists:
                logger.info("  ℹ️ 数据库中不存在同名配置，执行首次导入")
                config_file = initializer.find_config_file(config_name)
                if not config_file:
                    logger.error(f"[ERROR] 未找到本地配置文件: {Path(config_name).stem}.xlsx")
                    return 1
                success, import_results = initializer.import_config_from_excel(config_name, config_file)
                if not success:
                    logger.error("[ERROR] 首次导入配置失败")
                    return 1
                imported_count = len([r for r in import_results.values() if r >= 0])
                logger.info(f"  ✅ 首次导入完成: {imported_count} 个配置表")
            else:
                logger.info("  ℹ️ 检测到数据库存在同名配置，开始比对本地与数据库")
                db_snapshot = _load_config_from_database(db, config_name) or {}
                diff_result = _diff_local_vs_db_config(expected_config, db_snapshot)

                changed_count = len(diff_result["changed_tables"])
                new_count = len(diff_result["new_tables"])
                deleted_count = len(diff_result["deleted_tables"])

                if changed_count == 0 and new_count == 0 and deleted_count == 0:
                    logger.info("  ✅ 本地配置与数据库无差异，直接读取数据库配置")
                else:
                    logger.info(
                        f"  🔍 差异检测结果: 变更 {changed_count}，新增 {new_count}，删除 {deleted_count}"
                    )
                    _sync_config_by_diff(db, config_name, expected_config, diff_result, logger)
                    logger.info("  ✅ 差异同步完成")

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
                from pgsql_db.checkpoint import update_checkpoint_status, load_checkpoint
                _cp = load_checkpoint(db, config_name)
                if _cp:
                    update_checkpoint_status(db, _cp['run_id'], 'failed', error_message=str(sim_err)[:500])
                    logger.info(f"❌ 运行状态已更新为 failed (run_id={_cp['run_id']})")
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
                local_output_dir = project_root / "outputs" / config_name / f"run_{ts}"
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
                from pgsql_db.checkpoint import load_checkpoint, update_checkpoint_status
                _cp = load_checkpoint(db, config_name)
                if _cp and _cp.get('status') == 'running':
                    update_checkpoint_status(db, _cp['run_id'], 'interrupted',
                                             error_message='进程异常退出（kill / crash / KeyboardInterrupt）')
                    logger.info(f"checkpoint 状态已更新为 interrupted（run_id={_cp['run_id']}）")
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


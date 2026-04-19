#!/usr/bin/env python3
"""
run_main.py

供应链计划系统生产集成运行器主入口。

职责：
- 解析命令行参数并区分本地文件模式与数据库模式。
- 组织输出目录、日志系统与续跑能力检查。
- 将执行请求转交给集成仿真主流程或数据库运行流程。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

if sys.platform == 'win32':
    import io
    # 检查是否已经被包装（避免重复包装）
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass  # 忽略包装失败
    if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass  # 忽略包装失败

# 将项目根目录加入导入路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..main_integration import (
    run_integrated_simulation,
    load_configuration,
    check_resume_capability,
)
from ...utils.logger_config import setup_logging
from .output_dir import (
    _ensure_output_dir,
    get_or_init_simulation_start,
    _list_existing_runs,
)
from .db_runner import _run_with_database


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """解析命令行参数并返回运行配置。"""
    parser = argparse.ArgumentParser(
        prog="production_integrator",
        add_help=True,
        description=(
            "Run the integrated planning flow using a given configuration file. "
            "Outputs are written under a folder named after the configuration file. "
            "Supports automatic resume from interruption points with directory selection."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration .xlsx file",
    )
    parser.add_argument(
        "--start-date",
        required=False,
        help="Simulation start date in YYYY-MM-DD (required for first run)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Simulation end date in YYYY-MM-DD",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart from beginning, ignore resume capability",
    )
    parser.add_argument(
        "--check-resume",
        action="store_true",
        help="Check resume status only, do not execute simulation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Enable automatic resume from interruption point",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Specific run directory to resume from (e.g., run_20241203_120000)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all available run directories and their status",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts (auto-select most recent run)",
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="使用数据库模式：从数据库读取配置，输出写入数据库，本地只保存运行日志txt",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default=None,
        help="数据库主机地址 (默认读取自 config/defaults.yaml)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="数据库端口 (默认读取自 config/defaults.yaml)",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default=None,
        help="数据库名称 (默认读取自 config/defaults.yaml)",
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default=None,
        help="数据库用户名 (默认读取自 config/defaults.yaml)",
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default=None,
        help="数据库密码 (默认读取自 config/defaults.yaml)",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="运行目录后缀，用于区分不同运行 (例如: --run-suffix test 生成 run_YYYYMMDD_HHMMSS_test)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="数据库模式下启用本地文件输出（同时写入数据库和本地Excel文件）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """程序主入口。

    负责组织本地文件模式与数据库模式的启动流程，并在本地模式下接管续跑、日志与输出目录管理。
    """
    ns = _parse_args(argv or sys.argv[1:])

    # ==================== 数据库模式 ====================
    if ns.use_db:
        return _run_with_database(ns)
    
    # ==================== 本地文件模式 ====================
    cfg_path = Path(ns.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    if cfg_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("Configuration file must be an Excel file (.xlsx/.xlsm/.xls)")

    # 尽早加载配置以便在出现结构/格式问题时快速失败
    #（该调用会返回可供运行函数使用的对象，或用于校验配置文件。）
    _ = load_configuration(str(cfg_path))  # noqa: F841

    # 为 --list-runs 提前获取根目录与日期参数
    cfg_stem = cfg_path.stem
    project_root = Path.cwd()
    root_dir = project_root / "outputs" / cfg_stem
    root_dir.mkdir(parents=True, exist_ok=True)
    
    start_arg = ns["start_date"] if isinstance(ns, dict) else ns.start_date
    simulation_start = get_or_init_simulation_start(root_dir, start_arg)
    end_date = str(ns["end_date"]) if isinstance(ns, dict) else ns.end_date

    # 处理 --list-runs 命令
    if ns.list_runs:
        
        run_infos = _list_existing_runs(root_dir, simulation_start, end_date)
        
        if not run_infos:
            return 0
        
        for idx, info in enumerate(run_infos, 1):
            resume_info = info['resume_info']
            
            if resume_info.get('already_completed', False):
                pass
            elif resume_info['can_resume']:
                pass
            else:
                pass
        
        return 0

    # 确定输出目录与续跑模式
    enable_resume = (ns.resume or ns.resume_from) and not ns.force_restart
    output_base_dir = _ensure_output_dir(
        cfg_path, 
        resume_mode=enable_resume,
        resume_from=ns.resume_from,
        start_date=simulation_start,
        end_date=end_date,
        interactive=not ns.non_interactive
    )

    # [NEW] 设置日志系统 - 同时输出到terminal和文件
    logger, redirector = setup_logging(str(output_base_dir), log_level="INFO", redirect_print=True)
    
    import time
    program_start_time = time.time()
    program_start_datetime = datetime.now()
    
    logger.info("\n" + "=" * 60)
    logger.info("🕐 程序时间信息")
    logger.info("=" * 60)
    logger.info(f"📅 程序开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[RUN] 供应链仿真系统启动")
    logger.info(f"📂 配置文件: {cfg_path}")
    logger.info(f"[DIR] 输出目录: {output_base_dir}")
    logger.info(f"📅 仿真日期范围: {simulation_start} 到 {end_date}")
    
    try:
        # 处理续跑状态检查
        if ns.check_resume:
            logger.info("[DEBUG] 检查续跑状态...")
            resume_info = check_resume_capability(str(output_base_dir), simulation_start, end_date)
            
            logger.info(f"\n[DATA] 续跑状态报告:")
            logger.info(f"  配置文件: {cfg_path}")
            logger.info(f"  输出目录: {output_base_dir}")
            logger.info(f"  日期范围: {simulation_start} 到 {end_date}")
            
            if resume_info.get('already_completed', False):
                logger.info(f"  [OK] 仿真已完成!")
                logger.info(f"     最后处理日期: {resume_info['last_complete_date']}")
                logger.info(f"     总处理天数: {resume_info['days_completed']}")
            elif resume_info['can_resume']:
                logger.info(f"  🔄 可以续跑!")
                logger.info(f"     已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
                logger.info(f"     剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            else:
                logger.info(f"  [LOG] 无续跑能力，将从头开始")
                logger.info(f"     需处理天数: {resume_info['days_remaining']}")
            
            return 0

        # 将执行交由支持续跑能力的一体化仿真流程
        result = run_integrated_simulation(
            config_path=str(cfg_path),
            start_date=simulation_start,
            end_date=end_date,
            output_base_dir=str(output_base_dir),
            force_restart=ns.force_restart,
        )
        
        program_end_time = time.time()
        program_end_datetime = datetime.now()
        total_runtime = program_end_time - program_start_time
        
        # 格式化运行时间
        hours, remainder = divmod(total_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours >= 1:
            runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
        elif minutes >= 1:
            runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
        else:
            runtime_str = f"{seconds:.2f}秒"
        
        logger.info("\n" + "=" * 60)
        if result and result.get('simulation_completed'):
            logger.info("[OK] 仿真成功完成")
        else:
            failure_stage = (
                result.get('failure_stage', 'unknown')
                if isinstance(result, dict) else 'unknown'
            )
            logger.warning(f"[WARN] 仿真未完成，已在阶段 {failure_stage} 停止")
            if isinstance(result, dict) and result.get('validation_report'):
                logger.warning(f"[WARN] 验证报告: {result['validation_report']}")
        logger.info("=" * 60)
        logger.info("🕐 程序时间统计:")
        logger.info(f"   📅 开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   📅 结束时间: {program_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   [TIME]  总运行时间: {runtime_str}")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] 仿真执行出错: {str(e)}")
        raise
    finally:
        # 恢复原始输出
        if redirector:
            redirector.stop_redirect()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise e
    except Exception as exc:
        # 不使用 print；通过非零退出码与异常传播来表示失败
        #（调用方如有需要可捕获 stderr/traceback。）
        raise SystemExit(1) from exc

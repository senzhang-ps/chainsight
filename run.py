#!/usr/bin/env python3
"""
Production Integration Runner for Supply Chain Planning System

Purpose
- Read a user-provided configuration file path (.xlsx)
- Create an output directory in the SAME directory as the configuration file
  whose top-level folder name matches the configuration file's stem
- Dispatch a full integrated run and write results to that output directory

Notes
- No test scaffolding, validation stubs, or print statements
- Quiet by default; relies on exit codes and exceptions for failure signaling

CLI
  production_integrator.py \
    --config /path/to/your_config.xlsx \
    --start-date 2024-01-01 \
    --end-date 2024-01-31

The first invocation requires ``--start-date``. Subsequent runs reuse the
start date persisted in ``<config_dir>/<config_stem>/simulation_start.txt``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# External system imports (assumed available in project environment)
from main_integration import run_integrated_simulation, load_configuration, check_resume_capability  # type: ignore
from logger_config import setup_logging  # type: ignore


def _ensure_output_dir(config_path: Path, resume_mode: bool = False) -> Path:
    """Create the output directory rooted by the config filename stem.

    Structure:
      <config_dir>/<config_stem>/
        └─ run_YYYYMMDD_HHMMSS/  (actual write target to avoid overwrites)

    Args:
        config_path: Path to the configuration file
        resume_mode: If True, look for existing run directories for resume capability

    Returns the leaf path to be used as `output_base_dir`.
    """
    cfg_dir = config_path.parent
    cfg_stem = config_path.stem

    root_dir = cfg_dir / cfg_stem
    # Always ensure the top-level directory exists so its name matches the config
    root_dir.mkdir(parents=True, exist_ok=True)

    if resume_mode:
        # In resume mode, look for the most recent run directory
        existing_runs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if existing_runs:
            # Sort by creation time and return the most recent one
            latest_run = max(existing_runs, key=lambda d: d.stat().st_mtime)
            return latest_run
        # If no existing runs found in resume mode, fall through to create new one

    # Create a unique run folder under the top-level directory to avoid collisions
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir

def get_or_init_simulation_start(output_root: Path, provided_start: Optional[str]) -> str:
    """Return the persistent simulation start date for this configuration.

    ``output_root`` is the directory that contains all run folders. The start
    date is stored in a ``simulation_start.txt`` file within this directory. If
    the file exists, its contents are returned. Otherwise ``provided_start`` is
    written to the file and returned. ``provided_start`` must be supplied on the
    first run when the file does not yet exist.
    """
    start_file = output_root / "simulation_start.txt"
    if start_file.exists():
        return start_file.read_text(encoding="utf-8").strip()
    if not provided_start:
        raise ValueError("Simulation start date required for first run")
    start_file.write_text(provided_start, encoding="utf-8")
    return provided_start



def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="production_integrator",
        add_help=True,
        description=(
            "Run the integrated planning flow using a given configuration file. "
            "Outputs are written under a folder named after the configuration file. "
            "Supports automatic resume from interruption points."
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
        help="Enable automatic resume from interruption point (default: True)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv or sys.argv[1:])

    cfg_path = Path(ns.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    if cfg_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("Configuration file must be an Excel file (.xlsx/.xlsm/.xls)")

    # Ensure we can load configuration early to fail fast on schema issues
    # (This returns an object usable by your run function or validates the file.)
    _ = load_configuration(str(cfg_path))  # noqa: F841

    # Determine output directory and resume mode
    enable_resume = ns.resume and not ns.force_restart
    output_base_dir = _ensure_output_dir(cfg_path, resume_mode=enable_resume)
    root_dir = output_base_dir.parent
    start_arg = ns["start_date"] if isinstance(ns, dict) else ns.start_date
    simulation_start = get_or_init_simulation_start(root_dir, start_arg)
    end_date = str(ns["end_date"]) if isinstance(ns, dict) else ns.end_date

    # 🆕 设置日志系统 - 同时输出到terminal和文件
    logger, redirector = setup_logging(str(output_base_dir), log_level="INFO", redirect_print=True)
    logger.info(f"🚀 供应链仿真系统启动")
    logger.info(f"📂 配置文件: {cfg_path}")
    logger.info(f"📁 输出目录: {output_base_dir}")
    logger.info(f"📅 仿真日期范围: {simulation_start} 到 {end_date}")
    
    try:
        # Handle resume status check
        if ns.check_resume:
            logger.info("🔍 检查续跑状态...")
            resume_info = check_resume_capability(str(output_base_dir), simulation_start, end_date)
            
            logger.info(f"\n📊 续跑状态报告:")
            logger.info(f"  配置文件: {cfg_path}")
            logger.info(f"  输出目录: {output_base_dir}")
            logger.info(f"  日期范围: {simulation_start} 到 {end_date}")
            
            if resume_info.get('already_completed', False):
                logger.info(f"  ✅ 仿真已完成!")
                logger.info(f"     最后处理日期: {resume_info['last_complete_date']}")
                logger.info(f"     总处理天数: {resume_info['days_completed']}")
            elif resume_info['can_resume']:
                logger.info(f"  🔄 可以续跑!")
                logger.info(f"     已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
                logger.info(f"     剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            else:
                logger.info(f"  📝 无续跑能力，将从头开始")
                logger.info(f"     需处理天数: {resume_info['days_remaining']}")
            
            return 0

        # Delegate to the integrated simulation with resume capability
        _ = run_integrated_simulation(
            config_path=str(cfg_path),
            start_date=simulation_start,
            end_date=end_date,
            output_base_dir=str(output_base_dir),
            force_restart=ns.force_restart,
        )
        
        logger.info("✅ 仿真成功完成")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 仿真执行出错: {str(e)}")
        raise
    finally:
        # 恢复原始输出
        if redirector:
            redirector.stop_redirect()
            print(f"\n📝 完整日志已保存到: {output_base_dir}")  # 这条会显示在terminal


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise e
    except Exception as exc:
        # No prints; signal failure via non-zero exit and exception propagation
        # (Callers can capture stderr/traceback if needed.)
        raise SystemExit(1) from exc

# run examples
# First run requires --start-date; subsequent runs may omit it
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --start-date 2024-01-01 \
#   --end-date 2024-01-31

# Resume from interruption (automatic detection)
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --end-date 2024-01-31 \
#   --resume

# Check resume status without running
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --end-date 2024-01-31 \
#   --check-resume

# Force restart from beginning
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --start-date 2024-01-01 \
#   --end-date 2024-01-31 \
#   --force-restart
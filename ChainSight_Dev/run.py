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


def _list_existing_runs(root_dir: Path, start_date: str, end_date: str) -> list[dict]:
    """List all existing run directories with their resume status.
    
    Args:
        root_dir: The root directory containing run_* folders
        start_date: Start date for validation
        end_date: End date for validation
        
    Returns:
        List of dicts with run directory info and resume capability
    """
    from main_integration import check_resume_capability  # type: ignore
    
    existing_runs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not existing_runs:
        return []
    
    # Sort by name (which includes timestamp)
    existing_runs.sort(reverse=True)
    
    run_infos = []
    for run_dir in existing_runs:
        try:
            resume_info = check_resume_capability(str(run_dir), start_date, end_date)
            run_infos.append({
                'path': run_dir,
                'name': run_dir.name,
                'resume_info': resume_info
            })
        except Exception:
            # Skip directories that can't be analyzed
            continue
    
    return run_infos


def _prompt_user_run_selection(run_infos: list[dict]) -> Path:
    """Interactively prompt user to select a run directory.
    
    Args:
        run_infos: List of run directory information
        
    Returns:
        Selected run directory Path
    """
    print("\n" + "="*80)
    print("📂 发现多个运行目录，请选择要续跑的目录：")
    print("="*80)
    
    for idx, info in enumerate(run_infos, 1):
        resume_info = info['resume_info']
        print(f"\n[{idx}] {info['name']}")
        
        if resume_info.get('already_completed', False):
            print(f"    ✅ 状态: 已完成")
            print(f"    📅 最后日期: {resume_info['last_complete_date']}")
            print(f"    📊 完成天数: {resume_info['days_completed']}")
        elif resume_info['can_resume']:
            print(f"    🔄 状态: 可续跑")
            print(f"    📅 已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
            print(f"    📅 剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
        else:
            print(f"    📝 状态: 无可续跑数据")
            print(f"    📊 需处理: {resume_info['days_remaining']} 天")
    
    print("\n" + "="*80)
    print("请输入选项：")
    print("  - 输入数字 [1-{}] 选择对应目录".format(len(run_infos)))
    print("  - 输入 'n' 或 'new' 创建新的运行目录")
    print("  - 输入 'q' 或 'quit' 退出")
    
    while True:
        choice = input("\n👉 请选择: ").strip().lower()
        
        if choice in ['q', 'quit']:
            print("❌ 用户取消操作")
            sys.exit(0)
        
        if choice in ['n', 'new']:
            return None  # Signal to create new directory
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(run_infos):
                selected = run_infos[idx - 1]
                print(f"\n✅ 已选择: {selected['name']}")
                return selected['path']
            else:
                print(f"❌ 无效选择，请输入 1-{len(run_infos)} 之间的数字")
        except ValueError:
            print("❌ 无效输入，请输入数字、'n' 或 'q'")


def _ensure_output_dir(config_path: Path, resume_mode: bool = False, 
                      resume_from: Optional[str] = None, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interactive: bool = True) -> Path:
    """Create the output directory rooted by the config filename stem.

    Structure:
      <config_dir>/<config_stem>/
        └─ run_YYYYMMDD_HHMMSS/  (actual write target to avoid overwrites)

    Args:
        config_path: Path to the configuration file
        resume_mode: If True, allow resuming from existing run directories
        resume_from: Specific run directory name to resume from (e.g., "run_20241203_120000")
        start_date: Simulation start date (required for resume validation)
        end_date: Simulation end date (required for resume validation)
        interactive: If True, prompt user to select run directory when multiple exist

    Returns the leaf path to be used as `output_base_dir`.
    """
    cfg_dir = config_path.parent
    cfg_stem = config_path.stem

    root_dir = cfg_dir / cfg_stem
    # Always ensure the top-level directory exists so its name matches the config
    root_dir.mkdir(parents=True, exist_ok=True)

    # If specific run directory specified, validate and return it
    if resume_from:
        target_dir = root_dir / resume_from
        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(f"Specified run directory does not exist: {resume_from}")
        print(f"📂 使用指定的运行目录: {resume_from}")
        return target_dir

    # If resume mode enabled, check for existing runs
    if resume_mode and start_date and end_date:
        run_infos = _list_existing_runs(root_dir, start_date, end_date)
        
        if run_infos:
            # Filter out already completed runs for resume
            resumable_runs = [r for r in run_infos 
                            if r['resume_info']['can_resume'] or 
                               not r['resume_info'].get('already_completed', False)]
            
            if resumable_runs:
                if interactive and len(resumable_runs) > 1:
                    # Multiple runs available - let user choose
                    selected_dir = _prompt_user_run_selection(resumable_runs)
                    if selected_dir:
                        return selected_dir
                    # User chose 'new' - fall through to create new directory
                elif resumable_runs:
                    # Single run or non-interactive - use most recent
                    selected = resumable_runs[0]
                    print(f"📂 自动选择最新的运行目录: {selected['name']}")
                    return selected['path']

    # Create a unique run folder under the top-level directory to avoid collisions
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"📂 创建新的运行目录: run_{ts}")

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

    # Get root directory and dates early for --list-runs
    cfg_dir = cfg_path.parent
    cfg_stem = cfg_path.stem
    root_dir = cfg_dir / cfg_stem
    root_dir.mkdir(parents=True, exist_ok=True)
    
    start_arg = ns["start_date"] if isinstance(ns, dict) else ns.start_date
    simulation_start = get_or_init_simulation_start(root_dir, start_arg)
    end_date = str(ns["end_date"]) if isinstance(ns, dict) else ns.end_date

    # Handle --list-runs command
    if ns.list_runs:
        print("\n" + "="*80)
        print("📋 可用的运行目录列表")
        print("="*80)
        
        run_infos = _list_existing_runs(root_dir, simulation_start, end_date)
        
        if not run_infos:
            print("\n❌ 未找到任何运行目录")
            print(f"   目录: {root_dir}")
            return 0
        
        for idx, info in enumerate(run_infos, 1):
            resume_info = info['resume_info']
            print(f"\n[{idx}] {info['name']}")
            print(f"    📂 路径: {info['path']}")
            
            if resume_info.get('already_completed', False):
                print(f"    ✅ 状态: 已完成")
                print(f"    📅 最后日期: {resume_info['last_complete_date']}")
                print(f"    📊 完成天数: {resume_info['days_completed']}")
            elif resume_info['can_resume']:
                print(f"    🔄 状态: 可续跑")
                print(f"    📅 已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
                print(f"    📅 剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            else:
                print(f"    📝 状态: 无可续跑数据")
                print(f"    📊 需处理: {resume_info['days_remaining']} 天")
        
        print("\n" + "="*80)
        print(f"💡 续跑提示:")
        print(f"   python run.py --config {cfg_path} --end-date {end_date} --resume-from <run_dir_name>")
        print("="*80)
        return 0

    # Determine output directory and resume mode
    enable_resume = (ns.resume or ns.resume_from) and not ns.force_restart
    output_base_dir = _ensure_output_dir(
        cfg_path, 
        resume_mode=enable_resume,
        resume_from=ns.resume_from,
        start_date=simulation_start,
        end_date=end_date,
        interactive=not ns.non_interactive
    )

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

# ================================
# 运行示例 (Run Examples)
# ================================

# 1. 首次运行 (需要提供 --start-date)
# First run requires --start-date
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --start-date 2024-01-01 \
#   --end-date 2024-01-31

# 2. 列出所有可用的运行目录及其状态
# List all available run directories and their status
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --end-date 2024-01-31 \
#   --list-runs

# 3. 自动续跑 (交互式选择目录)
# Resume with interactive directory selection (multiple runs)
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --end-date 2024-01-31 \
#   --resume

# 4. 自动续跑 (非交互，自动选择最新目录)
# Resume non-interactively (auto-select most recent)
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --end-date 2024-01-31 \
#   --resume \
#   --non-interactive

# 5. 指定特定目录续跑
# Resume from specific run directory
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --end-date 2024-01-31 \
#   --resume-from run_20241203_120000

# 6. 检查续跑状态 (不运行)
# Check resume status without running
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --end-date 2024-01-31 \
#   --check-resume

# 7. 强制从头开始
# Force restart from beginning
# python run.py \
#   --config /path/to/YourConfig.xlsx \
#   --start-date 2024-01-01 \
#   --end-date 2024-01-31 \
#   --force-restart
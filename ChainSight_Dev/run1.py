#!/usr/bin/env python3
"""
ChainSight_Dev 运行脚本 - 带总运行时间记录功能

基于 run.py 增加了总运行时间统计和详细计时功能
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# External system imports
from main_integration import run_integrated_simulation, load_configuration, check_resume_capability
from logger_config import setup_logging


class TimingStats:
    """运行时间统计类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.daily_times = []
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        self.end_datetime = datetime.now()
    
    def add_daily_time(self, day_num: int, elapsed: float):
        """记录每日耗时"""
        self.daily_times.append({'day': day_num, 'elapsed': elapsed})
    
    @property
    def total_seconds(self) -> float:
        """总运行秒数"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    def format_duration(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}分钟 {secs:.2f}秒"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}小时 {mins}分钟 {secs:.2f}秒"
    
    def print_summary(self, num_days: int):
        """打印运行时间汇总"""
        print("\n" + "="*80)
        print("⏱️  运行时间统计报告")
        print("="*80)
        print(f"🕐 开始时间: {self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 结束时间: {self.end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  总运行时间: {self.format_duration(self.total_seconds)}")
        print(f"📊 仿真天数: {num_days} 天")
        if num_days > 0:
            avg_per_day = self.total_seconds / num_days
            print(f"📊 平均每天耗时: {self.format_duration(avg_per_day)}")
        print("="*80)
        
        return {
            'total_seconds': self.total_seconds,
            'total_formatted': self.format_duration(self.total_seconds),
            'num_days': num_days,
            'avg_per_day': self.total_seconds / num_days if num_days > 0 else 0,
        }


def _ensure_output_dir(config_path: Path, resume_mode: bool = False, 
                      resume_from: Optional[str] = None, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interactive: bool = True) -> Path:
    """创建输出目录"""
    cfg_dir = config_path.parent
    cfg_stem = config_path.stem

    root_dir = cfg_dir / cfg_stem
    root_dir.mkdir(parents=True, exist_ok=True)

    if resume_from:
        target_dir = root_dir / resume_from
        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(f"指定的运行目录不存在: {resume_from}")
        print(f"📂 使用指定的运行目录: {resume_from}")
        return target_dir

    # 创建新的运行目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"📂 创建新的运行目录: run_{ts}")

    return run_dir


def get_or_init_simulation_start(output_root: Path, provided_start: Optional[str]) -> str:
    """获取或初始化仿真开始日期"""
    start_file = output_root / "simulation_start.txt"
    if start_file.exists():
        return start_file.read_text(encoding="utf-8").strip()
    if not provided_start:
        raise ValueError("首次运行需要提供仿真开始日期")
    start_file.write_text(provided_start, encoding="utf-8")
    return provided_start


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run1",
        add_help=True,
        description="ChainSight_Dev 运行脚本 - 带总运行时间记录功能",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="配置文件路径 (.xlsx)",
    )
    parser.add_argument(
        "--start-date",
        required=False,
        help="仿真开始日期 YYYY-MM-DD (首次运行必须)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="仿真结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="强制从头开始，忽略续跑",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="指定续跑的运行目录 (例如 run_20241203_120000)",
    )
    return parser.parse_args(argv)


def calculate_simulation_days(start_date: str, end_date: str) -> int:
    """计算仿真天数"""
    from datetime import datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return (end - start).days + 1


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv or sys.argv[1:])

    cfg_path = Path(ns.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {cfg_path}")
    if cfg_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("配置文件必须是Excel文件 (.xlsx/.xlsm/.xls)")

    # 验证配置文件
    _ = load_configuration(str(cfg_path))

    # 获取根目录和日期
    cfg_dir = cfg_path.parent
    cfg_stem = cfg_path.stem
    root_dir = cfg_dir / cfg_stem
    root_dir.mkdir(parents=True, exist_ok=True)
    
    start_arg = ns.start_date
    simulation_start = get_or_init_simulation_start(root_dir, start_arg)
    end_date = ns.end_date

    # 计算仿真天数
    num_days = calculate_simulation_days(simulation_start, end_date)

    # 确定输出目录
    enable_resume = ns.resume_from and not ns.force_restart
    output_base_dir = _ensure_output_dir(
        cfg_path, 
        resume_mode=enable_resume,
        resume_from=ns.resume_from,
        start_date=simulation_start,
        end_date=end_date,
        interactive=True
    )

    # 设置日志系统
    logger, redirector = setup_logging(str(output_base_dir), log_level="INFO", redirect_print=True)
    
    # 初始化计时器
    timing = TimingStats()
    
    print("\n" + "="*80)
    print("🚀 ChainSight_Dev 供应链仿真系统启动")
    print("="*80)
    print(f"📂 配置文件: {cfg_path}")
    print(f"📁 输出目录: {output_base_dir}")
    print(f"📅 仿真日期范围: {simulation_start} 到 {end_date}")
    print(f"📊 预计仿真天数: {num_days} 天")
    print("="*80 + "\n")
    
    try:
        # 开始计时
        timing.start()
        
        # 运行仿真
        _ = run_integrated_simulation(
            config_path=str(cfg_path),
            start_date=simulation_start,
            end_date=end_date,
            output_base_dir=str(output_base_dir),
            force_restart=ns.force_restart,
        )
        
        # 停止计时
        timing.stop()
        
        # 打印时间统计
        stats = timing.print_summary(num_days)
        
        # 保存时间统计到文件
        timing_file = output_base_dir / "timing_stats.txt"
        with open(timing_file, 'w', encoding='utf-8') as f:
            f.write("ChainSight_Dev 运行时间统计报告\n")
            f.write("="*50 + "\n")
            f.write(f"配置文件: {cfg_path}\n")
            f.write(f"仿真日期: {simulation_start} 到 {end_date}\n")
            f.write(f"仿真天数: {num_days} 天\n")
            f.write(f"开始时间: {timing.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"结束时间: {timing.end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总运行时间: {stats['total_formatted']}\n")
            f.write(f"总运行秒数: {stats['total_seconds']:.2f}秒\n")
            f.write(f"平均每天耗时: {timing.format_duration(stats['avg_per_day'])}\n")
            f.write("="*50 + "\n")
        
        print(f"\n📝 时间统计已保存到: {timing_file}")
        print("✅ 仿真成功完成")
        return 0
        
    except Exception as e:
        timing.stop()
        print(f"\n❌ 仿真执行出错: {str(e)}")
        print(f"⏱️  运行时间: {timing.format_duration(timing.total_seconds)}")
        raise
    finally:
        if redirector:
            redirector.stop_redirect()
            print(f"\n📝 完整日志已保存到: {output_base_dir}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise e
    except Exception as exc:
        raise SystemExit(1) from exc

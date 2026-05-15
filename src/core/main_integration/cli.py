"""
cli.py

命令行接口入口点模块。
"""

import sys
import os
import argparse
import io
import logging
from pathlib import Path

# Windows UTF-8 编码设置 - 解决emoji和中文输出问题
if sys.platform == 'win32':
    # 设置stdout/stderr为UTF-8编码
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from .simulation_file import run_integrated_simulation
from .resume import check_resume_capability

# 复用 src/utils/logger_config.py::DualLogger 创建的同名 logger，
# 这样消息既能进控制台又能进 simulation_log_*.txt。
logger = logging.getLogger("SupplyChainSimulation")


def main():
    """主函数 - 命令行入口执行集成仿真

    目的：
    - 解析命令行参数，进行存在性检查与默认值处理，支持仅检查断点续跑状态或执行完整仿真。

    输入/输出/逻辑：
    - 解析参数→检查配置文件→构造默认输出目录→可选断点续跑检查→调用 `run_integrated_simulation` 并打印结果或错误。
    """
    # 配置文件路径（可以通过命令行参数或环境变量指定）
    parser = argparse.ArgumentParser(description="运行供应链集成仿真")
    cfg_group = parser.add_mutually_exclusive_group()
    cfg_group.add_argument("--config-dir",
                           default=None,
                           help=("场景 config/ 目录绝对路径，或 <project>/<scenario> 短格式；"
                                 "短格式按 workspace_root 展开"))
    cfg_group.add_argument("--config", "-c",
                           default=None,
                           help="配置文件路径或含唯一 Excel 的目录路径；与 --config-dir 平级支持")
    parser.add_argument("--start-date", "-s",
                       default="2024-01-01",
                       help="仿真开始日期 (默认: 2024-01-01)")
    parser.add_argument("--end-date", "-e",
                       default="2024-01-05",
                       help="仿真结束日期 (默认: 2024-01-05)")
    parser.add_argument("--output", "-o",
                       default=None,
                       help="输出目录 (默认: 根据配置文件名生成)")
    parser.add_argument("--force-restart",
                       action="store_true",
                       help="强制从头开始，忽略断点续跑状态 (默认: False)")
    parser.add_argument("--check-resume",
                       action="store_true",
                       help="仅检查断点续跑状态，不执行仿真 (默认: False)")

    args = parser.parse_args()

    if args.config_dir:
        try:
            from ..run.config_dir import ConfigDir
            from ..run.utils import expand_config_dir_arg

            cfg = ConfigDir.from_path(expand_config_dir_arg(args.config_dir))
            args.config = str(cfg.excel_path)
            if args.output is None:
                args.output = str(Path("outputs") / cfg.output_subpath)
                logger.info(f"💫 使用默认输出目录: {args.output}")
        except (ValueError, FileNotFoundError) as e:
            print(f"[ConfigError] {e}", file=sys.stderr)
            sys.exit(2)
    else:
        # 过渡期旧路径：保持原默认值与存在性检查。
        if args.config is None:
            args.config = "./config/integration_config.json"

        # 检查配置文件是否存在
        if not os.path.exists(args.config):
            logger.error(f"❌ 配置文件不存在: {args.config}")
            logger.error("请提供有效的配置文件路径，或使用测试脚本生成配置")
            sys.exit(1)

    # 如果没有指定输出目录，根据配置文件名生成
    if args.output is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output = str(Path("outputs") / config_name)
        logger.info(f"💫 使用默认输出目录: {args.output}")

    # 处理断点续跑检查选项
    if args.check_resume:
        logger.info("🔍 检查断点续跑状态...")
        resume_info = check_resume_capability(args.output, args.start_date, args.end_date)

        logger.info("📊 断点续跑状态报告:")
        logger.info(f"  输出目录: {args.output}")
        logger.info(f"  原始日期范围: {args.start_date} 到 {args.end_date}")

        if resume_info.get('already_completed', False):
            logger.info("  ✅ 仿真已完成！")
            logger.info(f"     最后处理日期: {resume_info['last_complete_date']}")
            logger.info(f"     总处理天数: {resume_info['days_completed']}")
        elif resume_info['can_resume']:
            logger.info("  🔄 可以断点续跑！")
            logger.info(
                f"     已完成: {resume_info['days_completed']} 天 (到 {resume_info['last_complete_date']})"
            )
            logger.info(
                f"     剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)"
            )
        else:
            logger.info("  📝 无法断点续跑，需要从头开始")
            logger.info(f"     需要处理: {resume_info['days_remaining']} 天")

        return  # 仅检查，不执行

    # 处理强制重启选项
    if args.force_restart:
        # 通过参数透传给仿真入口，显式要求忽略已有断点续跑状态
        logger.info("🔄 强制重启模式：将从头开始，忽略任何现有状态")

    try:
        # 向仿真入口透传 force_restart 参数
        result = run_integrated_simulation(
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            output_base_dir=args.output,
            force_restart=args.force_restart,
        )

        logger.info("✅ 仿真结果:")
        if result.get('is_resuming', False):
            logger.info("  断点续跑模式: 是")
            logger.info(f"  本次处理天数: {result.get('dates_processed_this_run', 0)}")
            logger.info(f"  总处理天数: {result.get('total_dates_processed', 0)}")
        else:
            logger.info("  全新运行: 是")
            logger.info(f"  处理天数: {result.get('dates_processed_this_run', 0)}")
        logger.info(f"  输出目录: {result.get('output_directory', 'Unknown')}")

        if result.get('already_completed', False):
            logger.info("  📝 注意: 仿真之前已完成，无需处理")

    except Exception as e:
        logger.error(f"❌ 集成仿真失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

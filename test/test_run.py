"""ChainSight 集成仿真测试入口。

运行方式:
    python test/test_run.py --config ./config/OC_Paste_S1_20251224.xlsx --start-date 2025-12-15 --end-date 2026-02-28
"""

import sys
import argparse
from pathlib import Path

# 确保项目根目录和 test 目录可导入
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="ChainSight 集成仿真测试")
    parser.add_argument("--config", required=True, help="配置文件路径 (xlsx)")
    parser.add_argument("--start-date", required=True, help="仿真开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="仿真结束日期 (YYYY-MM-DD)")
    parser.add_argument("--engine", choices=("pandas", "polars"), default="polars", help="计算后端（默认 polars）")
    parser.add_argument("--test", action="store_true", help="启用测试隔离模式")
    parser.add_argument("--test-schema", help="测试隔离模式写入的 PostgreSQL schema（默认 test）")
    parser.add_argument("--no-persist", action="store_true", help="禁用数据库持久化，仅在内存中运行")
    parser.add_argument("--skip-dq", action="store_true", help="跳过配置数据质量检测（性能测试使用）")
    parser.add_argument("--verbose", action="store_true", help="输出模块内部逐步骤耗时日志")
    parser.add_argument("--performance-report", help="性能报告 JSON 输出路径")
    parser.add_argument("--run-mode", default="continuous", help="性能报告运行模式标签")
    args = parser.parse_args()

    # 使用与 run_main.py 相同的输出目录逻辑: outputs/<config_stem>/run_YYYYMMDD_HHMMSS/
    from src.core.run.output_dir import _ensure_output_dir

    cfg_path = Path(args.config).resolve()
    output_base_dir = _ensure_output_dir(cfg_path)

    from test_integration import run_integrated_simulation

    result = run_integrated_simulation(
        config_path=str(cfg_path),
        start_date=args.start_date,
        end_date=args.end_date,
        output_base_dir=str(output_base_dir),
        engine=args.engine,
        test_mode=args.test,
        test_schema=args.test_schema,
        enable_persistence=not args.no_persist,
        skip_dq=args.skip_dq,
        verbose=args.verbose,
        performance_report=args.performance_report,
        run_mode=args.run_mode,
    )
    assert result is not None, "仿真返回 None"
    assert result.get('simulation_completed') is True, f"仿真未完成: {result}"
    print(f"\n✅ 仿真完成，共处理 {result['dates_processed']} 天")
    print(f"   输出目录: {result['output_directory']}")
    if result.get('performance_report'):
        print(f"   性能报告: {result['performance_report']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
demand_planning_refactor 每日 Excel 完整输出

支持指定 material（全部 location）或单个 (material, location)。

输出: outputs/demand_planning_refactor/<material>/
  ├── module1_output_20251215.xlsx   (5 sheets: OrderLog, ShipmentLog, CutLog, SupplyDemandLog, Summary)
  ├── module1_output_20251216.xlsx
  └── ...

用法:
  python generate_daily_excel_output.py                          # 默认 SKU
  python generate_daily_excel_output.py 80848026                 # 指定 material（全部 location）
  python generate_daily_excel_output.py 80848026 A668            # 指定 (material, location)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules import demand_planning_refactor as refactor_m1
from src.utils.logger_config import setup_logging
from functional_tests.utils.m1_fixtures import M1TestOrchestrator, load_real_m1_config


def main():
    # 解析参数
    args = sys.argv[1:]
    if len(args) >= 1:
        try:
            test_material = int(args[0])
        except ValueError:
            print(f"material 必须是整数，当前: {args[0]}")
            return 1
    else:
        test_material = 21029016

    test_location = args[1] if len(args) >= 2 else None

    # 输出目录以 material 命名
    output_dir = PROJECT_ROOT / "outputs" / "demand_planning_refactor" / str(test_material)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(str(output_dir), log_level="INFO")

    logger.info("=" * 70)
    logger.info("demand_planning_refactor 每日 Excel 完整输出")
    logger.info("=" * 70)

    # 加载配置
    config_file = PROJECT_ROOT / "config" / "OC_Paste_S1_20251224.xlsx"
    logger.info(f"配置文件: {config_file}")
    config = load_real_m1_config(config_file)

    # 过滤 DemandForecast
    df = config["M1_DemandForecast"]
    mask = df["material"] == test_material
    if test_location:
        mask = mask & (df["location"] == test_location)
    config["M1_DemandForecast"] = df[mask].copy()

    assert not config["M1_DemandForecast"].empty, f"material={test_material} 无数据"

    locations = config["M1_DemandForecast"]["location"].unique().tolist()
    sku_label = f"{test_material}" if not test_location else f"{test_material}/{test_location}"
    logger.info(f"测试 SKU: {sku_label}  ({len(locations)} 个 location: {locations})")

    # 仿真日期范围
    sim_start = pd.Timestamp("2025-12-15")
    sim_end = pd.Timestamp("2026-02-28")
    sim_dates = pd.to_datetime(config["M1_OrderCalendar"]["date"]).dt.normalize().unique()
    sim_dates = sorted([d for d in sim_dates if sim_start <= d <= sim_end])
    logger.info(f"仿真期间: {sim_start.date()} ~ {sim_end.date()}，{len(sim_dates)} 个订单日")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"文件命名: module1_output_YYYYMMDD.xlsx")
    logger.info("")

    orchestrator = M1TestOrchestrator(start_date=str(sim_start.date()))
    files_written = 0
    errors = 0

    for i, sim_date in enumerate(sim_dates):
        np.random.seed(42 + i)
        date_str = sim_date.strftime("%Y-%m-%d")

        try:
            result = refactor_m1.run_daily_order_generation(
                config,
                sim_date,
                str(output_dir),
                orchestrator=orchestrator,
                skip_file_output=False,
            )

            if result.get("output_file"):
                files_written += 1

        except Exception as e:
            logger.error(f"  {date_str} 失败: {e}")
            errors += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(sim_dates):
            orders_cnt = len(result.get("orders_df", [])) if result else 0
            logger.info(f"  [{i+1:3d}/{len(sim_dates)}] {date_str}  cumulative_orders={orders_cnt}")

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"✓ 完成: {files_written} 个 Excel 文件，{errors} 个错误")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  每个文件 5 sheets: OrderLog / ShipmentLog / CutLog / SupplyDemandLog / Summary")
    logger.info("=" * 70)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

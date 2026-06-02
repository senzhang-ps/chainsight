#!/usr/bin/env python3
"""
demand_planning_refactor 单 SKU 测试脚本

用 OC_Paste_S1_20251224.xlsx，选择 material=21029016, location=A668
从 2025-12-15 仿真到 2026-02-28，验证周级→日级订单拆分
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.demand_planning_refactor.order import generate_weekly_orders, split_weekly_orders_to_daily
from src.utils.logger_config import setup_logging

def main():
    output_dir = PROJECT_ROOT / "outputs" / "refactor_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(str(output_dir), log_level="INFO")

    logger.info("=" * 60)
    logger.info("demand_planning_refactor 单 SKU 仿真测试")
    logger.info("=" * 60)

    # 加载 xlsx
    config_file = PROJECT_ROOT / "config" / "OC_Paste_S1_20251224.xlsx"
    logger.info(f"配置文件: {config_file}")

    demand_forecast = pd.read_excel(config_file, sheet_name="M1_DemandForecast")
    ao_config = pd.read_excel(config_file, sheet_name="M1_AOConfig")
    order_calendar = pd.read_excel(config_file, sheet_name="M1_OrderCalendar")
    forecast_error = pd.read_excel(config_file, sheet_name="M1_ForecastError")

    # 筛选 SKU
    test_material = 21029016
    test_location = "A668"
    sku = demand_forecast[
        (demand_forecast["material"] == test_material) &
        (demand_forecast["location"] == test_location)
    ].copy()

    if sku.empty:
        logger.error(f"SKU {test_material}/{test_location} 无数据")
        return 1

    logger.info(f"\n测试 SKU: {test_material}/{test_location}")
    logger.info(f"  周级预测: {len(sku)} 行，week {sku['week'].min():.0f}-{sku['week'].max():.0f}")
    logger.info(f"  总需求量: {sku['quantity'].sum():.0f}")

    # 生成周级订单
    logger.info("\n[Step 1] 生成周级订单...")
    try:
        weekly_orders = generate_weekly_orders(sku, ao_config, forecast_error)
        logger.info(f"  周级订单: {len(weekly_orders)} 行")
        logger.info(f"    - AO 总量: {weekly_orders[weekly_orders['demand_type']=='AO']['quantity'].sum():.0f}")
        logger.info(f"    - Normal 总量: {weekly_orders[weekly_orders['demand_type']=='normal']['quantity'].sum():.0f}")
        logger.info(f"    - 合计: {weekly_orders['quantity'].sum():.0f}")
    except Exception as e:
        logger.error(f"周级订单生成失败: {e}", exc_info=True)
        return 1

    # 日期范围
    sim_dates = pd.to_datetime(order_calendar["date"]).dt.normalize().unique()
    sim_dates = sorted([d for d in sim_dates if pd.Timestamp("2025-12-15") <= d <= pd.Timestamp("2026-02-28")])
    logger.info(f"\n[Step 2] 日级拆分")
    logger.info(f"  仿真期间: {sim_dates[0].date()} ~ {sim_dates[-1].date()}，{len(sim_dates)} 个订单日")

    # 逐日拆分
    all_daily = []
    for i, sim_date in enumerate(sim_dates):
        try:
            daily = split_weekly_orders_to_daily(weekly_orders, sim_date, order_calendar)
            if not daily.empty:
                all_daily.append(daily)
        except Exception as e:
            logger.warning(f"  日期 {sim_date.date()} 失败: {e}")

    if all_daily:
        combined = pd.concat(all_daily, ignore_index=True)
        logger.info(f"  日级订单: {len(combined)} 行")
        logger.info(f"    - AO 总量: {combined[combined['demand_type']=='AO']['quantity'].sum():.0f}")
        logger.info(f"    - Normal 总量: {combined[combined['demand_type']=='normal']['quantity'].sum():.0f}")
        logger.info(f"    - 合计: {combined['quantity'].sum():.0f}")

        # 按 demand_date 分析
        combined['demand_date'] = pd.to_datetime(combined['date'])
        by_date = combined.groupby('demand_date')['quantity'].sum()
        logger.info(f"\n[Step 3] 需求日期分析")
        logger.info(f"  需求日期范围: {by_date.index.min().date()} ~ {by_date.index.max().date()}")
        logger.info(f"  需求日期数: {len(by_date)}")
        logger.info(f"  日均订单: {by_date.mean():.1f}")

        # 输出到 csv
        out_csv = output_dir / "test_sku_orders.csv"
        combined.to_csv(out_csv, index=False)
        logger.info(f"\n✓ 日级订单已保存: {out_csv}")
    else:
        logger.warning("未生成任何日级订单")
        return 1

    logger.info("\n" + "=" * 60)
    logger.info("✓ 仿真完毕")
    logger.info("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
生成 demand_planning_refactor 测试报告

- 运行单 SKU 3 个月仿真
- 输出详细数据和分析
"""

import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.demand_planning_refactor.order import generate_weekly_orders, split_weekly_orders_to_daily
from src.utils.logger_config import setup_logging

def main():
    output_dir = PROJECT_ROOT / "outputs" / "test_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(str(output_dir), log_level="INFO")

    logger.info("=" * 70)
    logger.info("demand_planning_refactor 测试报告生成")
    logger.info("=" * 70)

    # 加载配置
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
    weekly_orders = generate_weekly_orders(sku, ao_config, forecast_error)
    logger.info(f"  周级订单: {len(weekly_orders)} 行")
    logger.info(f"    - AO 总量: {weekly_orders[weekly_orders['demand_type']=='AO']['quantity'].sum():.0f}")
    logger.info(f"    - Normal 总量: {weekly_orders[weekly_orders['demand_type']=='normal']['quantity'].sum():.0f}")
    logger.info(f"    - 合计: {weekly_orders['quantity'].sum():.0f}")

    # 日期范围
    sim_dates = pd.to_datetime(order_calendar["date"]).dt.normalize().unique()
    sim_dates = sorted([d for d in sim_dates if pd.Timestamp("2025-12-15") <= d <= pd.Timestamp("2026-02-28")])
    logger.info(f"\n[Step 2] 日级拆分")
    logger.info(f"  仿真期间: {sim_dates[0].date()} ~ {sim_dates[-1].date()}，{len(sim_dates)} 个订单日")

    # 逐日拆分
    all_daily = []
    for i, sim_date in enumerate(sim_dates):
        daily = split_weekly_orders_to_daily(weekly_orders, sim_date, order_calendar)
        if not daily.empty:
            all_daily.append(daily)
        if (i + 1) % 20 == 0:
            logger.info(f"  进度: {i+1}/{len(sim_dates)}")

    if all_daily:
        combined = pd.concat(all_daily, ignore_index=True)
        logger.info(f"\n  日级订单: {len(combined)} 行")
        logger.info(f"    - AO 总量: {combined[combined['demand_type']=='AO']['quantity'].sum():.0f}")
        logger.info(f"    - Normal 总量: {combined[combined['demand_type']=='normal']['quantity'].sum():.0f}")
        logger.info(f"    - 合计: {combined['quantity'].sum():.0f}")

        # 按日期分析
        combined['demand_date'] = pd.to_datetime(combined['date'])
        by_date = combined.groupby('demand_date').agg({
            'quantity': ['sum', 'count', 'mean', 'min', 'max']
        }).round(1)
        by_date.columns = ['总量', '订单行数', '平均', '最小', '最大']

        logger.info(f"\n[Step 3] 需求日期分析")
        logger.info(f"  需求日期范围: {by_date.index.min().date()} ~ {by_date.index.max().date()}")
        logger.info(f"  需求日期数: {len(by_date)}")
        logger.info(f"  日均订单: {by_date['总量'].mean():.1f}")
        logger.info(f"  订单日均行数: {by_date['订单行数'].mean():.1f}")

        # 按 demand_type 统计
        logger.info(f"\n[Step 4] 订单类型分析")
        by_type = combined.groupby('demand_type')['quantity'].sum()
        for dtype in ['AO', 'normal']:
            if dtype in by_type.index:
                pct = by_type[dtype] / by_type.sum() * 100
                logger.info(f"  {dtype}: {by_type[dtype]:.0f} ({pct:.1f}%)")

        # 输出数据文件
        logger.info(f"\n[Step 5] 输出数据文件...")

        # 周级订单
        weekly_csv = output_dir / "weekly_orders.csv"
        weekly_orders.to_csv(weekly_csv, index=False)
        logger.info(f"  周级订单: {weekly_csv}")

        # 日级订单
        daily_csv = output_dir / "daily_orders.csv"
        combined.to_csv(daily_csv, index=False)
        logger.info(f"  日级订单: {daily_csv}")

        # 日期汇总
        summary_csv = output_dir / "daily_summary.csv"
        by_date.to_csv(summary_csv)
        logger.info(f"  日期汇总: {summary_csv}")

        # 生成 JSON 报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "sku": {
                "material": int(test_material),
                "location": test_location,
            },
            "config": {
                "config_file": str(config_file),
                "sim_period": {
                    "start": str(sim_dates[0].date()),
                    "end": str(sim_dates[-1].date()),
                    "days": len(sim_dates),
                },
            },
            "weekly_orders": {
                "count": len(weekly_orders),
                "ao_total": float(weekly_orders[weekly_orders['demand_type']=='AO']['quantity'].sum()),
                "normal_total": float(weekly_orders[weekly_orders['demand_type']=='normal']['quantity'].sum()),
                "total": float(weekly_orders['quantity'].sum()),
            },
            "daily_orders": {
                "count": len(combined),
                "ao_total": float(combined[combined['demand_type']=='AO']['quantity'].sum()),
                "normal_total": float(combined[combined['demand_type']=='normal']['quantity'].sum()),
                "total": float(combined['quantity'].sum()),
                "demand_dates": int(len(by_date)),
                "avg_daily_qty": float(by_date['总量'].mean()),
                "avg_lines_per_date": float(by_date['订单行数'].mean()),
            },
            "validation": {
                "all_non_negative": (combined['quantity'] >= 0).all().item() if hasattr((combined['quantity'] >= 0).all(), 'item') else bool((combined['quantity'] >= 0).all()),
                "weekly_daily_conserved": float(abs(weekly_orders['quantity'].sum() - combined['quantity'].sum()) < 1),
                "sim_dates_match": int(len(set(combined['simulation_date'].dt.date)) <= len(sim_dates)),
            },
        }

        report_json = output_dir / "test_report.json"
        with open(report_json, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"  测试报告: {report_json}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ 测试报告生成完毕")
    logger.info("=" * 70)
    logger.info(f"\n输出目录: {output_dir}")
    logger.info(f"文件列表：")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            logger.info(f"  - {f.name} ({size:,} bytes)")

    return 0

if __name__ == "__main__":
    sys.exit(main())

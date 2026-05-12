#!/usr/bin/env python3
"""
按日期输出测试数据 - 每一天一个 CSV 文件

输出结构：
functional_tests/_tmp/test_report_daily/
  ├── daily/
  │   ├── 2025-12-15.csv
  │   ├── 2025-12-16.csv
  │   └── ...
  ├── daily_summary.csv
  ├── summary.json
  └── README.md
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
    output_dir = PROJECT_ROOT / "outputs" / "test_report_daily"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(str(output_dir), log_level="INFO")

    logger.info("=" * 70)
    logger.info("demand_planning_refactor 按日期输出测试数据")
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
    logger.info(f"  周级预测: {len(sku)} 行，总需求: {sku['quantity'].sum():.0f}")

    # 生成周级订单
    logger.info("\n[Step 1] 生成周级订单...")
    weekly_orders = generate_weekly_orders(sku, ao_config, forecast_error)
    logger.info(f"  周级订单: {len(weekly_orders)} 行，总量: {weekly_orders['quantity'].sum():.0f}")

    # 日期范围
    sim_dates = pd.to_datetime(order_calendar["date"]).dt.normalize().unique()
    sim_dates = sorted([d for d in sim_dates if pd.Timestamp("2025-12-15") <= d <= pd.Timestamp("2026-02-28")])
    logger.info(f"\n[Step 2] 按日期生成订单文件")
    logger.info(f"  仿真期间: {sim_dates[0].date()} ~ {sim_dates[-1].date()}，{len(sim_dates)} 个订单日")

    # 创建日期子目录
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(exist_ok=True)

    # 逐日输出
    all_daily = []
    daily_stats = []

    for i, sim_date in enumerate(sim_dates):
        daily = split_weekly_orders_to_daily(weekly_orders, sim_date, order_calendar)

        if not daily.empty:
            # 保存该天的订单到独立文件
            date_str = sim_date.strftime("%Y-%m-%d")
            csv_file = daily_dir / f"{date_str}.csv"
            daily.to_csv(csv_file, index=False)

            # 统计该天的数据
            stats = {
                "date": date_str,
                "orders": len(daily),
                "ao_qty": int(daily[daily["demand_type"] == "AO"]["quantity"].sum()),
                "normal_qty": int(daily[daily["demand_type"] == "normal"]["quantity"].sum()),
                "total_qty": int(daily["quantity"].sum()),
                "demand_dates": int(daily["date"].nunique()),
            }
            daily_stats.append(stats)
            all_daily.append(daily)

            if (i + 1) % 20 == 0:
                logger.info(f"  进度: {i+1}/{len(sim_dates)} ✓")

    logger.info(f"\n[Step 3] 输出统计信息")
    if all_daily:
        combined = pd.concat(all_daily, ignore_index=True)
        logger.info(f"  总日级订单: {len(combined)} 行")
        logger.info(f"    - AO 总量: {combined[combined['demand_type']=='AO']['quantity'].sum():.0f}")
        logger.info(f"    - Normal 总量: {combined[combined['demand_type']=='normal']['quantity'].sum():.0f}")

    # 生成日期统计汇总 CSV
    logger.info(f"\n[Step 4] 生成汇总文件...")
    stats_df = pd.DataFrame(daily_stats)
    summary_csv = output_dir / "daily_summary.csv"
    stats_df.to_csv(summary_csv, index=False)
    logger.info(f"  日期汇总: {summary_csv}")

    # 生成 JSON 报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "sku": {
            "material": int(test_material),
            "location": test_location,
        },
        "sim_period": {
            "start": str(sim_dates[0].date()),
            "end": str(sim_dates[-1].date()),
            "days": len(sim_dates),
        },
        "output_format": "每日一个 CSV 文件",
        "files_generated": {
            "daily_files": len(daily_stats),
            "summary_csv": "daily_summary.csv",
            "daily_dir": "daily/",
        },
        "statistics": {
            "total_orders": int(stats_df["orders"].sum()) if not stats_df.empty else 0,
            "total_ao_qty": int(stats_df["ao_qty"].sum()) if not stats_df.empty else 0,
            "total_normal_qty": int(stats_df["normal_qty"].sum()) if not stats_df.empty else 0,
            "avg_daily_orders": float(stats_df["orders"].mean()) if not stats_df.empty else 0,
            "avg_daily_qty": float(stats_df["total_qty"].mean()) if not stats_df.empty else 0,
        },
    }

    report_json = output_dir / "summary.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"  测试报告: {report_json}")

    # 生成 README
    readme = output_dir / "README.md"
    with open(readme, "w") as f:
        f.write(f"""# 按日期输出的测试数据

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据结构

```
daily/
├── 2025-12-15.csv
├── 2025-12-16.csv
├── ...
└── 2026-02-28.csv

daily_summary.csv  # 所有日期的汇总统计
summary.json       # 结构化报告
```

## 文件格式

### daily_summary.csv

| 列名 | 说明 |
|------|------|
| date | 下单日期 (YYYY-MM-DD) |
| orders | 该日订单行数 |
| ao_qty | 该日 AO 总量 |
| normal_qty | 该日 Normal 总量 |
| total_qty | 该日总订单量 |
| demand_dates | 该日对应的需求日期数 |

### daily/*.csv (单日文件)

每个文件代表一个订单日的订单明细，包含列：
- date: 需求日期
- material: 物料
- location: 地点
- demand_type: AO/normal
- simulation_date: 下单日期
- advance_days: 提前期
- quantity: 订单量

## 统计概览

| 指标 | 值 |
|------|-----|
| 测试 SKU | {test_material}/{test_location} |
| 仿真期间 | {sim_dates[0].strftime('%Y-%m-%d')} ~ {sim_dates[-1].strftime('%Y-%m-%d')} |
| 订单日数 | {len(sim_dates)} |
| 有订单日数 | {len(daily_stats)} |
| 总订单行数 | {int(stats_df['orders'].sum()) if not stats_df.empty else 0} |
| 总订单量 | {int(stats_df['total_qty'].sum()) if not stats_df.empty else 0} |
| 日均订单量 | {float(stats_df['total_qty'].mean()):.1f} |

## 使用方式

### 查看某一天的订单
```bash
cat daily/2025-12-15.csv
```

### 统计所有日期汇总
```bash
cat daily_summary.csv
```

### 读取 JSON 报告
```bash
cat summary.json
```
""")
    logger.info(f"  README: {readme}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ 日期输出完毕")
    logger.info("=" * 70)
    logger.info(f"\n输出目录: {output_dir}")
    logger.info(f"\n快速查看:")
    logger.info(f"  - 汇总: cat {summary_csv}")
    logger.info(f"  - 某日订单: cat {daily_dir}/2025-12-15.csv")
    logger.info(f"  - 报告: cat {report_json}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

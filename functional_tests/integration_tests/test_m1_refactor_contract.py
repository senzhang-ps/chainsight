from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from functional_tests.utils.m1_fixtures import (
    CUT_LOG_COLUMNS,
    FUNCTIONAL_TEST_TMP,
    ORDER_LOG_COLUMNS,
    SHIPMENT_LOG_COLUMNS,
    SUPPLY_DEMAND_LOG_COLUMNS,
    M1TestOrchestrator,
    build_weekly_m1_config,
    normalize_for_compare,
    load_real_m1_config,
    filter_sku_from_config,
)


def test_refactored_m1_result_keeps_downstream_contract_shapes():
    from src.modules import demand_planning_refactor as refactor_m1

    config = build_weekly_m1_config()
    FUNCTIONAL_TEST_TMP.mkdir(parents=True, exist_ok=True)
    output_dir = FUNCTIONAL_TEST_TMP / "contract_m1_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(7)
    result = refactor_m1.run_daily_order_generation(
        config,
        pd.Timestamp("2026-05-04"),
        str(output_dir),
        orchestrator=M1TestOrchestrator(),
        skip_file_output=True,
    )

    orders = normalize_for_compare(result["orders_df"], ORDER_LOG_COLUMNS)
    shipments = normalize_for_compare(result["shipment_df"], SHIPMENT_LOG_COLUMNS)
    cuts = normalize_for_compare(result["cut_df"], CUT_LOG_COLUMNS)
    supply_demand = normalize_for_compare(result["supply_demand_df"], SUPPLY_DEMAND_LOG_COLUMNS)

    assert list(orders.columns) == ORDER_LOG_COLUMNS
    assert list(shipments.columns) == SHIPMENT_LOG_COLUMNS
    assert list(cuts.columns) == CUT_LOG_COLUMNS
    assert list(supply_demand.columns) == SUPPLY_DEMAND_LOG_COLUMNS
    assert (orders["quantity"] >= 0).all()
    assert (shipments["quantity"] >= 0).all()
    assert (cuts["quantity"] >= 0).all()
    assert (supply_demand["quantity"] >= 0).all()


def test_real_sku_three_month_order_generation():
    """真实数据集成测试：用 OC_Paste_S1_20251224.xlsx 的完整配置测试 3 个月仿真。

    验证：
    - 周级订单生成无异常
    - 日级拆分按有效订单日进行
    - 需求日期覆盖范围合理
    - 订单量非负
    - 仿真跨多个订单日成功
    """
    from src.modules import demand_planning_refactor as refactor_m1

    # 加载真实完整配置（不筛选 SKU，保留全部配置）
    config_file = Path(__file__).resolve().parents[2] / "config" / "OC_Paste_S1_20251224.xlsx"
    if not config_file.exists():
        pytest.skip(f"配置文件不存在: {config_file}")

    config = load_real_m1_config(config_file)

    # 筛选 DemandForecast 中的单个 SKU（但保留全部配置表）
    test_material = 21029016
    test_location = "A668"
    df = config["M1_DemandForecast"]
    config["M1_DemandForecast"] = df[
        (df["material"] == test_material) & (df["location"] == test_location)
    ].copy()

    # 验证配置非空
    assert not config["M1_DemandForecast"].empty, f"SKU {test_material}/{test_location} 无预测数据"

    FUNCTIONAL_TEST_TMP.mkdir(parents=True, exist_ok=True)
    output_dir = FUNCTIONAL_TEST_TMP / "real_sku_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成 3 个月的订单（2025-12-15 ~ 2026-02-28）
    sim_start = pd.Timestamp("2025-12-15")
    sim_dates = pd.to_datetime(config["M1_OrderCalendar"]["date"]).dt.normalize().unique()
    sim_dates = sorted([d for d in sim_dates if sim_start <= d <= pd.Timestamp("2026-02-28")])

    assert len(sim_dates) > 0, "无有效订单日"

    all_orders = []
    for i, sim_date in enumerate(sim_dates):
        np.random.seed(7 + i)  # 不同日期用不同 seed 保证独立
        result = refactor_m1.run_daily_order_generation(
            config,
            sim_date,
            str(output_dir),
            orchestrator=M1TestOrchestrator(start_date=str(sim_start.date())),
            skip_file_output=True,
        )
        if not result["orders_df"].empty:
            all_orders.append(result["orders_df"])

    # 验证结果
    assert len(all_orders) > 0, "未生成任何订单"

    combined = pd.concat(all_orders, ignore_index=True)
    orders = normalize_for_compare(combined, ORDER_LOG_COLUMNS)

    # 基本校验
    assert (orders["quantity"] >= 0).all(), "存在负订单"
    assert orders["quantity"].sum() > 0, "总订单量为 0"
    assert len(set(orders["simulation_date"])) <= len(sim_dates), "订单日期超出范围"

    # 需求日期覆盖
    demand_dates = pd.to_datetime(orders["date"]).dt.normalize().unique()
    assert len(demand_dates) > 0, "无需求日期"

    # 验证仿真跨度（应覆盖 3 个月）
    assert len(sim_dates) >= 60, f"订单日期数 {len(sim_dates)} 不足 2 个月"

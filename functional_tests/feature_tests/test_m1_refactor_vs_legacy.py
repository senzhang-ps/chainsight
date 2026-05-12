from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from functional_tests.utils.m1_fixtures import (
    CUT_LOG_COLUMNS,
    FUNCTIONAL_TEST_TMP,
    ORDER_LOG_COLUMNS,
    SHIPMENT_LOG_COLUMNS,
    SUPPLY_DEMAND_LOG_COLUMNS,
    M1TestOrchestrator,
    build_weekly_m1_config,
    normalize_for_compare,
)


def test_refactor_keeps_legacy_visible_output_columns():
    from src.modules import demand_planning as legacy_m1
    from src.modules import demand_planning_refactor as refactor_m1

    config = build_weekly_m1_config()
    orchestrator = M1TestOrchestrator()
    FUNCTIONAL_TEST_TMP.mkdir(parents=True, exist_ok=True)
    old_dir = FUNCTIONAL_TEST_TMP / "legacy_m1_output"
    new_dir = FUNCTIONAL_TEST_TMP / "refactor_m1_output"
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(7)
    old_result = legacy_m1.run_daily_order_generation(
        config,
        pd.Timestamp("2026-05-04"),
        str(old_dir),
        orchestrator=orchestrator,
        skip_file_output=True,
    )

    np.random.seed(7)
    new_result = refactor_m1.run_daily_order_generation(
        config,
        pd.Timestamp("2026-05-04"),
        str(new_dir),
        orchestrator=orchestrator,
        skip_file_output=True,
    )

    assert list(new_result["orders_df"].columns) == list(old_result["orders_df"].columns)
    assert list(normalize_for_compare(new_result["shipment_df"], SHIPMENT_LOG_COLUMNS).columns) == SHIPMENT_LOG_COLUMNS
    assert list(normalize_for_compare(new_result["cut_df"], CUT_LOG_COLUMNS).columns) == CUT_LOG_COLUMNS
    assert list(normalize_for_compare(new_result["supply_demand_df"], SUPPLY_DEMAND_LOG_COLUMNS).columns) == SUPPLY_DEMAND_LOG_COLUMNS

    new_orders = normalize_for_compare(new_result["orders_df"], ORDER_LOG_COLUMNS)
    assert set(new_orders["simulation_date"]) == {pd.Timestamp("2026-05-04")}
    assert int(new_orders["quantity"].sum()) == 35
    assert int(new_orders[new_orders["demand_type"] == "AO"]["quantity"].sum()) == 7
    assert int(new_orders[new_orders["demand_type"] == "normal"]["quantity"].sum()) == 28


@pytest.mark.parametrize(
    "missing_key",
    ["M1_DemandForecast", "M1_OrderCalendar", "M1_AOConfig", "M1_ForecastError"],
)
def test_refactor_raises_value_error_on_missing_required_config(missing_key):
    """文档 §2 要求：缺任一必填表必须抛 ValueError，不可被静默吞掉。"""
    from src.modules import demand_planning_refactor as refactor_m1

    config = build_weekly_m1_config()
    config[missing_key] = pd.DataFrame()
    orchestrator = M1TestOrchestrator()
    FUNCTIONAL_TEST_TMP.mkdir(parents=True, exist_ok=True)
    out_dir = FUNCTIONAL_TEST_TMP / "refactor_m1_validate"
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match=missing_key):
        refactor_m1.run_daily_order_generation(
            config,
            pd.Timestamp("2026-05-04"),
            str(out_dir),
            orchestrator=orchestrator,
            skip_file_output=True,
        )


def test_refactor_summary_date_equals_simulation_date():
    """Fix #2: Summary.Date 取 simulation_date，而非累积订单池的首行 date。

    AO 订单的 date = simulation_date + advance_days，若 Summary.Date 用 iloc[0]
    就会取到 AO 的需求日期（这里 advance_days=2 → 2026-05-06），与下单日 5-04 不符。
    """
    from src.modules import demand_planning_refactor as refactor_m1

    config = build_weekly_m1_config()
    orchestrator = M1TestOrchestrator()
    FUNCTIONAL_TEST_TMP.mkdir(parents=True, exist_ok=True)
    out_dir = FUNCTIONAL_TEST_TMP / "refactor_m1_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_date = pd.Timestamp("2026-05-04")
    np.random.seed(7)
    result = refactor_m1.run_daily_order_generation(
        config,
        sim_date,
        str(out_dir),
        orchestrator=orchestrator,
        skip_file_output=True,
    )

    summary = result["summary_df"]
    assert not summary.empty
    assert pd.Timestamp(summary.iloc[0]["Date"]) == sim_date

    # 同时确认订单池里确实存在 date != sim_date 的 AO 行（即旧实现会取错的那种数据）
    orders = result["orders_df"]
    ao_rows = orders[orders["demand_type"] == "AO"]
    assert not ao_rows.empty
    assert (pd.to_datetime(ao_rows["date"]) != sim_date).any()


def test_apply_orders_consumption_normalizes_date_dtypes():
    """Fix #3: forecast 与 orders 的 date dtype 不一致时，idx_map 仍能命中。

    修复前：consumed['date'] 与订单端 pd.to_datetime(r.date) 类型不一致时
    （例如 forecast 用 datetime64[ns] 而订单 date 是 python date / 字符串），
    key 静默失配，消耗为 0；修复后两边显式归一化为 normalize 后的 Timestamp。
    """
    from src.modules.demand_planning_refactor.integration import _apply_orders_consumption

    forecast = pd.DataFrame(
        {
            "material": ["1001", "1001", "1001"],
            "location": ["0001", "0001", "0001"],
            "date": pd.to_datetime(["2026-05-05", "2026-05-06", "2026-05-07"]),
            "quantity": [100, 100, 100],
        }
    )
    # 故意混用 dtype：字符串 + 带时分秒，覆盖修复目标
    orders = pd.DataFrame(
        [
            {
                "material": "1001",
                "location": "0001",
                "date": "2026-05-06",
                "demand_type": "normal",
                "quantity": 30,
                "simulation_date": pd.Timestamp("2026-05-06"),
                "advance_days": 0,
            },
            {
                "material": "1001",
                "location": "0001",
                "date": pd.Timestamp("2026-05-07 09:30:00"),
                "demand_type": "AO",
                "quantity": 25,
                "simulation_date": pd.Timestamp("2026-05-05"),
                "advance_days": 2,
            },
        ]
    )

    consumed = _apply_orders_consumption(forecast, orders)
    consumed = consumed.set_index("date")["quantity"].astype(int).to_dict()

    # 5-06 normal 消耗 30 → 70
    assert consumed[pd.Timestamp("2026-05-06")] == 70
    # 5-07 AO 消耗 25（offset=0 命中）→ 75
    assert consumed[pd.Timestamp("2026-05-07")] == 75
    # 5-05 未被命中
    assert consumed[pd.Timestamp("2026-05-05")] == 100

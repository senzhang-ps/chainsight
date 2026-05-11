from __future__ import annotations

import numpy as np
import pandas as pd

from functional_tests.utils.m1_fixtures import build_daily_forecast, build_weekly_m1_config


def _timestamp(year: int, month: int, day: int) -> pd.Timestamp:
    value = pd.Timestamp(year=year, month=month, day=day)
    if not isinstance(value, pd.Timestamp):
        raise ValueError(f"Invalid timestamp: {year}-{month}-{day}")
    return value


WEEK1_ORDER_DAY = _timestamp(2026, 5, 4)
WEEK1_NON_ORDER_DAY = _timestamp(2026, 5, 5)
WEEK1_SECOND_ORDER_DAY = _timestamp(2026, 5, 6)
WEEK2_ORDER_DAY = _timestamp(2026, 5, 11)
WEEK2_SECOND_ORDER_DAY = _timestamp(2026, 5, 13)


def test_weekly_orders_split_only_to_order_calendar_days_and_apply_advance_days_last():
    from src.modules.demand_planning_refactor.order import generate_daily_orders

    config = build_weekly_m1_config()
    weekly_forecast = config["M1_DemandForecast"]
    daily_forecast = build_daily_forecast()

    np.random.seed(7)
    orders_df, _ = generate_daily_orders(
        WEEK1_ORDER_DAY,
        weekly_forecast,
        daily_forecast,
        config["M1_AOConfig"],
        config["M1_OrderCalendar"],
        config["M1_ForecastError"],
    )

    assert list(orders_df.columns) == [
        "date",
        "material",
        "location",
        "demand_type",
        "simulation_date",
        "advance_days",
        "quantity",
    ]
    assert set(pd.to_datetime(orders_df["simulation_date"])) == {WEEK1_ORDER_DAY}

    ao = orders_df[orders_df["demand_type"] == "AO"].iloc[0]
    normal = orders_df[orders_df["demand_type"] == "normal"].iloc[0]

    assert int(ao["quantity"]) == 7
    assert pd.Timestamp(ao["date"]) == WEEK1_SECOND_ORDER_DAY
    assert pd.Timestamp(ao["simulation_date"]) == WEEK1_ORDER_DAY
    assert int(ao["advance_days"]) == 2

    assert int(normal["quantity"]) == 28
    assert pd.Timestamp(normal["date"]) == WEEK1_ORDER_DAY
    assert pd.Timestamp(normal["simulation_date"]) == WEEK1_ORDER_DAY
    assert int(normal["advance_days"]) == 0


def test_weekly_orders_do_not_generate_on_non_calendar_day():
    from src.modules.demand_planning_refactor.order import generate_daily_orders

    config = build_weekly_m1_config()

    orders_df, consumed_forecast = generate_daily_orders(
        WEEK1_NON_ORDER_DAY,
        config["M1_DemandForecast"],
        build_daily_forecast(),
        config["M1_AOConfig"],
        config["M1_OrderCalendar"],
        config["M1_ForecastError"],
    )

    assert orders_df.empty
    assert not consumed_forecast.empty


def test_weekly_order_quantity_is_conserved_across_calendar_days():
    from src.modules.demand_planning_refactor.order import generate_daily_orders

    config = build_weekly_m1_config()
    daily_forecast = build_daily_forecast()
    all_orders = []

    for sim_date in [WEEK1_ORDER_DAY, WEEK1_SECOND_ORDER_DAY]:
        np.random.seed(7)
        orders_df, _ = generate_daily_orders(
            sim_date,
            config["M1_DemandForecast"],
            daily_forecast,
            config["M1_AOConfig"],
            config["M1_OrderCalendar"],
            config["M1_ForecastError"],
        )
        all_orders.append(orders_df)

    combined = pd.concat(all_orders, ignore_index=True)

    assert int(combined[combined["demand_type"] == "AO"]["quantity"].sum()) == 14
    assert int(combined[combined["demand_type"] == "normal"]["quantity"].sum()) == 56
    assert int(combined["quantity"].sum()) == 70


def test_weekly_total_cov_fallback_and_component_cov_are_reconciled():
    from src.modules.demand_planning_refactor.order import generate_weekly_orders

    config = build_weekly_m1_config()
    config["M1_ForecastError"].loc[
        config["M1_ForecastError"]["order_type"] == "normal", "error_std_percent"
    ] = 0.5
    config["M1_ForecastError"].loc[
        config["M1_ForecastError"]["order_type"] == "AO", "error_std_percent"
    ] = 0.0

    weekly_orders = generate_weekly_orders(
        config["M1_DemandForecast"],
        config["M1_AOConfig"],
        config["M1_ForecastError"],
    )

    total_qty = int(weekly_orders["quantity"].sum())
    ao_qty = int(weekly_orders[weekly_orders["demand_type"] == "AO"]["quantity"].sum())
    normal_qty = int(weekly_orders[weekly_orders["demand_type"] == "normal"]["quantity"].sum())

    assert total_qty == 83
    assert ao_qty == 11
    assert normal_qty == 72
    assert ao_qty + normal_qty == total_qty


def test_daily_forecast_with_week_column_keeps_daily_generation_path():
    from src.modules.demand_planning_refactor.order import generate_daily_orders

    config = build_weekly_m1_config()
    daily_forecast = build_daily_forecast()

    orders_df, _ = generate_daily_orders(
        WEEK1_ORDER_DAY,
        daily_forecast,
        daily_forecast,
        config["M1_AOConfig"],
        config["M1_OrderCalendar"],
        config["M1_ForecastError"],
    )

    assert int(orders_df["quantity"].sum()) == 10
    assert int(orders_df[orders_df["demand_type"] == "AO"]["quantity"].sum()) == 2
    assert int(orders_df[orders_df["demand_type"] == "normal"]["quantity"].sum()) == 8


def test_cross_week_calendar_uses_daily_reference_week_boundaries():
    from src.modules.demand_planning_refactor.order import generate_daily_orders

    config = build_weekly_m1_config()
    weekly_forecast = pd.DataFrame(
        [
            {"material": "1001", "location": "0001", "week": 1, "quantity": 70},
            {"material": "1001", "location": "0001", "week": 2, "quantity": 140},
        ]
    )
    first_week_daily = build_daily_forecast("2026-05-04")
    second_week_daily = build_daily_forecast("2026-05-11")
    second_week_daily["week"] = 2
    second_week_daily["quantity"] = 20
    second_week_daily["original_quantity"] = 20
    daily_reference = pd.concat([first_week_daily, second_week_daily], ignore_index=True)
    order_calendar = pd.DataFrame(
        [
            {"date": WEEK1_ORDER_DAY},
            {"date": WEEK1_SECOND_ORDER_DAY},
            {"date": WEEK2_ORDER_DAY},
            {"date": WEEK2_SECOND_ORDER_DAY},
        ]
    )

    orders_df, _ = generate_daily_orders(
        WEEK2_ORDER_DAY,
        weekly_forecast,
        daily_reference,
        config["M1_AOConfig"],
        order_calendar,
        config["M1_ForecastError"],
    )

    ao = orders_df[orders_df["demand_type"] == "AO"].iloc[0]
    normal = orders_df[orders_df["demand_type"] == "normal"].iloc[0]

    assert int(orders_df["quantity"].sum()) == 70
    assert int(ao["quantity"]) == 14
    assert pd.Timestamp(ao["simulation_date"]) == WEEK2_ORDER_DAY
    assert pd.Timestamp(ao["date"]) == WEEK2_SECOND_ORDER_DAY
    assert int(normal["quantity"]) == 56
    assert pd.Timestamp(normal["simulation_date"]) == WEEK2_ORDER_DAY
    assert pd.Timestamp(normal["date"]) == WEEK2_ORDER_DAY


def test_weekly_total_cov_prefers_total_forecast_error_in_mixed_config():
    from src.modules.demand_planning_refactor.order import generate_weekly_orders

    config = build_weekly_m1_config()
    mixed_forecast_error = pd.DataFrame(
        [
            {
                "material": "1001",
                "location": "0001",
                "order_type": "AO",
                "error_std_percent": 0.9,
            },
            {
                "material": "1001",
                "location": "0001",
                "order_type": "normal",
                "error_std_percent": 0.5,
            },
            {
                "material": "1001",
                "location": "0001",
                "order_type": "total",
                "error_std_percent": 0.0,
            },
        ]
    )

    weekly_orders = generate_weekly_orders(
        config["M1_DemandForecast"],
        config["M1_AOConfig"],
        mixed_forecast_error,
    )

    assert int(weekly_orders["quantity"].sum()) == 70
    assert int(weekly_orders[weekly_orders["demand_type"] == "AO"]["quantity"].sum()) == 21
    assert int(weekly_orders[weekly_orders["demand_type"] == "normal"]["quantity"].sum()) == 49


def test_weekly_total_cov_reconciles_ao_normal_cov_structure():
    from src.modules.demand_planning_refactor.order import generate_weekly_orders

    config = build_weekly_m1_config()
    mixed_forecast_error = pd.DataFrame(
        [
            {
                "material": "1001",
                "location": "0001",
                "order_type": "AO",
                "error_std_percent": 0.0,
            },
            {
                "material": "1001",
                "location": "0001",
                "order_type": "normal",
                "error_std_percent": 0.5,
            },
            {
                "material": "1001",
                "location": "0001",
                "order_type": "total",
                "error_std_percent": 0.0,
            },
        ]
    )

    weekly_orders = generate_weekly_orders(
        config["M1_DemandForecast"],
        config["M1_AOConfig"],
        mixed_forecast_error,
    )

    assert int(weekly_orders["quantity"].sum()) == 70
    assert int(weekly_orders[weekly_orders["demand_type"] == "AO"]["quantity"].sum()) == 9
    assert int(weekly_orders[weekly_orders["demand_type"] == "normal"]["quantity"].sum()) == 61


def test_weekly_total_cov_fallback_is_per_material_location():
    from src.modules.demand_planning_refactor.order import generate_weekly_orders

    config = build_weekly_m1_config()
    weekly_forecast = pd.DataFrame(
        [
            {"material": "1001", "location": "0001", "week": 1, "quantity": 70},
            {"material": "1003", "location": "0001", "week": 1, "quantity": 100},
        ]
    )
    mixed_forecast_error = pd.DataFrame(
        [
            {
                "material": "1001",
                "location": "0001",
                "order_type": "total",
                "error_std_percent": 0.0,
            },
            {
                "material": "1003",
                "location": "0001",
                "order_type": "normal",
                "error_std_percent": 0.5,
            },
        ]
    )

    weekly_orders = generate_weekly_orders(
        weekly_forecast,
        config["M1_AOConfig"],
        mixed_forecast_error,
    )

    material_1001 = weekly_orders[weekly_orders["material"] == "1001"]
    material_1003 = weekly_orders[weekly_orders["material"] == "1003"]

    assert int(material_1001["quantity"].sum()) == 70
    assert int(material_1003["quantity"].sum()) == 144

"""日末 StateContext 视图刷新契约。"""
from __future__ import annotations

import pandas as pd

from src.modules.state_context import StateContext


def test_day_end_refreshes_views_after_module_state_changes() -> None:
    """持久化前的 ``views`` 必须包含当天 M4 新增的生产 backlog。"""
    ctx = StateContext("2025-12-16")
    ctx.day_start("2025-12-16")
    assert ctx.views["production_plan_backlog"].empty

    ctx.production_plan_backlog = [{
        "material": "21143102",
        "location": "0386",
        "available_date": pd.Timestamp("2025-12-19"),
        "quantity": 950,
    }]
    ctx.day_end("2025-12-16")

    actual = ctx.views["production_plan_backlog"]
    assert len(actual) == 1
    assert actual.iloc[0][["material", "location", "quantity"]].tolist() == [
        "21143102", "0386", 950,
    ]


def test_backlog_view_retains_legacy_plan_and_arrived_quantity() -> None:
    """legacy backlog 在后续快照中仍保留计划和已到货生产量。"""
    ctx = StateContext("2025-12-19")
    ctx.production_plan_backlog = [{
        "material": "21143102",
        "location": "0386",
        "available_date": pd.Timestamp("2025-12-19"),
        "quantity": 950,
    }]
    arrival = {
        "date": pd.Timestamp("2025-12-19"),
        "material": "21143102",
        "location": "0386",
        "quantity": 950,
    }
    ctx.production_gr = [arrival]
    ctx.production_gr_by_date["2025-12-19"] = [arrival]

    backlog = ctx.get_production_plan_backlog_view("2025-12-20")

    assert backlog.iloc[0]["quantity"] == 1900
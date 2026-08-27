"""日末 StateContext 视图刷新契约。"""

# 测试文件说明
# 测试目的：集中验证日末状态视图与汇总数据的正确结转。
# 测试方法：按 `unit` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保日末状态视图与汇总数据的正确结转变更时能够快速定位回归影响。



from __future__ import annotations

import pandas as pd

from src.modules.state_context import StateContext


def test_day_end_refreshes_views_after_module_state_changes() -> None:
    # 测试目的：验证“day、end、refreshes、views、after、module、state、changes”场景下日末状态视图与汇总数据的正确结转。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `StateContext()`，再通过 3 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止日末状态视图与汇总数据的正确结转在重构、引擎切换或跨日运行时发生静默偏差。
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
    # 测试目的：验证“backlog、view、retains、legacy、plan、and、arrived、quantity”场景下日末状态视图与汇总数据的正确结转。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `StateContext()`，再通过 1 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止日末状态视图与汇总数据的正确结转在重构、引擎切换或跨日运行时发生静默偏差。
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
"""ModuleFive 分配例程的严格行为回归测试。

这些用例覆盖向量化前的逐节点实现所采用的优先级、比例向下取整及
三池连续扣减口径；不依赖数据库或完整仿真。
"""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from src.modules.deployment_planning.integration_refactor import ModuleFive


def test_allocate_priority_preserves_priority_order_and_floor_rounding():
    demand = pd.DataFrame(
        [
            {"material": "A", "node": "N1", "priority": 2, "planned_qty": 5},
            {"material": "A", "node": "N1", "priority": 1, "planned_qty": 4},
            {"material": "A", "node": "N1", "priority": 2, "planned_qty": 3},
            {"material": "B", "node": "N2", "priority": 1, "planned_qty": 4},
            {"material": "C", "node": "N3", "priority": 1, "planned_qty": 2},
        ]
    )
    stock = pd.DataFrame(
        [
            {"material": "A", "node": "N1", "qty": 7},
            {"material": "B", "node": "N2", "qty": 1},
        ]
    )

    result = ModuleFive._allocate_priority(demand, stock)

    pdt.assert_series_equal(
        result["deployed_qty_invCon"],
        pd.Series([1, 4, 1, 1, 0], name="deployed_qty_invCon", dtype="int64"),
    )
    assert result.index.equals(demand.index)
    assert pd.api.types.is_integer_dtype(result["deployed_qty_invCon"])


def test_allocate_pipeline_preserves_three_pool_deduction_and_self_demand_scope():
    demand = pd.DataFrame(
        [
            {
                "material": "A", "node": "N1", "receiving": "N1", "priority": 2,
                "row_id": 2, "planned_qty": 10, "deployed_qty_invCon": 2,
            },
            {
                "material": "A", "node": "N1", "receiving": "N1", "priority": 1,
                "row_id": 1, "planned_qty": 3, "deployed_qty_invCon": 0,
            },
            {
                "material": "A", "node": "N1", "receiving": "N2", "priority": 1,
                "row_id": 3, "planned_qty": 5, "deployed_qty_invCon": 0,
            },
            {
                "material": "B", "node": "N2", "receiving": "N2", "priority": 1,
                "row_id": 4, "planned_qty": 4, "deployed_qty_invCon": 0,
            },
        ]
    )
    pools = pd.DataFrame(
        [
            {
                "material": "A", "node": "N1", "future_intransit": "5",
                "open_inbound": "4", "future_production": "9",
            },
            {
                "material": "B", "node": "N2", "future_intransit": None,
                "open_inbound": None, "future_production": "2",
            },
        ]
    )

    result = ModuleFive._allocate_pipeline(demand, pools)

    expected = pd.DataFrame(
        {
            "deploy_from_in_transit": [3, 1, 0, 0],
            "deploy_from_open_deployment_inbound": [0, 0, 0, 0],
            "deploy_from_future_production": [5, 2, 0, 2],
            "deploy_qty_with_plan_order": [8, 3, 0, 2],
        },
        dtype="int64",
    )
    pdt.assert_frame_equal(result.loc[:, expected.columns], expected)
    assert result.index.equals(demand.index)

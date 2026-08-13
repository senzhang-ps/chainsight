"""ModuleFive 分配例程的严格行为回归测试。

这些用例覆盖向量化前的逐节点实现所采用的优先级、比例向下取整及
三池连续扣减口径；不依赖数据库或完整仿真。
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pandas.testing as pdt

from src.modules.deployment_planning.backends import _PandasBackend, _PolarsBackend


class _Owner:
    def __init__(self):
        self.layer_map = {("A", "PL01"): 0, ("A", "DC01"): 1}


def test_polars_normalise_preserves_object_timestamp_dates():
    """真实配置中 object dtype 的 pandas Timestamp 不能在 Polars 端变成 null。"""
    frame = pd.DataFrame({
        "date": pd.Series([pd.Timestamp("2025-12-15")], dtype=object),
        "material": ["A"],
    })

    result = _PolarsBackend._normalise(frame)

    assert result.get_column("date").to_list() == [date(2025, 12, 15)]


def test_polars_route_parameters_match_pandas_plant_and_root_leadtime_rules():
    """Plant/root 的 MCT/PTF/LSK 提前期必须与 pandas 基线一致。"""
    active = pd.DataFrame([
        {"material": "A", "node": "DC01", "upstream": "PL01", "location_type": "DC"},
    ])
    config = {
        "LeadTime": pd.DataFrame([
            {"sending": "PL01", "receiving": "DC01", "PDT": 2, "GR": 1, "MCT": 5},
        ]),
        "MaterialLocation": pd.DataFrame([
            {"material": "A", "location": "PL01", "ptf": 2, "lsk": 3},
        ]),
        "DeployConfig": pd.DataFrame([
            {"material": "A", "sending": "PL01", "receiving": "DC01", "moq": 4, "rv": 2},
        ]),
    }
    pandas_backend = _PandasBackend(_Owner())
    polars_backend = _PolarsBackend(_Owner())

    expected = pandas_backend.route_parameters(active, config)
    actual = polars_backend.route_parameters(
        polars_backend._normalise(active),
        {name: polars_backend._normalise(frame) for name, frame in config.items()},
    ).to_pandas()

    pdt.assert_frame_equal(
        actual.sort_values(list(actual.columns)).reset_index(drop=True),
        expected.sort_values(list(expected.columns)).reset_index(drop=True),
        check_dtype=False,
    )


def test_polars_space_constraints_match_pandas_priority_proportional_allocation():
    """同优先级受限空间必须按 pandas 的比例向下取整，而非按行耗尽。"""
    plan = pd.DataFrame([
        {"date": "2025-01-02", "material": "A", "sending": "PL01", "receiving": "DC01", "demand_element": "normal", "demand_qty": 5, "deployed_qty_invCon": 5},
        {"date": "2025-01-02", "material": "B", "sending": "PL02", "receiving": "DC01", "demand_element": "normal", "demand_qty": 5, "deployed_qty_invCon": 5},
        {"date": "2025-01-02", "material": "C", "sending": "PL03", "receiving": "DC01", "demand_element": "low", "demand_qty": 4, "deployed_qty_invCon": 4},
    ])
    space = pd.DataFrame([{"receiving": "DC01", "date": "2025-01-02", "max_qty": 7}])
    priority = {"normal": 1, "low": 2}

    pandas_backend = _PandasBackend()
    expected_plan, expected_gap = pandas_backend.apply_space(
        pandas_backend._normalise(plan), pandas_backend._normalise(space), priority
    )
    polars_backend = _PolarsBackend()
    actual_plan, actual_gap = polars_backend.apply_space(
        polars_backend._normalise(plan), polars_backend._normalise(space), priority
    )

    for actual, expected in ((actual_plan.to_pandas(), expected_plan), (actual_gap.to_pandas(), expected_gap)):
        pdt.assert_frame_equal(
            actual.sort_index(axis=1).sort_values(list(actual.columns), kind="mergesort").reset_index(drop=True),
            expected.sort_index(axis=1).sort_values(list(expected.columns), kind="mergesort").reset_index(drop=True),
            check_dtype=False,
        )


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

    result = _PandasBackend.allocate_priority(demand, stock)

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

    result = _PandasBackend.allocate_pipeline(demand, pools)

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


def test_polars_allocate_priority_matches_pandas_baseline():
    """Polars 的优先级窗口累计与比例向下取整不得偏离 pandas 基线。"""
    demand = pd.DataFrame(
        [
            {"material": "A", "node": "N1", "priority": 2, "planned_qty": 5},
            {"material": "A", "node": "N1", "priority": 1, "planned_qty": 4},
            {"material": "A", "node": "N1", "priority": 2, "planned_qty": 3},
            {"material": "B", "node": "N2", "priority": 1, "planned_qty": 4},
        ]
    )
    stock = pd.DataFrame(
        [
            {"material": "A", "node": "N1", "qty": 7},
            {"material": "B", "node": "N2", "qty": 1},
        ]
    )

    pdt.assert_frame_equal(
        _PolarsBackend.allocate_priority(demand, stock),
        _PandasBackend.allocate_priority(demand, stock),
        check_dtype=False,
    )


def test_polars_allocate_pipeline_matches_pandas_baseline():
    """Polars 的三池连续扣减必须严格复现 pandas 的历史口径。"""
    demand = pd.DataFrame(
        [
            {"material": "A", "node": "N1", "receiving": "N1", "priority": 2, "row_id": 2, "planned_qty": 10, "deployed_qty_invCon": 2},
            {"material": "A", "node": "N1", "receiving": "N1", "priority": 1, "row_id": 1, "planned_qty": 3, "deployed_qty_invCon": 0},
            {"material": "A", "node": "N1", "receiving": "N2", "priority": 1, "row_id": 3, "planned_qty": 5, "deployed_qty_invCon": 0},
            {"material": "B", "node": "N2", "receiving": "N2", "priority": 1, "row_id": 4, "planned_qty": 4, "deployed_qty_invCon": 0},
        ]
    )
    pools = pd.DataFrame(
        [
            {"material": "A", "node": "N1", "future_intransit": "5", "open_inbound": "4", "future_production": "9"},
            {"material": "B", "node": "N2", "future_intransit": None, "open_inbound": None, "future_production": "2"},
        ]
    )

    pdt.assert_frame_equal(
        _PolarsBackend.allocate_pipeline(demand, pools),
        _PandasBackend.allocate_pipeline(demand, pools),
        check_dtype=False,
    )

"""ModuleFive 分配例程的严格行为回归测试。

这些用例覆盖向量化前的逐节点实现所采用的优先级、比例向下取整及
三池连续扣减口径；不依赖数据库或完整仿真。
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pandas.testing as pdt

from src.modules.deployment_planning.backends import _PandasBackend, _PolarsBackend
from src.modules.deployment_planning.cache_utils import build_lead_time_cache
from src.modules.deployment_planning.data_loader import _load_static_config, _static_config_cache
from src.modules.deployment_planning.horizon_batch_calculator import build_horizon_cache


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


def test_lead_time_cache_accepts_lowercase_excel_columns():
    """legacy 读取 Excel/DB 配置时不得将小写 PDT/GR/MCT 静默视为零。"""
    cache = build_lead_time_cache(pd.DataFrame([
        {"sending": "PL01", "receiving": "DC01", "pdt": 2, "gr": 1, "mct": 5},
    ]))

    assert cache[("PL01", "DC01")] == (2, 1, 5)


def test_root_horizon_cache_accepts_lowercase_excel_columns():
    """根节点没有 self-route 时，horizon cache 也必须读取小写提前期字段。"""
    network = pd.DataFrame([
        {"material": "A", "location": "PL01", "sourcing": "", "location_type": "Plant",
         "eff_from": "2025-01-01", "eff_to": "2040-12-31"},
    ])
    lead = pd.DataFrame([
        {"sending": "PL01", "receiving": "DC01", "pdt": 2, "gr": 1, "mct": 5},
    ])
    cache = build_horizon_cache(
        all_pairs={("A", "PL01")}, sim_date=pd.Timestamp("2025-01-01"),
        network_df=network.assign(eff_from=pd.to_datetime(network.eff_from), eff_to=pd.to_datetime(network.eff_to)),
        leadtime_df=lead, m4_mlcfg_df=pd.DataFrame(), ptf_lsk_cache={},
        lead_time_cache=build_lead_time_cache(lead), active_network_cache={},
        location_layer_map={("A", "PL01"): 0},
    )

    assert cache[("A", "PL01")]["horizon"] == 5


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


def test_explicit_dc_root_does_not_receive_plant_ptf_lsk_leadtime():
    """Network 显式标记为 DC 的根节点必须优先使用其 DC 路线提前期。"""
    active = pd.DataFrame([
        {"material": "A", "node": "PL01", "upstream": "", "location_type": "DC"},
        {"material": "A", "node": "DC01", "upstream": "PL01", "location_type": "DC"},
    ])
    config = {
        "LeadTime": pd.DataFrame([
            {"sending": "PL01", "receiving": "DC01", "PDT": 2, "GR": 1, "MCT": 9},
        ]),
        "MaterialLocation": pd.DataFrame([
            {"material": "A", "location": "PL01", "ptf": 4, "lsk": 3},
        ]),
        "DeployConfig": pd.DataFrame([
            {"material": "A", "sending": "PL01", "receiving": "DC01", "moq": 1, "rv": 1},
        ]),
    }
    pandas_backend = _PandasBackend(_Owner())
    polars_backend = _PolarsBackend(_Owner())

    pandas_route = pandas_backend.route_parameters(active, config)
    polars_route = polars_backend.route_parameters(
        polars_backend._normalise(active),
        {name: polars_backend._normalise(frame) for name, frame in config.items()},
    ).to_pandas()

    assert pandas_route.loc[0, "leadtime"] == 3
    assert polars_route.loc[0, "leadtime"] == 3


def test_push_pull_static_config_normalises_excel_location_key():
    """PushPullModel 的数值 Excel 地点必须规范为 Network 使用的字符串键。"""
    data = {
        "M3_SafetyStock": pd.DataFrame(),
        "Global_Network": pd.DataFrame(),
        "Global_LeadTime": pd.DataFrame(),
        "Global_DemandPriority": pd.DataFrame(),
        "M5_PushPullModel": pd.DataFrame([
            {"material": "A", "sending": "386", "model": "push"},
        ]),
        "M5_DeployConfig": pd.DataFrame(),
        "M5_SupplyDemandLog": pd.DataFrame(),
        "M4_MaterialLocationLineCfg": pd.DataFrame(),
    }

    pandas_static = _PandasBackend(_Owner()).normalise_static_config(data)
    polars_static = _PolarsBackend(_Owner()).normalise_static_config(data)

    assert pandas_static["PushPullModel"].loc[0, "sending"] == "0386"
    assert polars_static["PushPullModel"].get_column("sending").item() == "0386"


def test_legacy_static_loader_normalises_push_pull_location_when_skipped():
    """legacy 的已规范化调用路径也不能遗漏 PushPullModel 的地点键。"""
    config_dict = {
        "M5_PushPullModel": pd.DataFrame([
            {"material": "A", "sending": 386, "model": "push"},
        ]),
    }
    config = {}
    _static_config_cache.clear()

    _load_static_config(config_dict, config, skip_normalize=True)

    assert config["PushPullModel"].loc[0, "sending"] == "0386"


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


def test_next_gap_preserves_individual_shortage_facts():
    """跨层 AO 缺口不能按路线聚合，否则会改变 legacy 的计划行粒度。"""
    demand = pd.DataFrame([
        {"material": "A", "node": "DC01", "receiving": "S1", "demand_element": "AO", "residual_qty": 1, "requirement_date": "2025-01-02", "orig_location": "S1"},
        {"material": "A", "node": "DC01", "receiving": "S1", "demand_element": "AO", "residual_qty": 2, "requirement_date": "2025-01-02", "orig_location": "S1"},
    ])
    active = pd.DataFrame([
        {"material": "A", "node": "DC01", "upstream": "PL01"},
    ])

    pandas_result = _PandasBackend.next_gap(demand, active)
    polars_result = _PolarsBackend().next_gap(
        _PolarsBackend._normalise(demand), _PolarsBackend._normalise(active),
    ).to_pandas()

    expected = pd.DataFrame({
        "material": ["A", "A"], "node": ["PL01", "PL01"],
        "receiving": ["DC01", "DC01"],
        "demand_element": ["net demand for AO", "net demand for AO"],
        "demand_qty": [1, 2],
        "requirement_date": [pd.Timestamp("2025-01-02")] * 2,
        "orig_location": ["S1", "S1"],
    })
    for result in (pandas_result, polars_result):
        result = result.copy()
        result["requirement_date"] = pd.to_datetime(result["requirement_date"])
        pdt.assert_frame_equal(result.reset_index(drop=True), expected, check_dtype=False)


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


def test_allocate_pipeline_preserves_legacy_float_operation_order():
    """49 单位池覆盖 5/4/4/4/3/3/26 时，三个 4 必须保留 1 单位 gap。

    legacy 先计算 ``gap / total`` 再乘 pool；若先乘再除，三个 4 会因中间
    浮点舍入变成精确 4，错误吞掉应向上游传播的三个单位。
    """
    demand = pd.DataFrame([
        {"material": "A", "node": "N1", "receiving": "N1", "planned_qty": qty, "deployed_qty_invCon": 0}
        for qty in (5, 4, 4, 4, 3, 3, 26)
    ])
    pools = pd.DataFrame([{
        "material": "A", "node": "N1", "future_intransit": 49,
        "open_inbound": 0, "future_production": 0,
    }])

    result = _PandasBackend.allocate_pipeline(demand, pools)

    assert result["deploy_from_in_transit"].tolist() == [5, 3, 3, 3, 3, 3, 26]
    assert result["deploy_qty_with_plan_order"].sum() == 46


def test_polars_allocate_pipeline_preserves_legacy_float_operation_order():
    """Polars 也必须保留 49 × (4 / 49) 的非精确中间结果。"""
    demand = pd.DataFrame([
        {"material": "A", "node": "N1", "receiving": "N1", "planned_qty": qty, "deployed_qty_invCon": 0}
        for qty in (5, 4, 4, 4, 3, 3, 26)
    ])
    pools = pd.DataFrame([{
        "material": "A", "node": "N1", "future_intransit": 49,
        "open_inbound": 0, "future_production": 0,
    }])

    result = _PolarsBackend.allocate_pipeline(demand, pools)

    assert result["deploy_from_in_transit"].tolist() == [5, 3, 3, 3, 3, 3, 26]
    assert result["deploy_qty_with_plan_order"].sum() == 46


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

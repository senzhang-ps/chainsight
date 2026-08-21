"""ModuleFive 的配置注入与模型注册契约测试。"""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from src.modules.deployment_planning.backends import _PandasBackend, _PolarsBackend
from src.models.module import (
    Module5OutputDeploymentplan,
    Module5OutputStockonhandlog,
    Module5OutputUnfulfilledlog,
    Module5OutputValidation,
    get_output_table_name,
)
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.state_context import StateContext


class _Orch:
    """最小新 Orch 替身：只允许 ModuleFive 使用 load_datas 注入配置。"""

    def __init__(self, config: dict[str, pd.DataFrame], engine: str = "pandas"):
        self._config = config
        self.engine = engine
        self.load_count = 0

    def get_module_config(self, module_name: str) -> dict:
        return {}

    def load_datas(self, module) -> None:
        self.load_count += 1
        module.datas = {
            name: self._config.get(name, pd.DataFrame())
            for name in module.schema
        }


def _config() -> dict[str, pd.DataFrame]:
    return {
        "M3_SafetyStock": pd.DataFrame(
            columns=["material", "location", "date", "safety_stock_qty"]
        ),
        "Global_Network": pd.DataFrame([
            {
                "material": "100", "location": "DC01", "sourcing": "PL01",
                "location_type": "DC", "eff_from": "2025-01-01", "eff_to": "2025-12-31",
            }
        ]),
        "Global_LeadTime": pd.DataFrame([
            {"sending": "PL01", "receiving": "DC01", "PDT": 1, "GR": 0, "MCT": 0}
        ]),
        "Global_DemandPriority": pd.DataFrame([
            {"demand_element": "normal", "priority": 2}
        ]),
        "M5_PushPullModel": pd.DataFrame([
            {"material": "100", "sending": "PL01", "model": "pull"}
        ]),
        "M5_DeployConfig": pd.DataFrame([
            {"material": "100", "sending": "PL01", "moq": 1, "rv": 1}
        ]),
        "M5_SupplyDemandLog": pd.DataFrame(
            columns=["date", "material", "location", "demand_element", "quantity"]
        ),
        "M4_MaterialLocationLineCfg": pd.DataFrame(
            columns=["material", "location", "ptf", "lsk"]
        ),
    }


def test_module_five_prepare_uses_orch_data_injection_only():
    """prepare 只调用新 Orch 的 load_datas，且可安全重复调用。"""
    orch = _Orch(_config())
    module = ModuleFive(
        simulation_date="2025-01-01",
        simulation_start_date="2025-01-01",
        orch=orch,
    )

    module.prepare()
    module.prepare()

    assert orch.load_count == 1
    assert module.static_config["Network"].iloc[0]["location"] == "DC01"
    assert module.location_to_layer[("100", "DC01")] == 1


def test_module_five_selects_polars_allocation_backend_from_orch_engine():
    """M5 遵循 Orch.engine；pandas 仍为默认兼容后端。"""
    pandas_module = ModuleFive(
        simulation_date="2025-01-01",
        simulation_start_date="2025-01-01",
        orch=_Orch(_config()),
    )
    polars_module = ModuleFive(
        simulation_date="2025-01-01",
        simulation_start_date="2025-01-01",
        orch=_Orch(_config(), engine="polars"),
    )

    assert isinstance(pandas_module._backend, _PandasBackend)
    assert isinstance(polars_module._backend, _PolarsBackend)


def test_module_five_run_does_not_implicitly_prepare():
    """Orch 独立控制 prepare/run，run 只编排 backend 的计算步骤。"""
    module = ModuleFive(
        simulation_date="2025-01-01",
        simulation_start_date="2025-01-01",
        orch=_Orch(_config()),
    )
    module._prepared = True
    calls: list[str] = []
    empty = pd.DataFrame()
    supply = pd.DataFrame(columns=["material", "node", "qty"])

    def fail_prepare():
        raise AssertionError("run() 不得调用 prepare()")

    module.prepare = fail_prepare
    module._backend.daily_inputs = lambda day: calls.append("daily_inputs") or {"ReceivingSpace": empty}
    module._backend.active_network = lambda config, day: calls.append("active_network") or empty
    module._backend.validate = lambda config, active: calls.append("validate") or ({}, [])
    module._backend.route_parameters = lambda active, config: calls.append("route_parameters") or empty
    module._backend.supply_ledger = lambda config, day: calls.append("supply_ledger") or (supply, empty, supply, empty, empty)
    module._backend.plan_layers = lambda *args: calls.append("plan_layers") or (empty, empty, empty)
    module._backend.push = lambda *args: calls.append("push") or empty
    module._backend.apply_space = lambda plan, space, priority: calls.append("apply_space") or (plan, empty)
    module._backend.finalise_result = lambda *args: calls.append("finalise_result") or {"deployment_plan": empty}

    module.run()

    assert calls == [
        "daily_inputs", "active_network", "validate", "route_parameters",
        "supply_ledger", "plan_layers", "push", "apply_space", "finalise_result",
    ]
    assert module.output() is module._backend.result


def test_state_context_owns_module_five_dynamic_demand_inputs():
    """M1 到 M5 的动态事实由 StateContext 按日期保存，而非挂在 M5 实例上。"""
    context = StateContext(simulation_date="2025-01-01")
    supply_demand = pd.DataFrame([{"material": "100", "quantity": 12}])
    orders = pd.DataFrame([{"material": "100", "quantity": 3}])

    context.apply_deployment_demand_inputs(supply_demand, orders, "2025-01-01")
    supply_demand.loc[0, "quantity"] = 99

    stored_supply_demand = context.get_deployment_supply_demand_view("2025-01-01")
    stored_supply_demand.loc[0, "quantity"] = 88

    assert context.get_deployment_supply_demand_view("2025-01-01").loc[0, "quantity"] == 12
    assert context.get_deployment_order_log_view("2025-01-01").loc[0, "quantity"] == 3


def test_module_five_output_models_match_registered_tables():
    """四张 M5 输出表均有声明式模型，且不会改变既有 registry 名称。"""
    expected = {
        "deployment_plan": (Module5OutputDeploymentplan, "module5_output_deploymentplan"),
        "unfulfilled_log": (Module5OutputUnfulfilledlog, "module5_output_unfulfilledlog"),
        "stock_on_hand_log": (Module5OutputStockonhandlog, "module5_output_stockonhandlog"),
        "validation_log": (Module5OutputValidation, "module5_output_validation"),
    }
    for output_key, (model, table_name) in expected.items():
        assert model.__tablename__ == table_name
        assert get_output_table_name("module5", output_key) == table_name
        assert {"run_id", "sim_date", "config_name", "db_write_time"}.issubset(
            model.__table__.columns.keys()
        )


def test_pipeline_supply_pools_are_independent_in_both_engines():
    """Polars 不能将上一 pipeline 池的使用量扣减到下一独立池。"""
    demand = pd.DataFrame([
        {"material": "MAT-1", "node": "LOC-1", "receiving": "LOC-1", "planned_qty": 45, "deployed_qty_invCon": 0},
        {"material": "MAT-1", "node": "LOC-1", "receiving": "LOC-1", "planned_qty": 19, "deployed_qty_invCon": 0},
    ])
    pools = pd.DataFrame([{
        "material": "MAT-1", "node": "LOC-1",
        "future_intransit": 0, "open_inbound": 21, "future_production": 0,
    }])

    pandas_result = _PandasBackend.allocate_pipeline(demand, pools)
    polars_result = _PolarsBackend.allocate_pipeline(demand, pools)
    columns = [
        "planned_qty", "deploy_from_in_transit",
        "deploy_from_open_deployment_inbound", "deploy_from_future_production",
        "deploy_qty_with_plan_order",
    ]
    pdt.assert_frame_equal(
        pandas_result.loc[:, columns].reset_index(drop=True),
        polars_result.loc[:, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_priority_allocation_preserves_fully_covered_integer_demand():
    """库存足够时 Polars 不得因浮点 floor 少分一个单位。"""
    demand = pd.DataFrame([
        {"material": "MAT-1", "node": "LOC-1", "priority": 1, "planned_qty": 13},
        {"material": "MAT-1", "node": "LOC-1", "priority": 1, "planned_qty": 29},
    ])
    stock = pd.DataFrame([{"material": "MAT-1", "node": "LOC-1", "qty": 42}])

    pandas_result = _PandasBackend.allocate_priority(demand, stock)
    polars_result = _PolarsBackend.allocate_priority(demand, stock)

    assert pandas_result["deployed_qty_invCon"].tolist() == [13, 29]
    assert polars_result["deployed_qty_invCon"].tolist() == [13, 29]

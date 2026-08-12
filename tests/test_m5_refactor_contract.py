"""ModuleFive 的配置注入与模型注册契约测试。"""
from __future__ import annotations

import pandas as pd

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

    engine = "pandas"

    def __init__(self, config: dict[str, pd.DataFrame]):
        self._config = config
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

"""无数据库的 M5 Polars 全日规划回归。"""

# 测试文件说明
# 测试目的：集中验证部署计划、优先级分配与供给池扣减的一致性。
# 测试方法：按 `unit` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保部署计划、优先级分配与供给池扣减的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from src.modules.deployment_planning.integration_refactor import ModuleFive


class _Orch:
    def __init__(self, config: dict[str, pd.DataFrame], engine: str):
        self.config = config
        self.engine = engine

    def get_module_config(self, module_name: str) -> dict:
        return {}

    def load_datas(self, module) -> None:
        module.datas = {name: self.config.get(name, pd.DataFrame()) for name in module.schema}


class _State:
    """只实现 M5 当日读取的 StateContext 接口。"""

    def get_deployment_supply_demand_view(self, day):
        return pd.DataFrame()

    def get_deployment_order_log_view(self, day):
        return pd.DataFrame([
            {"material": "100", "location": "DC01", "date": day,
             "demand_type": "normal", "quantity": 6}
        ])

    def get_shipment_log_view(self, day):
        return pd.DataFrame(columns=["material", "location", "date", "quantity"])

    def get_beginning_inventory_view(self, day):
        return pd.DataFrame([{"material": "100", "location": "PL01", "quantity": 10}])

    def get_planning_intransit_view(self, day):
        return pd.DataFrame(columns=["material", "receiving", "actual_delivery_date", "quantity"])

    def get_delivery_gr_view(self, day):
        return pd.DataFrame(columns=["material", "receiving", "quantity"])

    def get_open_deployment_view(self, day):
        return pd.DataFrame(columns=["material", "sending", "receiving", "deployed_qty"])

    def get_deployment_production_view(self, day):
        return pd.DataFrame(columns=["material", "location", "available_date", "produced_qty"])

    def get_space_quota_view(self, day):
        return pd.DataFrame(columns=["receiving", "date", "max_qty"])


def _config() -> dict[str, pd.DataFrame]:
    return {
        "M3_SafetyStock": pd.DataFrame([
            {"material": "100", "location": "DC01", "date": "2025-01-03", "safety_stock_qty": 2},
        ]),
        "Global_Network": pd.DataFrame([
            {"material": "100", "location": "DC01", "sourcing": "PL01", "eff_from": "2025-01-01", "eff_to": "2025-12-31"}
        ]),
        "Global_LeadTime": pd.DataFrame([{"sending": "PL01", "receiving": "DC01", "PDT": 1, "GR": 0, "MCT": 1}]),
        "Global_DemandPriority": pd.DataFrame([{"demand_element": "normal", "priority": 2}]),
        "M5_PushPullModel": pd.DataFrame(columns=["material", "sending", "model"]),
        "M5_DeployConfig": pd.DataFrame([{"material": "100", "sending": "PL01", "receiving": "DC01", "moq": 1, "rv": 1}]),
        "M5_SupplyDemandLog": pd.DataFrame([
            {"material": "100", "location": "DC01", "date": "2025-01-02", "demand_element": "forecast", "quantity": 3},
        ]),
        "M4_MaterialLocationLineCfg": pd.DataFrame([{"material": "100", "location": "PL01", "ptf": 0, "lsk": 1}]),
    }


def _run(engine: str) -> dict:
    module = ModuleFive("2025-01-02", "2025-01-01", state_context=_State(), orch=_Orch(_config(), engine))
    module.prepare()
    # Static state is Polars-only for the Polars engine; output is still the public pandas API.
    assert module._backend.static["Network"].__class__.__module__.startswith("polars") if engine == "polars" else True
    module.run()
    return module.output()


def test_polars_plan_day_uses_static_and_fake_state_context_with_pandas_parity():
    # 测试目的：验证“polars、plan、day、uses、static、and、fake、state、context、with、pandas、parity”场景下部署计划、优先级分配与供给池扣减的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_run()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止部署计划、优先级分配与供给池扣减的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    pandas_result = _run("pandas")
    polars_result = _run("polars")

    for key in ("deployment_plan", "unfulfilled_log", "stock_on_hand_log", "validation_log"):
        left = pandas_result[key].sort_index(axis=1).sort_values(list(pandas_result[key].columns), kind="mergesort").reset_index(drop=True) if not pandas_result[key].empty else pandas_result[key]
        right = polars_result[key].sort_index(axis=1).sort_values(list(polars_result[key].columns), kind="mergesort").reset_index(drop=True) if not polars_result[key].empty else polars_result[key]
        pdt.assert_frame_equal(left, right, check_dtype=False)
    assert polars_result["statistics"] == pandas_result["statistics"]
    assert not polars_result["deployment_plan"].empty

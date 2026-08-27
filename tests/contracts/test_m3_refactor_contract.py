"""M3 refactor contracts: M5 PlanningFacts handoff and dual-engine parity."""

# 测试文件说明
# 测试目的：集中验证净需求计算及跨日计划事实传递的一致性。
# 测试方法：按 `contracts` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保净需求计算及跨日计划事实传递的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.mrp_planning.backends import _PandasBackend, _PolarsBackend
from src.modules.mrp_planning.integration_refactor import ModuleThree
from src.modules.state_context import StateContext


class _Orch:
    def __init__(self, config: dict[str, pd.DataFrame], engine: str = "pandas"):
        self.config = config
        self.engine = engine
        self.load_count = 0

    def get_module_config(self, module_name: str) -> dict:
        return {}

    def load_datas(self, module) -> None:
        self.load_count += 1
        module.datas = {name: self.config.get(name, pd.DataFrame()) for name in module.schema}


def _config() -> dict[str, pd.DataFrame]:
    return {
        "M3_SafetyStock": pd.DataFrame([
            {"material": "100", "location": "DC01", "date": "2025-01-03", "safety_stock_qty": 2},
        ]),
        "Global_Network": pd.DataFrame([
            {"material": "100", "location": "DC01", "sourcing": "PL01", "eff_from": "2025-01-01", "eff_to": "2025-12-31"},
        ]),
        "Global_LeadTime": pd.DataFrame([
            {"sending": "PL01", "receiving": "DC01", "PDT": 1, "GR": 0, "MCT": 1},
        ]),
        "Global_DemandPriority": pd.DataFrame([{"demand_element": "AO", "priority": 1}]),
        "M5_PushPullModel": pd.DataFrame(columns=["material", "sending", "model"]),
        "M5_DeployConfig": pd.DataFrame([
            {"material": "100", "sending": "PL01", "receiving": "DC01", "moq": 1, "rv": 1},
        ]),
        "M5_SupplyDemandLog": pd.DataFrame([
            {"material": "100", "location": "DC01", "date": "2025-01-02", "demand_element": "forecast", "quantity": 3},
        ]),
        "M4_MaterialLocationLineCfg": pd.DataFrame([
            {"material": "100", "location": "PL01", "ptf": 0, "lsk": 1},
        ]),
    }


def _context() -> StateContext:
    ctx = StateContext("2025-01-02")
    ctx.day_start("2025-01-02")
    ctx.unrestricted_inventory[("100", "PL01")] = 10
    ctx.apply_deployment_demand_inputs(
        pd.DataFrame(),
        pd.DataFrame([{"material": "100", "location": "DC01", "date": "2025-01-02", "demand_type": "AO", "quantity": 6}]),
        "2025-01-02",
    )
    return ctx


def _run(engine: str) -> tuple[StateContext, ModuleThree]:
    orch = _Orch(_config(), engine)
    context = _context()
    m5 = ModuleFive("2025-01-02", "2025-01-01", state_context=context, orch=orch)
    m3 = ModuleThree("2025-01-02", "2025-01-01", state_context=context, orch=orch)
    m5.prepare()
    m3.prepare()
    m5.run()
    m3.run()
    return context, m3


def test_module_three_selects_independent_backend():
    # 测试目的：验证“module、three、selects、independent、backend”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `isinstance()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    assert isinstance(ModuleThree("2025-01-02", "2025-01-01", orch=_Orch(_config()))._backend, _PandasBackend)
    assert isinstance(ModuleThree("2025-01-02", "2025-01-01", orch=_Orch(_config(), "polars"))._backend, _PolarsBackend)


def test_module_three_requires_m5_planning_facts():
    # 测试目的：验证“module、three、requires、m5、planning、facts”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_context()`，再验证调用完成后的可观察结果。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    ctx = _context()
    module = ModuleThree("2025-01-02", "2025-01-01", state_context=ctx, orch=_Orch(_config()))
    module.prepare()
    with pytest.raises(RuntimeError, match="缺少 PlanningFacts"):
        module.run()


def test_planning_facts_are_isolated_and_expire_next_day():
    # 测试目的：验证“planning、facts、are、isolated、and、expire、next、day”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_run()`，再通过 1 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    context, _ = _run("pandas")
    facts = context.get_planning_facts("2025-01-02")
    facts["routes"].loc[:, "moq"] = 999
    assert context.get_planning_facts("2025-01-02")["routes"]["moq"].max() == 1
    context.day_start("2025-01-03")
    with pytest.raises(RuntimeError, match="缺少 PlanningFacts"):
        context.get_planning_facts("2025-01-02")


def test_state_context_exposes_only_previous_day_m3_result():
    # 测试目的：验证“state、context、exposes、only、previous、day、m3、result”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_context()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    context = _context()
    net = pd.DataFrame([{"material": "100", "location": "PL01", "quantity": -4}])
    context.apply_m3_net_demand(net, "2025-01-02")
    net.loc[0, "quantity"] = -99
    previous = context.get_previous_m3_result("2025-01-03")
    previous["net_demand_df"].loc[0, "quantity"] = -88

    assert context.get_previous_m3_result("2025-01-03")["net_demand_df"].loc[0, "quantity"] == -4
    assert context.get_previous_m3_result("2025-01-02")["net_demand_df"].empty


def test_module_three_pandas_and_polars_have_strict_output_parity():
    # 测试目的：验证“module、three、pandas、and、polars、have、strict、output、parity”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_run()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    _, pandas_module = _run("pandas")
    _, polars_module = _run("polars")
    left = pandas_module.output()["net_demand_df"].sort_values(
        ["material", "location", "requirement_date", "demand_element", "layer"], kind="mergesort"
    ).reset_index(drop=True)
    right = polars_module.output()["net_demand_df"].sort_values(
        ["material", "location", "requirement_date", "demand_element", "layer"], kind="mergesort"
    ).reset_index(drop=True)
    pdt.assert_frame_equal(left, right, check_dtype=False)
    assert pandas_module.output()["net_demand_count"] == polars_module.output()["net_demand_count"]
    assert not left.empty
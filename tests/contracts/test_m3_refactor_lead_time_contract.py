"""M3 refactor 的内部提前期列 canonicalization 契约。"""

# 测试文件说明
# 测试目的：集中验证净需求计算及跨日计划事实传递的一致性。
# 测试方法：按 `contracts` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保净需求计算及跨日计划事实传递的一致性变更时能够快速定位回归影响。



from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.modules.mrp_planning.backends import _PandasBackend, _PolarsBackend
from src.modules.mrp_planning.lead_time import compute_root_horizon


_SCHEMA = {
    "Global_LeadTime": {},
    "M4_MaterialLocationLineCfg": {},
}


def _owner() -> SimpleNamespace:
    return SimpleNamespace(schema=_SCHEMA)


def _config() -> dict[str, pd.DataFrame]:
    return {
        "Global_LeadTime": pd.DataFrame([{
            "sending": "0386", "receiving": "A672",
            "pdt": 10, "gr": 2, "mct": 16,
        }]),
        "M4_MaterialLocationLineCfg": pd.DataFrame([{
            "material": "MAT-1", "location": "0386", "ptf": 2, "lsk": 2,
        }]),
    }


def test_pandas_m3_canonicalises_lowercase_lead_time_internally() -> None:
    # 测试目的：验证“pandas、m3、canonicalises、lowercase、lead、time、internally”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_PandasBackend()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    backend = _PandasBackend(_owner())
    static = backend.normalise_static_config(_config())

    assert {"PDT", "GR", "MCT"}.issubset(static["Global_LeadTime"].columns)
    assert compute_root_horizon(
        "MAT-1", "0386", static["Global_LeadTime"], static["M4_MaterialLocationLineCfg"],
    ) == 19


def test_polars_m3_canonicalises_lowercase_lead_time_internally() -> None:
    # 测试目的：验证“polars、m3、canonicalises、lowercase、lead、time、internally”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_PolarsBackend()`，再通过 1 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    backend = _PolarsBackend(_owner())
    static = backend.normalise_static_config(_config())

    assert {"PDT", "GR", "MCT"}.issubset(static["Global_LeadTime"].columns)
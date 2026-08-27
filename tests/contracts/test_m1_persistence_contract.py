"""M1 累计订单视图与日粒度 OrderLog 持久化契约。"""

# 测试文件说明
# 测试目的：集中验证订单生成、库存消耗与发运数据的一致性。
# 测试方法：按 `contracts` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保订单生成、库存消耗与发运数据的一致性变更时能够快速定位回归影响。



from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.core.orchestrator.persistence_manager import PersistenceManager
from tests.helpers.compare_utils import compare_dataframes_by_key


def test_m1_orderlog_persists_only_daily_orders(monkeypatch) -> None:
    # 测试目的：验证“m1、orderlog、persists、only、daily、orders”场景下订单生成、库存消耗与发运数据的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `DataFrame()`，再通过 3 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """累计 ``orders_df`` 不得在每个 ``sim_date`` 重复写入 OrderLog。"""
    cumulative = pd.DataFrame({
        "material": ["old", "new"],
        "location": ["0001", "0001"],
        "date": ["2025-12-16", "2025-12-17"],
        "quantity": [10, 20],
    })
    daily = cumulative.iloc[[1]].copy()
    module = SimpleNamespace(
        module_config="M1",
        output=lambda: {
            "orders_df": cumulative,
            "orders_to_persist": daily,
        },
    )
    manager = PersistenceManager(SimpleNamespace(db=object(), run_id="run", config_name="cfg"))
    captured: list[tuple[str, pd.DataFrame]] = []
    monkeypatch.setattr(
        PersistenceManager,
        "_inject_meta",
        lambda self, frame, run_id, sim_date, now: frame.assign(run_id=run_id, sim_date=sim_date),
    )
    monkeypatch.setattr(
        PersistenceManager,
        "_write_idempotent",
        lambda self, table_name, frame, run_id, sim_date: captured.append((table_name, frame.copy())),
    )

    manager.save_module_output(module, "2025-12-17")

    assert len(captured) == 1
    table_name, persisted = captured[0]
    assert table_name == "module1_output_orderlog"
    assert persisted["material"].tolist() == ["new"]


def test_m4_production_persists_blank_changeover_as_null(monkeypatch) -> None:
    # 测试目的：验证“m4、production、persists、blank、changeover、as、null”场景下订单生成、库存消耗与发运数据的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `SimpleNamespace()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """空换产标识必须与 legacy 的 SQL NULL 表示保持一致。"""
    module = SimpleNamespace(
        module_config="M4",
        output=lambda: {
            "production_df": pd.DataFrame({
                "material": ["MAT-1"], "location": ["0001"],
                "changeover_id": [""],
            }),
        },
    )
    manager = PersistenceManager(SimpleNamespace(db=object(), run_id="run", config_name="cfg"))
    captured: list[pd.DataFrame] = []
    monkeypatch.setattr(
        PersistenceManager,
        "_inject_meta",
        lambda self, frame, run_id, sim_date, now: frame,
    )
    monkeypatch.setattr(
        PersistenceManager,
        "_write_idempotent",
        lambda self, table_name, frame, run_id, sim_date: captured.append(frame.copy()),
    )

    manager.save_module_output(module, "2025-12-17")

    assert len(captured) == 1
    assert pd.isna(captured[0].loc[0, "changeover_id"])


def test_comparator_treats_blank_changeover_identifier_as_missing() -> None:
    # 测试目的：验证“comparator、treats、blank、changeover、identifier、as、missing”场景下订单生成、库存消耗与发运数据的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `DataFrame()`，再通过 3 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """审计关联中 SQL NULL、空串和数据库字符串 ``nan`` 表示同一缺失标识。"""
    left = pd.DataFrame({"material": ["MAT-1"], "changeover_id": [pd.NA], "quantity": [1]})
    right = pd.DataFrame({"material": ["MAT-1"], "changeover_id": ["nan"], "quantity": [1]})

    comparison = compare_dataframes_by_key(left, right)

    assert comparison["left_only_keys"] == 0
    assert comparison["right_only_keys"] == 0
    assert not comparison["column_differences"]
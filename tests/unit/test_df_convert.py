
# 测试文件说明
# 测试目的：集中验证pandas 与 Polars 数据类型转换的一致性。
# 测试方法：按 `unit` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保pandas 与 Polars 数据类型转换的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import pandas as pd

from src.utils.df_convert import pandas_to_polars


def test_pandas_nullable_numeric_converts_to_polars_numeric_with_null():
    # 测试目的：验证“pandas、nullable、numeric、converts、to、polars、numeric、with、null”场景下pandas 与 Polars 数据类型转换的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `DataFrame()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止pandas 与 Polars 数据类型转换的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """DB 读回的 pandas nullable 数值列不能被推断为 Polars Object。"""
    frame = pd.DataFrame({"PDT": pd.Series([2, pd.NA], dtype="Int64")})

    converted = pandas_to_polars(frame)

    assert converted.schema["PDT"].is_numeric()
    assert converted["PDT"].to_list() == [2, None]


def test_database_read_polars_returns_filtered_rows(db):
    # 测试目的：验证“database、read、polars、returns、filtered、rows”场景下pandas 与 Polars 数据类型转换的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `DataFrame()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止pandas 与 Polars 数据类型转换的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """直接 Polars 读取应保留 DB 过滤语义，供大快照恢复复用。"""
    frame = pd.DataFrame({
        "run_id": ["run-a", "run-b"],
        "sim_date": ["2025-12-15", "2025-12-15"],
        "quantity": [3, 5],
    })
    db.write_df("test_read_polars", frame)

    result = db.read_polars(
        "test_read_polars", run_id="run-a", sim_date="2025-12-15",
    )

    assert result["run_id"].to_list() == ["run-a"]
    assert result["quantity"].to_list() == ["3"]
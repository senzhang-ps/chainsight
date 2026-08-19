from __future__ import annotations

import pandas as pd

from src.utils.df_convert import pandas_to_polars


def test_pandas_nullable_numeric_converts_to_polars_numeric_with_null():
    """DB 读回的 pandas nullable 数值列不能被推断为 Polars Object。"""
    frame = pd.DataFrame({"PDT": pd.Series([2, pd.NA], dtype="Int64")})

    converted = pandas_to_polars(frame)

    assert converted.schema["PDT"].is_numeric()
    assert converted["PDT"].to_list() == [2, None]


def test_database_read_polars_returns_filtered_rows(db):
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
"""M1 累计订单视图与日粒度 OrderLog 持久化契约。"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.core.orchestrator.persistence_manager import PersistenceManager
from tests.compare_utils import compare_dataframes_by_key


def test_m1_orderlog_persists_only_daily_orders(monkeypatch) -> None:
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
    """审计关联中 SQL NULL、空串和数据库字符串 ``nan`` 表示同一缺失标识。"""
    left = pd.DataFrame({"material": ["MAT-1"], "changeover_id": [pd.NA], "quantity": [1]})
    right = pd.DataFrame({"material": ["MAT-1"], "changeover_id": ["nan"], "quantity": [1]})

    comparison = compare_dataframes_by_key(left, right)

    assert comparison["left_only_keys"] == 0
    assert comparison["right_only_keys"] == 0
    assert not comparison["column_differences"]
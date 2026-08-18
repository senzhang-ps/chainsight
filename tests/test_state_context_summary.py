"""StateContext 内存 Summary 与 `summary_*` 表持久化契约测试。"""

from __future__ import annotations

import pandas as pd

from src.core.orchestrator.persistence_manager import PersistenceManager
from src.models.base import Base
from src.models.viewcontext import SUMMARY_REGISTRY
from src.modules.state_context import StateContext


class _FakeDB:
    def __init__(self):
        self.frames: dict[str, pd.DataFrame] = {}
        self.deletes: list[tuple[str, dict]] = []

    def delete_where(self, table_name: str, conditions: dict) -> int:
        self.deletes.append((table_name, conditions))
        return 0

    def write_df(self, table_name: str, frame: pd.DataFrame) -> None:
        self.frames[table_name] = frame.copy(deep=True)


class _FakeOrchestrator:
    def __init__(self):
        self.db = _FakeDB()
        self.run_id = "summary-test-run"
        self.config_name = "summary-test"
        self.all_config = {
            "M3_SafetyStock": pd.DataFrame([
                {"material": "MAT-1", "location": "1000", "safety_stock_qty": 7},
            ]),
        }


def _result(**frames: pd.DataFrame) -> dict:
    return frames


def test_summary_tables_are_registered_for_migration() -> None:
    """`migrate()` 必须能从 Base.metadata 预建全部 `summary_*` 表。"""
    for table_name in SUMMARY_REGISTRY.values():
        assert table_name.startswith("summary_")
        table = Base.metadata.tables[table_name]
        assert {"run_id", "sim_date", "config_name", "db_write_time"}.issubset(
            table.columns.keys()
        )


def test_state_context_builds_all_registered_summary_outputs() -> None:
    ctx = StateContext("2025-01-01")
    ctx.unrestricted_inventory = {("MAT-1", "1000"): 80}
    ctx.record_summary_module_result("module1", _result(
        orders_df=pd.DataFrame([
            {"date": "2025-01-01", "material": "MAT-1", "location": "1000", "quantity": 10},
        ]),
        shipment_df=pd.DataFrame([
            {"date": "2025-01-01", "material": "MAT-1", "location": "1000", "quantity": 4},
        ]),
        cut_df=pd.DataFrame([
            {"date": "2025-01-01", "material": "MAT-1", "location": "1000", "quantity": 6},
        ]),
        supply_demand_df=pd.DataFrame([
            {"date": "2025-01-01", "material": "MAT-1", "location": "1000", "quantity": 10},
        ]),
    ), "2025-01-01")
    # 模拟 M1 跨日累计 OrderLog：第二天不应把同一业务订单重复计入 Summary。
    ctx.record_summary_module_result("module1", _result(
        orders_df=pd.DataFrame([
            {"date": "2025-01-01", "material": "MAT-1", "location": "1000", "quantity": 10},
        ]),
        shipment_df=pd.DataFrame(), cut_df=pd.DataFrame(), supply_demand_df=pd.DataFrame(),
    ), "2025-01-02")
    ctx.record_summary_module_result("module4", _result(
        production_df=pd.DataFrame([
            {"available_date": "2025-01-02", "material": "MAT-1", "location": "1000", "quantity": 9},
        ]),
        changeover_log=pd.DataFrame([
            {"changeover_end_date": "2025-01-01", "line": "L1"},
        ]),
        exceed_log=pd.DataFrame([
            {"date": "2025-01-01", "location": "1000", "material": "MAT-1", "exceed_qty": 1},
        ]),
    ), "2025-01-01")
    ctx.record_summary_module_result("module5", _result(
        deployment_plan=pd.DataFrame([
            {"date": "2025-01-01", "material": "MAT-1", "sending": "1000", "receiving": "2000", "deployed_qty": 3},
        ]),
    ), "2025-01-01")
    ctx.record_summary_module_result("module6", _result(
        delivery_plan=pd.DataFrame([
            {"date": "2025-01-01", "material": "MAT-1", "sending": "1000", "receiving": "2000", "delivered_qty": 3},
        ]),
        truck_usage=pd.DataFrame([
            {"date": "2025-01-01", "sending": "1000", "receiving": "2000", "used_trucks": 1},
        ]),
    ), "2025-01-01")
    ctx.day_end("2025-01-01")
    ctx.unrestricted_inventory[("MAT-1", "1000")] = 71
    ctx.day_end("2025-01-02")

    outputs = ctx.build_summary_outputs("2025-01-01", "2025-01-02", _FakeOrchestrator().all_config)

    assert set(outputs) == set(SUMMARY_REGISTRY)
    order_summary = outputs["full_order_shipment_cut_report"]
    assert len(order_summary) == 1
    assert order_summary.iloc[0][["order_qty", "shipment_qty", "cut_qty"]].tolist() == [10.0, 4.0, 6.0]
    inventory = outputs["historical_inventory_record"]
    first_day = inventory[(inventory["date"] == "2025-01-01") & (inventory["material"] == "MAT-1")].iloc[0]
    assert first_day["ending_inventory"] == 80
    assert first_day["safety_stock"] == 7


def test_summary_persistence_uses_only_registered_summary_tables() -> None:
    orch = _FakeOrchestrator()
    ctx = StateContext("2025-01-01")
    ctx.unrestricted_inventory = {("MAT-1", "1000"): 1}
    ctx.day_end("2025-01-01")

    results = PersistenceManager(orch).save_summary_outputs(ctx, "2025-01-01", "2025-01-01")

    assert set(results) == set(SUMMARY_REGISTRY.values())
    assert all(table_name.startswith("summary_") for table_name in results)
    assert {table_name for table_name, _ in orch.db.deletes} == set(SUMMARY_REGISTRY.values())
    assert set(orch.db.frames).issubset(set(SUMMARY_REGISTRY.values()))
    for frame in orch.db.frames.values():
        assert frame["run_id"].eq("summary-test-run").all()
        assert frame["config_name"].eq("summary-test").all()

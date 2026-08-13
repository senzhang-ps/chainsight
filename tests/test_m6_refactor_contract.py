"""ModuleSix 的生命周期、Context 与双引擎契约测试。"""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from src.modules.logistics_execution.backends import _PandasBackend, _PolarsBackend
from src.modules.logistics_execution.integration_refactor import ModuleSix
from src.modules.logistics_execution.main import run_daily_physical_flow
from src.modules.state_context import StateContext


class _Orch:
    def __init__(self, config: dict[str, pd.DataFrame], engine: str = "pandas"):
        self._config = config
        self.engine = engine
        self.load_count = 0

    def get_module_config(self, module_name: str) -> dict:
        return {}

    def load_datas(self, module) -> None:
        self.load_count += 1
        module.datas = {name: self._config.get(name, pd.DataFrame()) for name in module.schema}


class _Context:
    def __init__(self, deployment: pd.DataFrame | None = None, inventory: pd.DataFrame | None = None):
        self.deployment = deployment if deployment is not None else pd.DataFrame()
        self.inventory = inventory if inventory is not None else pd.DataFrame()
        self.requested_days: list[str] = []

    def get_open_deployment_view(self, day: str) -> pd.DataFrame:
        self.requested_days.append(day)
        return self.deployment.copy()

    def get_unrestricted_inventory_view(self, day: str) -> pd.DataFrame:
        self.requested_days.append(day)
        return self.inventory.copy()


def _config() -> dict[str, pd.DataFrame]:
    return {
        "M6_TruckReleaseCon": pd.DataFrame([{
            "sending": "PL01", "receiving": "DC01", "truck_type": "T1",
            "WFR": 0.5, "VFR": 0.5, "MDQ": 1, "optimal_type": "Y",
        }]),
        "M6_TruckCapacityPlan": pd.DataFrame([{
            "date": "2025-01-01", "sending": "PL01", "receiving": "DC01",
            "truck_type": "T1", "truck_number": 1,
        }]),
        "M6_TruckTypeSpecs": pd.DataFrame([{
            "truck_type": "T1", "capacity_qty_in_weight": 100, "capacity_qty_in_volume": 100,
        }]),
        "M6_MaterialMD": pd.DataFrame([{
            "material": "100", "demand_unit_to_weight": 1, "demand_unit_to_volume": 1,
        }]),
        "M6_DeliveryDelayDistribution": pd.DataFrame([{
            "sending": "PL01", "receiving": "DC01", "delay_days": 0, "probability": 1.0,
        }]),
        "M6_MDQBypassRules": pd.DataFrame(columns=["sending", "receiving", "truck_type", "demand_element", "condition_logic", "rule_id"]),
        "Global_DemandPriority": pd.DataFrame([{"demand_element": "normal", "priority": 1}]),
        "Global_LeadTime": pd.DataFrame([{"sending": "PL01", "receiving": "DC01", "PDT": 1, "OTD": 1, "GR": 0}]),
    }


def _deployment() -> pd.DataFrame:
    return pd.DataFrame([{
        "ori_deployment_uid": "D1", "material": "100", "sending": "PL01", "receiving": "DC01",
        "planned_deployment_date": "2025-01-01", "deployed_qty": 60, "demand_element": "normal",
    }])


def _inventory() -> pd.DataFrame:
    return pd.DataFrame([{"material": "100", "location": "PL01", "quantity": 60}])


def test_module_six_prepare_uses_orch_data_injection_once():
    orch = _Orch(_config())
    module = ModuleSix("2025-01-01", state_context=_Context(), orch=orch)

    module.prepare()
    module.prepare()

    assert orch.load_count == 1
    assert module._backend.static["TruckReleaseCon"].iloc[0]["sending"] == "PL01"


def test_module_six_selects_backend_from_orch_engine():
    assert isinstance(ModuleSix("2025-01-01", state_context=_Context(), orch=_Orch(_config()))._backend, _PandasBackend)
    assert isinstance(ModuleSix("2025-01-01", state_context=_Context(), orch=_Orch(_config(), "polars"))._backend, _PolarsBackend)


def test_module_six_polars_path_executes_independently():
    module = ModuleSix(
        "2025-01-01", state_context=_Context(_deployment(), _inventory()),
        orch=_Orch(_config(), "polars"), random_seed=7,
    )
    module.prepare()
    module.run()
    assert module.output()["delivery_plan"]["delivery_qty"].sum() == 60


def test_module_six_run_requires_prepare():
    module = ModuleSix("2025-01-01", state_context=_Context(), orch=_Orch(_config()))
    with pytest.raises(RuntimeError, match="prepare"):
        module.run()


def test_module_six_reads_context_views_and_returns_delivery_without_mutating_context():
    context = _Context(_deployment(), _inventory())
    module = ModuleSix("2025-01-01", state_context=context, orch=_Orch(_config()), random_seed=7)
    module.prepare()
    module.run()
    result = module.output()

    assert set(context.requested_days) == {"2025-01-01"}
    assert result["delivery_plan"]["delivery_qty"].sum() == 60
    assert result["delivery_plan"].iloc[0]["ori_deployment_uid"] == "D1"
    assert context.inventory.iloc[0]["quantity"] == 60


def test_module_six_skips_self_loop_deployment():
    deployment = _deployment()
    deployment.loc[0, "receiving"] = "PL01"
    module = ModuleSix("2025-01-01", state_context=_Context(deployment, _inventory()), orch=_Orch(_config()))
    module.prepare()
    module.run()

    assert module.output()["delivery_plan"].empty


def test_module_six_matches_legacy_api_against_current_state_context():
    """同一日初 StateContext 下，pandas 门面保持旧函数入口的输出语义。"""
    config = _config()
    legacy_context = StateContext("2025-01-01", orch=_Orch(config))
    refactor_context = StateContext("2025-01-01", orch=_Orch(config))
    for context in (legacy_context, refactor_context):
        context.initialize({"M1_InitialInventory": _inventory()})
        context.apply_deployment(_deployment(), "2025-01-01")
        context.day_start("2025-01-01")

    legacy = run_daily_physical_flow(
        config, legacy_context, pd.Timestamp("2025-01-01"),
        output_dir=".", random_seed=7, skip_file_output=True,
    )
    module = ModuleSix(
        "2025-01-01", state_context=refactor_context, orch=_Orch(config),
        random_seed=7,
    )
    module.prepare()
    module.run()

    pdt.assert_frame_equal(
        legacy["delivery_plan"].reset_index(drop=True),
        module.output()["delivery_plan"].reset_index(drop=True),
        check_dtype=False,
    )

    refactor_context.apply_delivery(module.output()["delivery_plan"], "2025-01-01")
    assert refactor_context.get_open_deployment_view("2025-01-01").empty
    assert refactor_context.get_unrestricted_inventory_view("2025-01-01").iloc[0]["quantity"] == 0
    assert refactor_context.get_planning_intransit_view("2025-01-01")["quantity"].sum() == 60
    assert refactor_context.get_delivery_shipment_log_view("2025-01-01")["quantity"].sum() == 60

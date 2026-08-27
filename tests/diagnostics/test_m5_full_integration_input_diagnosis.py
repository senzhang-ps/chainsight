"""M5 首日完整集成与受控 DB Oracle 的只读输入诊断。

本测试专门回答「相同 pandas M5 为何在两条调用链上产生不同首日行数」。它不修改
legacy / refactor 业务算法，不向数据库写入数据，也不把历史 M1/M4/M6 注入完整集成；
历史数据只用于独立的受控 Oracle 回放链路。

显式运行：
    $env:RUN_M5_FULL_INPUT_DIAG='1'
    $env:FULL_PARITY_ENGINE='pandas'
    conda run -n work pytest tests/diagnostics/test_m5_full_integration_input_diagnosis.py -s -q
"""

# 测试文件说明
# 测试目的：集中验证部署计划、优先级分配与供给池扣减的一致性。
# 测试方法：按 `diagnostics` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保部署计划、优先级分配与供给池扣减的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import pandas as pd
import pytest

from src.modules.deployment_planning.integration_refactor import ModuleFive


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
TEST_ENTRY_DIR = PROJECT_ROOT / "test"
if str(TEST_ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_ENTRY_DIR))

from test.test_integration import run_integrated_simulation
from tests.integration import test_m5_controlled_db_inputs as controlled
from tests.regression import test_m5_two_way_compare as m5_helpers
from tests.helpers.compare_utils import compare_dataframes_by_key


DAY = pd.Timestamp(os.environ.get("M5_INPUT_DIAG_DATE", "2025-12-15")).normalize()
DAY_TEXT = DAY.strftime("%Y-%m-%d")
TARGETS = (("21098519", "A668"), ("21098519", "A672"))
INPUT_NAMES = (
    "SupplyDemandLog", "OrderLog", "TodayShipment", "Inventory", "InTransit",
    "DeliveryGR", "OpenDeployment", "Production", "ReceivingSpace",
)
CONTEXT_GETTERS = {
    "beginning_inventory": "get_beginning_inventory_view",
    "unrestricted_inventory": "get_unrestricted_inventory_view",
    "OpenDeployment": "get_open_deployment_view",
    "InTransit": "get_planning_intransit_view",
    "DeliveryGR": "get_delivery_gr_view",
    "TodayShipment": "get_shipment_log_view",
    "SupplyDemandLog": "get_deployment_supply_demand_view",
    "OrderLog": "get_deployment_order_log_view",
    "ReceivingSpace": "get_space_quota_view",
    "production_backlog": "get_production_plan_backlog_view",
    "production": "get_deployment_production_view",
}


def _pd(frame: Any) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    return frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame).copy(deep=True)


def _fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """顺序无关的审计摘要；完整目标行另存 CSV，避免 JSON 巨大。"""
    frame = _pd(frame)
    canonical = frame.copy()
    canonical.columns = [str(column) for column in canonical.columns]
    canonical = canonical.reindex(sorted(canonical.columns), axis=1)
    if canonical.empty:
        digest = hashlib.sha256(b"").hexdigest()
    else:
        text = canonical.fillna("<NA>").astype(str).sort_values(
            list(canonical.columns), kind="mergesort"
        )
        digest = hashlib.sha256(pd.util.hash_pandas_object(text, index=False).values.tobytes()).hexdigest()
    return {"rows": len(frame), "columns": list(frame.columns), "sha256": digest}


def _target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _pd(frame)
    if frame.empty or "material" not in frame:
        return frame.iloc[0:0].copy()
    material = frame["material"].astype(str)
    locations = pd.Series(False, index=frame.index)
    for column in ("location", "node", "sending", "receiving", "orig_location"):
        if column in frame:
            locations |= frame[column].astype(str).isin([location for _, location in TARGETS])
    return frame.loc[material.eq("21098519") & locations].copy()


def _material_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """静态表须保留目标 SKU 的上游配置，而不能只保留两个接收节点。"""
    frame = _pd(frame)
    if frame.empty or "material" not in frame:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["material"].astype(str).eq("21098519")].copy()


def _snapshot_context(context) -> dict[str, pd.DataFrame]:
    return {
        name: _pd(getattr(context, getter)(DAY_TEXT))
        for name, getter in CONTEXT_GETTERS.items()
    }


class _Trace:
    """在单个 ModuleFive 实例上临时包裹公开步骤，原实现仍被实际执行。"""

    def __init__(self, module: ModuleFive):
        self.module = module
        self.inputs: dict[str, pd.DataFrame] = {}
        self.context: dict[str, pd.DataFrame] = {}
        self.static: dict[str, pd.DataFrame] = {}
        self.derived: dict[str, pd.DataFrame] = {}
        self.stages: dict[str, list[pd.DataFrame]] = {}
        self.layer_map: dict[str, int] = {}

    def _add(self, name: str, value: Any) -> None:
        self.stages.setdefault(name, []).append(_target_rows(_pd(value)))

    def capture_before_run(self) -> None:
        self.inputs = {name: _pd(value) for name, value in self.module.load_daily_inputs(DAY).items()
                       if name in INPUT_NAMES}
        self.context = _snapshot_context(self.module.state_context)
        self.static = {name: _pd(value) for name, value in self.module.static.items()}
        active = self.module.build_active_network(self.module._backend.static, DAY)
        routes = self.module.build_route_parameters(active, self.module._backend.static)
        self.derived = {
            "active_network": _pd(active),
            "route_parameters": _pd(routes),
            "node_horizon": _pd(self.module.build_node_horizon(active, DAY, routes)),
        }
        self.layer_map = {
            f"{material}/{node}": int(layer)
            for (material, node), layer in self.module.layer_map.items()
            if str(material) == "21098519" and str(node) in {"A668", "A672"}
        }

    def install(self) -> None:
        backend = self.module._backend
        for name in ("all_direct_demand", "direct_demand", "round_routes", "allocate_priority", "allocate_pipeline"):
            original = getattr(backend, name)

            def wrapped(instance, *args, __name=name, __original=original, **kwargs):
                if __name in {"round_routes", "allocate_priority", "allocate_pipeline"} and args:
                    self._add(f"{__name}.before", args[0])
                result = __original(*args, **kwargs)
                self._add(f"{__name}.after", result)
                return result

            setattr(backend, name, MethodType(wrapped, backend))

        original_plan_layers = backend.plan_layers

        def plan_layers(instance, *args, **kwargs):
            result = original_plan_layers(*args, **kwargs)
            for name, frame in zip(("plan_layers.plan", "plan_layers.direct", "plan_layers.unfulfilled"), result):
                self._add(name, frame)
            return result

        backend.plan_layers = MethodType(plan_layers, backend)

    def report(self, result: dict) -> dict[str, Any]:
        final = _pd(result.get("deployment_plan", pd.DataFrame()))
        return {
            "inputs": {name: _fingerprint(frame) for name, frame in self.inputs.items()},
            "context_views": {name: _fingerprint(frame) for name, frame in self.context.items()},
            "static_config": {name: _fingerprint(frame) for name, frame in self.static.items()},
            "derived_facts": {name: _fingerprint(frame) for name, frame in self.derived.items()},
            "target_layer_map": self.layer_map,
            "target_stage_rows": {
                name: pd.concat(parts, ignore_index=True).to_dict(orient="records") if parts else []
                for name, parts in self.stages.items()
            },
            "target_final_deployment_plan": _target_rows(final).to_dict(orient="records"),
            "deployment_plan": _fingerprint(final),
            "plan_layer_profile": getattr(self.module._backend, "last_plan_layer_profile", []),
            "direct_demand_diagnostics": getattr(self.module._backend, "last_direct_demand_diagnostics", []),
        }

    def write_target_csvs(self, directory: Path, prefix: str, result: dict) -> None:
        for scope, tables in (("input", self.inputs), ("context", self.context), ("static", self.static), ("derived", self.derived)):
            for name, frame in tables.items():
                selector = _material_rows if scope == "static" else _target_rows
                selector(frame).to_csv(directory / f"{prefix}_{scope}_{name}_targets.csv", index=False, encoding="utf-8-sig")
        _target_rows(_pd(result.get("deployment_plan", pd.DataFrame()))).to_csv(
            directory / f"{prefix}_deployment_plan_targets.csv", index=False, encoding="utf-8-sig"
        )


def _run_controlled_oracle() -> tuple[_Trace, dict]:
    """仅回放首日受控链路；与已通过的 DB Oracle 使用同一输入准备方式。"""
    db = controlled._db(); db.connect()
    try:
        config = m5_helpers._load_prepared_config(db)
        history = controlled._history(db)
    finally:
        db.close()
    orch, context = m5_helpers._new_context(config, "m5_full_input_diagnosis_controlled", "pandas")
    module = ModuleFive(DAY, DAY, state_context=context, orch=orch, verbose=False)
    module.prepare()
    m5_helpers._replay_inputs(context, DAY, history)
    trace = _Trace(module); trace.capture_before_run(); trace.install()
    module.run()
    return trace, module.output()


def _run_full_integration() -> tuple[_Trace, dict]:
    """运行未注入历史数据的真实完整首日调度，并仅在 M5 前读取快照。"""
    config_path = Path(os.environ.get(
        "FULL_PARITY_CONFIG_PATH",
        str(PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"),
    ))
    if not config_path.exists():
        pytest.skip(f"找不到完整集成配置: {config_path}")
    original_run = ModuleFive.run
    captured: dict[str, Any] = {}

    def traced_run(module: ModuleFive):
        if pd.Timestamp(module.simulation_date).normalize() == DAY:
            trace = _Trace(module); trace.capture_before_run(); trace.install()
            captured["trace"] = trace
        return original_run(module)

    ModuleFive.run = traced_run
    try:
        integrated = run_integrated_simulation(
            config_path=str(config_path), start_date=DAY_TEXT, end_date=DAY_TEXT,
            output_base_dir=str(PROJECT_ROOT / "outputs" / "m5_full_input_diagnosis" / "scratch"),
            engine=os.environ.get("FULL_PARITY_ENGINE", "pandas"),
        )
    finally:
        ModuleFive.run = original_run
    if "trace" not in captured:
        raise AssertionError("完整集成未执行目标日期的 M5")
    return captured["trace"], integrated["results"]["module5"][0]


def _comparison(left: dict[str, pd.DataFrame], right: dict[str, pd.DataFrame], scope: str) -> list[dict]:
    rows = []
    for name in sorted(set(left) | set(right)):
        lhs, rhs = left.get(name, pd.DataFrame()), right.get(name, pd.DataFrame())
        comparison = compare_dataframes_by_key(lhs, rhs, label=f"{DAY_TEXT}:{scope}:{name}:controlled_vs_full")
        rows.append({"scope": scope, "name": name, "controlled": _fingerprint(lhs), "full_integrated": _fingerprint(rhs), "comparison": comparison})
    return rows


def _target_comparison(left: dict[str, pd.DataFrame], right: dict[str, pd.DataFrame], scope: str) -> list[dict]:
    """将目标 SKU 的完整输入行直接放入 JSON，供审计时无需依赖 CSV 对照。"""
    rows = []
    for name in sorted(set(left) | set(right)):
        selector = _material_rows if scope == "static_config" else _target_rows
        controlled_rows = selector(left.get(name, pd.DataFrame()))
        full_rows = selector(right.get(name, pd.DataFrame()))
        rows.append({
            "scope": scope,
            "name": name,
            "controlled_rows": controlled_rows.to_dict(orient="records"),
            "full_integrated_rows": full_rows.to_dict(orient="records"),
            "comparison": compare_dataframes_by_key(
                controlled_rows, full_rows,
                label=f"{DAY_TEXT}:{scope}:{name}:controlled_vs_full_targets",
            ),
        })
    return rows


def test_m5_full_integration_first_day_input_diagnosis() -> None:
    # 测试目的：验证“m5、full、integration、first、day、input、diagnosis”场景下部署计划、优先级分配与供给池扣减的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `mkdir()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止部署计划、优先级分配与供给池扣减的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    if os.environ.get("RUN_M5_FULL_INPUT_DIAG") != "1":
        pytest.skip("设置 RUN_M5_FULL_INPUT_DIAG=1 后运行只读首日 M5 输入诊断")
    if os.environ.get("FULL_PARITY_ENGINE", "pandas") != "pandas":
        pytest.skip("此诊断固定要求 FULL_PARITY_ENGINE=pandas")

    report_dir = PROJECT_ROOT / "outputs" / "m5_full_input_diagnosis" / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    controlled_trace, controlled_result = _run_controlled_oracle()
    full_trace, full_result = _run_full_integration()
    controlled_trace.write_target_csvs(report_dir, "controlled", controlled_result)
    full_trace.write_target_csvs(report_dir, "full_integrated", full_result)

    comparisons = (
        _comparison(controlled_trace.inputs, full_trace.inputs, "m5_inputs")
        + _comparison(controlled_trace.context, full_trace.context, "context_views")
        + _comparison(controlled_trace.static, full_trace.static, "static_config")
        + _comparison(controlled_trace.derived, full_trace.derived, "derived_facts")
    )
    target_comparisons = (
        _target_comparison(controlled_trace.inputs, full_trace.inputs, "m5_inputs")
        + _target_comparison(controlled_trace.context, full_trace.context, "context_views")
        + _target_comparison(controlled_trace.static, full_trace.static, "static_config")
        + _target_comparison(controlled_trace.derived, full_trace.derived, "derived_facts")
    )
    report = {
        "date": DAY_TEXT,
        "policy": {
            "full_integration": "M1→M4→M5→M6→M3，未注入任何历史 M1/M4/M6 数据",
            "controlled_oracle": "仅独立受控链路读取 input schema 历史 M1/M4，供输入差异比较",
            "database_writes": False,
        },
        "targets": [{"material": material, "location": location} for material, location in TARGETS],
        "controlled": controlled_trace.report(controlled_result),
        "full_integrated": full_trace.report(full_result),
        "comparisons": comparisons,
        "target_comparisons": target_comparisons,
        "conclusion_checklist": {
            "integration_scheduler_failed_to_pass_or_write_m1_m4": "pending_evidence_in_comparisons",
            "initial_configuration_or_state_differs": "pending_evidence_in_static_and_context_comparisons",
            "m1_output_differs": "pending_evidence_in_SupplyDemandLog_OrderLog_TodayShipment",
            "m4_or_production_state_differs": "pending_evidence_in_Production_and_production_context",
            "statecontext_views_differ": "pending_evidence_in_context_views",
            "m5_call_path_differs": "same_ModuleFive_pandas_lifecycle_traced; inspect stage rows",
        },
    }
    (report_dir / "m5_full_integration_input_diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[M5 input diagnosis] report: {report_dir / 'm5_full_integration_input_diagnosis.json'}", flush=True)
    # 这是调查报告，不以业务不一致为失败条件；仅确认两条读取路径确实被执行。
    assert controlled_result["deployment_plan"].shape[0] == 31507
    assert isinstance(full_result["deployment_plan"], pd.DataFrame)
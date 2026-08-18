"""M6 五日 A/B 回放：固定历史 M5 部署计划，仅切换配置加载路径。

不运行 M1/M4/M5/M3，也不向数据库写入数据。每条链路使用独立的
StateContext，并在每天 M6 后调用 ``apply_delivery()`` 推进次日状态。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from src.core.orchestrator import Orch
from src.modules.logistics_execution.integration_refactor import ModuleSix
from src.modules.state_context import StateContext
from tests import test_m5_controlled_db_inputs as controlled
from tests import test_m5_two_way_compare as helpers
from tests.compare_utils import compare_dataframes_by_key, dataframe_difference_details


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
START_DATE, END_DATE = "2025-12-15", "2025-12-19"
RUNTIME_EXCEL = PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"
REPORT_BASE = PROJECT_ROOT / "outputs" / "m6_ab_five_day_config_parity"
M6_OUTPUTS = ("delivery_plan", "vehicle_log", "truck_usage", "unsatisfied_log", "validation_log", "bypass_log")
STATE_VIEWS = {
    "OpenDeployment": "get_open_deployment_view",
    "InTransit": "get_planning_intransit_view",
    "DeliveryShipment": "get_delivery_shipment_log_view",
    "Inventory": "get_unrestricted_inventory_view",
}


def _file_context(report_dir: Path) -> tuple[Orch, StateContext]:
    if not RUNTIME_EXCEL.exists():
        pytest.skip(f"找不到运行时 Excel: {RUNTIME_EXCEL}")
    orch = Orch(START_DATE, END_DATE, config_path=str(RUNTIME_EXCEL),
                output_path=str(report_dir / "file_config"), engine="pandas",
                skip_dq=True, enable_persistence=False)
    context = StateContext(START_DATE, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _db_context(report_dir: Path) -> tuple[Orch, StateContext]:
    db = controlled._db()
    db.connect()
    try:
        config = helpers._load_prepared_config(db)
    finally:
        db.close()
    return helpers._new_context(config, f"{report_dir.name}_db_prepared", "pandas")


def _historical_deployments() -> dict[str, pd.DataFrame]:
    """以 ``SELECT *`` 读取混合大小写的 legacy M5 输出列。"""
    db = controlled._db()
    db.connect()
    try:
        columns = [row[0] for row in db.execute_query(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (controlled.DB_SCHEMA, "module5_output_deploymentplan"),
        )]
        rows_by_day = {
            day.strftime("%Y-%m-%d"): db.execute_query(
                'SELECT * FROM "input"."module5_output_deploymentplan" '
                "WHERE run_id = %s AND sim_date::date = %s::date",
                (controlled.RUN_ID, day.strftime("%Y-%m-%d")),
            )
            for day in pd.date_range(START_DATE, END_DATE, freq="D")
        }
    finally:
        db.close()
    result = {}
    required = ["material", "sending", "receiving", "planned_deployment_date", "deployed_qty_invCon", "deployed_qty", "demand_element"]
    for day, rows in rows_by_day.items():
        frame = pd.DataFrame(rows, columns=columns).drop(columns=controlled.META, errors="ignore")
        if "deployed_qty_invcon" in frame and "deployed_qty_invCon" not in frame:
            frame = frame.rename(columns={"deployed_qty_invcon": "deployed_qty_invCon"})
        if "deployed_qty_invCon" not in frame and "deployed_qty" in frame:
            frame["deployed_qty_invCon"] = frame["deployed_qty"]
        if "deployed_qty" not in frame and "deployed_qty_invCon" in frame:
            frame["deployed_qty"] = frame["deployed_qty_invCon"]
        if "planned_deployment_date" not in frame and "date" in frame:
            frame = frame.rename(columns={"date": "planned_deployment_date"})
        missing = set(required) - set(frame.columns)
        if missing:
            pytest.skip(f"{day} 历史 M5 部署计划缺少 M6 回放字段: {sorted(missing)}")
        result[day] = frame.loc[:, required].copy()
    if all(frame.empty for frame in result.values()):
        pytest.skip(f"历史 M5 部署计划为空: run_id={controlled.RUN_ID}")
    return result


def _historical_delivery_plans() -> dict[str, pd.DataFrame]:
    """加载同一历史运行的 M6 Oracle，供受控回放确认历史一致性。"""
    db = controlled._db()
    db.connect()
    try:
        return {
            day.strftime("%Y-%m-%d"): controlled._load(db, "module6_output_deliveryplan", day)
            for day in pd.date_range(START_DATE, END_DATE, freq="D")
        }
    finally:
        db.close()


def _replay(
    orch: Orch,
    context: StateContext,
    deployments: dict[str, pd.DataFrame],
    history: dict[str, dict[str, pd.DataFrame]],
) -> list[dict]:
    """仅重放历史状态推进与 M6；不运行 M5。

    历史 M5 deployment plan 的可发运性依赖当日 M1 发货扣减、M4 生产入库和
    前序 M6 在途状态。若只注入 deployment plan，跨日库存不会与该业务事实对齐。
    """
    module = ModuleSix(START_DATE, state_context=context, orch=orch, random_seed=42)
    module.prepare()
    days = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        day_text = day.strftime("%Y-%m-%d")
        helpers._replay_inputs(context, day, history)
        context.apply_deployment(deployments[day_text], day_text)
        module.simulation_date = day
        module.run()
        result = module.output()
        context.apply_delivery(result["delivery_plan"], day_text)
        snapshots = {name: getattr(context, getter)(day_text) for name, getter in STATE_VIEWS.items()}
        context.day_end(day_text)
        days.append({"date": day_text, "result": result, "state": snapshots})
    return days


def _comparison_row(day: str, scope: str, name: str, left: pd.DataFrame, right: pd.DataFrame) -> dict:
    comparison = (
        {
            "label": f"{day}:M6:{scope}:{name}:db_vs_file",
            "left_rows": 0, "right_rows": 0, "key_columns": [],
            "left_only_keys": 0, "right_only_keys": 0, "matched_rows": 0,
            "column_differences": {}, "precision_differences": {},
            "schema": {"left_only_columns": [], "right_only_columns": []},
            "reason": "both_empty",
        }
        if left.empty and right.empty
        else compare_dataframes_by_key(left, right, label=f"{day}:M6:{scope}:{name}:db_vs_file")
    )
    return {
        "date": day, "scope": scope, "name": name,
        "db_rows": len(left), "file_rows": len(right),
        "comparison": comparison,
    }


def test_m6_ab_five_day_configuration_parity() -> None:
    if os.environ.get("RUN_M6_AB_FIVE_DAY_DIAG") != "1":
        pytest.skip("设置 RUN_M6_AB_FIVE_DAY_DIAG=1 后运行 M6 五日配置路径验证")
    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    deployments = _historical_deployments()
    historical_delivery = _historical_delivery_plans()
    db = controlled._db()
    db.connect()
    try:
        history = controlled._history(db)
    finally:
        db.close()
    db_orch, db_context = _db_context(report_dir)
    file_orch, file_context = _file_context(report_dir)
    db_days = _replay(db_orch, db_context, deployments, history)
    file_days = _replay(file_orch, file_context, deployments, history)

    comparisons, details = [], []
    for db_day, file_day in zip(db_days, file_days):
        for scope, names, left_data, right_data in (
            ("output", M6_OUTPUTS, db_day["result"], file_day["result"]),
            ("state", tuple(STATE_VIEWS), db_day["state"], file_day["state"]),
        ):
            for name in names:
                left = left_data.get(name, pd.DataFrame())
                right = right_data.get(name, pd.DataFrame())
                comparisons.append(_comparison_row(db_day["date"], scope, name, left, right))
                # 部分 M6 空输出没有列；compare_utils 的多重集明细无法为这种
                # 两端均为空且无键的 DataFrame 构造 merge key。
                detail = (
                    pd.DataFrame()
                    if left.empty and right.empty
                    else dataframe_difference_details(left, right)
                )
                detail.insert(0, "name", name)
                detail.insert(0, "scope", scope)
                detail.insert(0, "date", db_day["date"])
                details.append(detail)

        oracle = _comparison_row(
            db_day["date"], "historical_oracle", "delivery_plan",
            db_day["result"]["delivery_plan"], historical_delivery[db_day["date"]],
        )
        comparisons.append(oracle)
        oracle_detail = dataframe_difference_details(
            db_day["result"]["delivery_plan"], historical_delivery[db_day["date"]]
        ) if not (db_day["result"]["delivery_plan"].empty and historical_delivery[db_day["date"]].empty) else pd.DataFrame()
        oracle_detail.insert(0, "name", "delivery_plan")
        oracle_detail.insert(0, "scope", "historical_oracle")
        oracle_detail.insert(0, "date", db_day["date"])
        details.append(oracle_detail)

    pd.DataFrame([{
        "date": item["date"], "scope": item["scope"], "name": item["name"],
        "db_rows": item["db_rows"], "file_rows": item["file_rows"],
    } for item in comparisons]).to_csv(report_dir / "ab_m6_five_day_row_counts.csv", index=False, encoding="utf-8-sig")
    pd.concat(details, ignore_index=True).to_csv(report_dir / "ab_m6_five_day_differences.csv", index=False, encoding="utf-8-sig")
    report_path = report_dir / "ab_m6_five_day_config_parity.json"
    report_path.write_text(json.dumps({
        "historical_m5_run_id": controlled.RUN_ID,
        "policy": "固定历史 M1/M4 状态写回与 M5 deployment_plan；仅运行 M6 和 StateContext.apply_delivery；不运行 M1/M4/M5/M3 算法，不写数据库",
        "date_range": [START_DATE, END_DATE], "comparisons": comparisons,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[M6 A/B five day] report: {report_path}", flush=True)
    assert len(comparisons) == 5 * (len(M6_OUTPUTS) + len(STATE_VIEWS) + 1)
    assert report_path.exists()
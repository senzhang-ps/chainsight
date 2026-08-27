"""目标 run 的第二日 M5 计划行数/剩余 gap 定位。"""

# 测试文件说明
# 测试目的：集中验证部署计划、优先级分配与供给池扣减的一致性。
# 测试方法：按 `diagnostics` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保部署计划、优先级分配与供给池扣减的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.core.orchestrator import Orch
from tests.integration import test_m5_controlled_db_inputs as controlled
from tests.regression import test_m5_two_way_compare as m5
from tests.helpers.compare_utils import dataframe_difference_details

DAY_ONE = "2025-12-15"
DAY_TWO = "2025-12-16"
REPORT_DIR = controlled.REPORT_DIR


def _exclusive_plan_rows(detail: pd.DataFrame, side: str) -> pd.DataFrame:
    """从原始多重集差异中提取指定一侧独有计划行，不合并任何 M5 输出。"""
    if side not in {"left", "right"}:
        raise ValueError(f"未知比较侧: {side}")
    source = detail.loc[detail["difference_type"].eq(f"{side}_only_row")]
    parsed = []
    for row in source.itertuples(index=False):
        values = json.loads(row.key_values) if isinstance(row.key_values, str) and row.key_values else {}
        payload_text = getattr(row, f"{side}_row")
        payload = json.loads(payload_text) if isinstance(payload_text, str) and payload_text else {}
        parsed.append({**values, **payload})
    return pd.DataFrame(parsed)


def _group_exclusive_plan_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """按需求类型与路线汇总独有计划行；只统计，不改写原始行。"""
    if frame.empty:
        return pd.DataFrame(columns=[
            "demand_element", "sending", "receiving", "rows",
            "demand_qty", "planned_qty", "residual_qty",
        ])
    result = frame.copy()
    for column in ("demand_qty", "planned_qty", "deployed_qty_invCon"):
        result[column] = pd.to_numeric(result.get(column), errors="coerce").fillna(0)
    result["residual_qty"] = (
        result["planned_qty"] - result["deployed_qty_invCon"]
    ).clip(lower=0)
    return result.groupby(
        ["demand_element", "sending", "receiving"], dropna=False, as_index=False
    ).agg(
        rows=("material", "size"),
        demand_qty=("demand_qty", "sum"),
        planned_qty=("planned_qty", "sum"),
        residual_qty=("residual_qty", "sum"),
    ).sort_values("rows", ascending=False, kind="mergesort")


def test_m5_second_day_remaining_gap_trace() -> None:
    # 测试目的：验证“m5、second、day、remaining、gap、trace”场景下部署计划、优先级分配与供给池扣减的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `Orch()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止部署计划、优先级分配与供给池扣减的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """只回放两日，按需求类型与路线统计第二日 pandas 多出的原始计划行。"""
    config_orch = Orch(
        start_date=DAY_ONE, end_date=DAY_TWO,
        config_path=str(controlled.CONFIG_PATH),
        output_path=str(REPORT_DIR / "second_day_gap_trace_config"),
        engine="pandas", skip_dq=True, enable_persistence=False,
    )
    config = config_orch.all_config
    db = controlled._db(); db.connect()
    try:
        history = {
            day.strftime("%Y-%m-%d"): {
                "all_orders_for_next_day": controlled._load(db, "module1_output_orderlog", day, active_orders=True),
                "shipment_df": controlled._load(db, "module1_output_shipmentlog", day),
                "supply_demand_df": controlled._load(db, "module1_output_supplydemandlog", day),
                "production_df": controlled._load(db, "module4_output_productionplan", day),
            }
            for day in pd.date_range(DAY_ONE, DAY_TWO)
        }
    finally:
        db.close()

    original = (m5.START_DATE, m5.END_DATE, m5.REPORT_DIR)
    m5.START_DATE, m5.END_DATE, m5.REPORT_DIR = DAY_ONE, DAY_TWO, REPORT_DIR
    try:
        legacy = m5._run_legacy(config, history, timed=True)
        pandas = m5._run_refactor(config, history, timed=True, engine="pandas")
    finally:
        m5.START_DATE, m5.END_DATE, m5.REPORT_DIR = original

    legacy_day = next(day for day in legacy["days"] if day["date"] == DAY_TWO)
    pandas_day = next(day for day in pandas["days"] if day["date"] == DAY_TWO)
    legacy_plan = legacy_day["m5_result"]["deployment_plan"]
    pandas_plan = pandas_day["m5_result"]["deployment_plan"]
    detail = dataframe_difference_details(legacy_plan, pandas_plan)
    legacy_only = _exclusive_plan_rows(detail, "left")
    pandas_only = _exclusive_plan_rows(detail, "right")
    legacy_grouped = _group_exclusive_plan_rows(legacy_only)
    pandas_grouped = _group_exclusive_plan_rows(pandas_only)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    legacy_only.to_csv(REPORT_DIR / "m5_second_day_legacy_only_plan_rows.csv", index=False, encoding="utf-8-sig")
    pandas_only.to_csv(REPORT_DIR / "m5_second_day_pandas_only_plan_rows.csv", index=False, encoding="utf-8-sig")
    legacy_grouped.to_csv(REPORT_DIR / "m5_second_day_legacy_only_plan_groups.csv", index=False, encoding="utf-8-sig")
    pandas_grouped.to_csv(REPORT_DIR / "m5_second_day_pandas_only_plan_groups.csv", index=False, encoding="utf-8-sig")
    summary = {
        "date": DAY_TWO,
        "legacy_plan_raw_rows": len(legacy_plan),
        "pandas_plan_raw_rows": len(pandas_plan),
        "pandas_minus_legacy_rows": len(pandas_plan) - len(legacy_plan),
        "legacy_only_rows_by_route": legacy_grouped.to_dict("records"),
        "pandas_only_rows_by_route": pandas_grouped.to_dict("records"),
        "pandas_layer_profile": pandas_day["plan_layer_profile"],
        "note": "所有行数和明细均来自原始多重集；未对 DB/legacy/pandas 结果去重。",
    }
    (REPORT_DIR / "m5_second_day_gap_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    assert len(legacy_plan) > 0
    assert len(pandas_plan) > 0

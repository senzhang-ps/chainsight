"""A/B 五日 M5 回放：固定历史业务输入，仅切换配置加载路径。"""

# 测试文件说明
# 测试目的：集中验证部署计划、优先级分配与供给池扣减的一致性。
# 测试方法：按 `diagnostics` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保部署计划、优先级分配与供给池扣减的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from src.core.orchestrator import Orch
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.state_context import StateContext
from tests.integration import test_m5_controlled_db_inputs as controlled
from tests.regression import test_m5_two_way_compare as helpers
from tests.helpers.compare_utils import compare_dataframes_by_key, dataframe_difference_details


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
START_DATE, END_DATE = "2025-12-15", "2025-12-19"
RUNTIME_EXCEL = PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"
REPORT_BASE = PROJECT_ROOT / "outputs" / "m5_ab_five_day_config_parity"
M5_OUTPUTS = ("deployment_plan", "unfulfilled_log", "stock_on_hand_log", "validation_log")


def _file_context(report_dir: Path) -> tuple[Orch, StateContext]:
    if not RUNTIME_EXCEL.exists():
        pytest.skip(f"找不到运行时 Excel: {RUNTIME_EXCEL}")
    orch = Orch(START_DATE, END_DATE, config_path=str(RUNTIME_EXCEL),
                output_path=str(report_dir / "file_config"), engine="pandas",
                skip_dq=True, enable_persistence=False)
    context = StateContext(START_DATE, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _db_context(report_dir: Path) -> tuple[Orch, StateContext, dict]:
    db = controlled._db(); db.connect()
    try:
        config = helpers._load_prepared_config(db)
        history = controlled._history(db)
    finally:
        db.close()
    orch, context = helpers._new_context(config, f"{report_dir.name}_db_prepared", "pandas")
    return orch, context, history


def _replay(name: str, orch: Orch, context: StateContext, history: dict) -> list[dict]:
    module = ModuleFive(START_DATE, START_DATE, state_context=context, orch=orch, verbose=False)
    module.prepare()
    days = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        m1, m4 = helpers._replay_inputs(context, day, history)
        module.simulation_date = day
        module.run()
        result = module.output()
        helpers._finish_day(context, day, result)
        # 本测试不运行 M6；仅注入历史 M6 delivery 以推进次日 StateContext，
        # 与已验证的 controlled M5 Oracle 链路保持一致。
        delivery = history[day.strftime("%Y-%m-%d")]["delivery_plan"]
        if not delivery.empty:
            context.apply_delivery(delivery, day.strftime("%Y-%m-%d"))
        days.append({"date": day.strftime("%Y-%m-%d"), "result": result})
    return days


def test_m5_ab_five_day_configuration_parity() -> None:
    # 测试目的：验证“m5、ab、five、day、configuration、parity”场景下部署计划、优先级分配与供给池扣减的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `mkdir()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止部署计划、优先级分配与供给池扣减的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    if os.environ.get("RUN_M5_AB_FIVE_DAY_DIAG") != "1":
        pytest.skip("设置 RUN_M5_AB_FIVE_DAY_DIAG=1 后运行 A/B 五日配置路径验证")
    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    db_orch, db_context, history = _db_context(report_dir)
    file_orch, file_context = _file_context(report_dir)
    a_days = _replay("A_db_prepared_config", db_orch, db_context, history)
    b_days = _replay("B_file_config", file_orch, file_context, history)

    comparisons, details, row_counts = [], [], []
    for a_day, b_day in zip(a_days, b_days):
        for output in M5_OUTPUTS:
            a_frame = a_day["result"].get(output, pd.DataFrame())
            b_frame = b_day["result"].get(output, pd.DataFrame())
            comparison = compare_dataframes_by_key(
                a_frame, b_frame,
                label=f"{a_day['date']}:M5:{output}:A_db_prepared_vs_B_file_config",
            )
            comparisons.append({"date": a_day["date"], "output": output,
                                "a_rows": len(a_frame), "b_rows": len(b_frame),
                                "comparison": comparison})
            detail = dataframe_difference_details(a_frame, b_frame)
            detail.insert(0, "output", output); detail.insert(0, "date", a_day["date"])
            details.append(detail)
            row_counts.append({"date": a_day["date"], "output": output,
                               "A_db_prepared_rows": len(a_frame), "B_file_config_rows": len(b_frame)})

    pd.DataFrame(row_counts).to_csv(report_dir / "ab_m5_five_day_row_counts.csv", index=False, encoding="utf-8-sig")
    pd.concat(details, ignore_index=True).to_csv(report_dir / "ab_m5_five_day_differences.csv", index=False, encoding="utf-8-sig")
    report = {
        "policy": "A/B 均注入同一历史 M1/M4，且仅注入历史 M6 delivery 推进次日状态；不运行 M6/M3，不写数据库",
        "A": "prepared DB configuration", "B": str(RUNTIME_EXCEL),
        "date_range": [START_DATE, END_DATE], "comparisons": comparisons,
    }
    report_path = report_dir / "ab_m5_five_day_config_parity.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[M5 A/B five day] report: {report_path}", flush=True)
    assert len(comparisons) == 5 * len(M5_OUTPUTS)
    assert report_path.exists()
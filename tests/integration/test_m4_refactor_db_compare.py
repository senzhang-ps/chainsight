"""重构 pandas M1 + M4 与 input 数据库 M4 历史基准的只读审计测试。

运行策略：
- 每天先运行 refactor pandas M1；其内存结果及 shipment 状态变更属于本次回放链路。
- M3 绝不运行重构计算：首日明确传空，后续 M4 仅消费同一 run 的前一日
  ``input.module3_output_netdemand``。
- M4 只运行 refactor pandas；逐日与 ``module4_%`` 输出表比较。M4 跨日状态
    属于 ``DbRuntimeState``，在正常 DB 链路中只写入 checkpoint JSON，不是
    orchestrator 的逐日输出表，故仅记录状态规模、不将其伪装成逐日数据库表比较。
    业务差异写入审计报告，不使测试失败。

显式运行：
    conda run --no-capture-output -n work pytest tests/integration/test_m4_refactor_db_compare.py -s -q

可通过 M4_DB_COMPARE_* 环境变量覆盖 schema、run_id、配置名和日期范围。
"""

# 测试文件说明
# 测试目的：集中验证生产排程、产能分配与生产结果的一致性。
# 测试方法：按 `integration` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保生产排程、产能分配与生产结果的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.core.main_integration.config_loader import (
    load_configuration_from_dict,
    prepare_configuration,
)
from src.core.orchestrator import Orch
from src.core.run.db_config import _load_config_from_database
from src.models.module import OUTPUT_REGISTRY
from src.modules.demand_planning.integration_refactor import ModuleOne
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.state_context import StateContext
from tests.helpers.compare_utils import (
    compare_dataframes_by_key,
    dataframe_difference_details,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DB_SCHEMA = os.getenv("M4_DB_COMPARE_DB_SCHEMA", "input")
CONFIG_SCHEMA = os.getenv("M4_DB_COMPARE_CONFIG_SCHEMA", DB_SCHEMA)
CONFIG_NAME = os.getenv("M4_DB_COMPARE_CONFIG_NAME", "OC_Paste_S1_20251224_repare")
HISTORICAL_RUN_ID = os.getenv(
    "M4_DB_COMPARE_HISTORICAL_RUN_ID",
    "db_OC_Paste_S1_20251224_repare_20260814_132746",
)
START_DATE = os.getenv("M4_DB_COMPARE_START_DATE", "2025-12-15")
END_DATE = os.getenv("M4_DB_COMPARE_END_DATE", "2025-12-19")
REPORT_DIR = PROJECT_ROOT / "outputs" / "m4_refactor_db_compare"
META_COLUMNS = {"run_id", "sim_date", "config_name", "db_write_time", "file_date"}
M3_TABLE = "module3_output_netdemand"
M4_OUTPUT_TABLES = OUTPUT_REGISTRY["module4"]


def _db(schema: str) -> DatabaseConnection:
    settings = get_database_config()
    return DatabaseConnection(
        host=settings["host"],
        port=settings["port"],
        database=settings["database"],
        user=settings["user"],
        password=settings["password"],
        schema=schema,
        auto_create_schema=False,
    )


def _qualified(table_name: str) -> str:
    return f'"{DB_SCHEMA}"."{table_name}"'


def _table_columns(db: DatabaseConnection, table_name: str) -> list[str]:
    rows = db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (DB_SCHEMA, table_name),
    )
    return [row[0] for row in rows]


def _read_run_day(
    db: DatabaseConnection,
    table_name: str,
    day: pd.Timestamp,
) -> pd.DataFrame:
    """按历史 run 与 sim_date 只读读取一张基准表，并移除运行元数据。"""
    columns = _table_columns(db, table_name)
    if not columns:
        return pd.DataFrame()
    rows = db.execute_query(
        f"SELECT {', '.join(columns)} FROM {_qualified(table_name)} "
        "WHERE run_id = %s AND sim_date::date = %s::date",
        (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d")),
    )
    return pd.DataFrame(rows, columns=columns).drop(columns=list(META_COLUMNS), errors="ignore")


def _load_m3_inputs(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    """读取历史 M3；只为 D>首日准备 D-1 的输入，首日绝不读取 M3。"""
    if not _table_columns(db, M3_TABLE):
        pytest.fail(f"找不到 M3 历史输入表: {DB_SCHEMA}.{M3_TABLE}")

    inputs: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    for day in dates[1:]:
        source_day = day - pd.Timedelta(days=1)
        frame = _read_run_day(db, M3_TABLE, source_day)
        if frame.empty:
            missing.append(source_day.strftime("%Y-%m-%d"))
        inputs[day.strftime("%Y-%m-%d")] = frame
    if missing:
        pytest.fail(
            f"run_id={HISTORICAL_RUN_ID} 缺少 M4 所需的前一日 M3: {', '.join(missing)}"
        )
    return inputs


def _load_config() -> dict[str, pd.DataFrame]:
    """从配置 schema 读取真实配置，供同一次 pandas M1/M4 回放共用。"""
    db = _db(CONFIG_SCHEMA)
    try:
        db.connect()
        raw = _load_config_from_database(db, CONFIG_NAME)
    finally:
        db.close()
    if not raw:
        pytest.fail(f"找不到数据库配置: {CONFIG_SCHEMA}/{CONFIG_NAME}")
    config = prepare_configuration(load_configuration_from_dict(raw, CONFIG_NAME))
    # 当前 DB 配置可能保留 Excel 的历史 MCT 命名；M4 refactor schema 使用小写 mct。
    mlcfg = config.get("M4_MaterialLocationLineCfg")
    if mlcfg is not None and "MCT" in mlcfg.columns and "mct" not in mlcfg.columns:
        config["M4_MaterialLocationLineCfg"] = mlcfg.rename(columns={"MCT": "mct"})
    return config


def _new_runtime(config: dict[str, pd.DataFrame]) -> tuple[Orch, StateContext]:
    orch = Orch(
        start_date=START_DATE,
        end_date=END_DATE,
        config_dict={name: value.copy() if isinstance(value, pd.DataFrame) else value
                     for name, value in config.items()},
        output_path=str(REPORT_DIR / "scratch"),
        engine="pandas",
        skip_dq=True,
        enable_persistence=False,
    )
    context = StateContext(simulation_date=START_DATE, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _comparison(left: pd.DataFrame, right: pd.DataFrame, *, label: str, key_columns=None) -> dict:
    """保留空表语义，并将比较结果转换为清晰的 current/historical 命名。"""
    left = pd.DataFrame() if left is None else left
    right = pd.DataFrame() if right is None else right
    if left.empty and right.empty:
        return {
            "label": label, "current_rows": 0, "historical_rows": 0,
            "consistent": True, "reason": "both_empty",
        }
    result = compare_dataframes_by_key(
        left, right, label=label, key_columns=key_columns,
    )
    result["current_rows"] = result.pop("left_rows")
    result["historical_rows"] = result.pop("right_rows")
    result["consistent"] = not (
        result.get("error") or result["left_only_keys"] or result["right_only_keys"]
        or result["column_differences"] or result["precision_differences"]
    )
    return result


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(".", "_")


def _write_details(
    day: str,
    name: str,
    current: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    key_columns=None,
) -> str:
    # pandas 3 + pyarrow string dtype 在“双方均为空但带业务键”的 outer merge
    # 中会触发 ArrowInvalid。空侧本身已有明确的表级语义，直接写一个审计摘要。
    if current.empty or historical.empty:
        if current.empty and historical.empty:
            difference_type = "both_empty"
        elif current.empty:
            difference_type = "current_empty"
        else:
            difference_type = "historical_empty"
        details = pd.DataFrame([{
            "difference_type": difference_type,
            "current_rows": len(current),
            "historical_rows": len(historical),
        }])
    else:
        details = dataframe_difference_details(
            current, historical, key_columns=key_columns,
        )
    path = REPORT_DIR / "details" / f"{day}_{_safe_name(name)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _write_report(report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "m4_refactor_db_compare.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# pandas M1 + M4 与数据库 M4 基准对比", "",
        f"- schema: `{DB_SCHEMA}`", f"- 历史 run_id: `{HISTORICAL_RUN_ID}`",
        f"- 范围: {START_DATE} 至 {END_DATE}",
        "- M1：每日 refactor pandas 内存计算；不读取历史 M1 输出。",
        "- M3：首日为空；后续 M4 仅消费指定 run 的前一日数据库 M3。",
        "- M4：refactor pandas 内存计算；当前运行不写数据库。", "",
        "## 汇总", "",
        "| 日期 | M1 orders / shipments | M3 来源 | M4 不一致表数 |",
        "|---|---:|---|---:|",
    ]
    for day in report["days"]:
        m1 = day["m1"]
        mismatch_count = sum(not item["comparison"]["consistent"] for item in day["comparisons"])
        lines.append(
            f"| {day['date']} | {m1['orders_rows']} / {m1['shipments_rows']} | "
            f"{day['m3_source']} | {mismatch_count} |"
        )
    lines.extend([
        "", "## M4 输出行数差异", "",
        "| 日期 | 表 | pandas 当前行数 | DB 基准行数 | 差异（当前−基准） |",
        "|---|---|---:|---:|---:|",
    ])
    for day in report["days"]:
        for item in day["comparisons"]:
            comparison = item["comparison"]
            current_rows = comparison["current_rows"]
            historical_rows = comparison["historical_rows"]
            table_name = item["table"].rsplit(".", 1)[-1]
            lines.append(
                f"| {day['date']} | {table_name} | {current_rows} | "
                f"{historical_rows} | {current_rows - historical_rows:+d} |"
            )
    lines.extend(["", "业务差异只写入 JSON 与 details CSV，不作为 pytest 失败条件。"])
    markdown_path = REPORT_DIR / "m4_refactor_db_compare.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


@pytest.fixture(scope="module")
def db_compare_report() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    db = _db(DB_SCHEMA)
    try:
        db.connect()
        discovered_m4_tables = [row[0] for row in db.execute_query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name LIKE 'module4\\_%%' ESCAPE '\\' "
            "ORDER BY table_name",
            (DB_SCHEMA,),
        )]
        missing_tables = sorted(set(M4_OUTPUT_TABLES.values()) - set(discovered_m4_tables))
        if missing_tables:
            pytest.fail(f"缺少 M4 输出基准表: {', '.join(missing_tables)}")
        m3_inputs = _load_m3_inputs(db)
    finally:
        db.close()

    unmapped_m4_tables = sorted(set(discovered_m4_tables) - set(M4_OUTPUT_TABLES.values()))
    config = _load_config()
    orch, context = _new_runtime(config)
    m1 = ModuleOne(simulation_date=pd.Timestamp(START_DATE), orchestrator=context, orch=orch)
    m1.prepare()
    m4 = ModuleFour(
        simulation_date=pd.Timestamp(START_DATE),
        simulation_start_date=pd.Timestamp(START_DATE),
        orchestrator=orch,
        orch=orch,
    )
    m4.prepare()

    line_states_by_date: dict[str, dict] = {}
    allocated_by_date: dict[str, dict] = {}
    days: list[dict] = []
    db = _db(DB_SCHEMA)
    try:
        db.connect()
        for day in pd.date_range(START_DATE, END_DATE, freq="D"):
            day_key = day.strftime("%Y-%m-%d")
            previous_key = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            # M1 是当次 pandas 计算的当前业务输入，不能由历史 M1 表替代。
            context.day_start(day_key)
            m1.simulation_date = day
            m1.run()
            m1_result = m1.output()
            shipments = m1_result.get("shipment_df", pd.DataFrame())
            if not shipments.empty:
                context.apply_shipments(shipments, day_key)

            # 首日 M3 明确为空；之后严格消费数据库 D-1 的 M3，绝不运行 M3。
            m3_frame = None if day_key == START_DATE else m3_inputs[day_key].copy()
            m4.module3_result = None if m3_frame is None else {"net_demand_df": m3_frame}
            m4.simulation_date = day
            previous_capacity: dict = {}
            for capacity_day in sorted(allocated_by_date):
                if capacity_day >= day_key:
                    break
                for key, value in allocated_by_date[capacity_day].items():
                    previous_capacity[key] = previous_capacity.get(key, 0.0) + value
            m4.previous_line_states_override = line_states_by_date.get(previous_key, {})
            m4.allocated_capacity_override = previous_capacity
            m4.run()
            m4_result = m4.output()

            line_states = m4_result.get("current_line_states", {})
            allocated = m4_result.get("current_allocated_capacity", {})
            line_states_by_date[day_key] = line_states
            allocated_by_date[day_key] = allocated
            context.day_end(day_key)

            current_by_table = {
                table: m4_result.get(output_key, pd.DataFrame())
                for output_key, table in M4_OUTPUT_TABLES.items()
            }
            comparisons = []
            for table_name, current in current_by_table.items():
                historical = _read_run_day(db, table_name, day)
                comparison = _comparison(
                    current, historical, label=f"{day_key}:{table_name}",
                )
                comparisons.append({
                    "table": f"{DB_SCHEMA}.{table_name}",
                    "comparison": comparison,
                    "details_file": _write_details(
                        day_key, table_name, current, historical,
                    ),
                })
            days.append({
                "date": day_key,
                "m1": {
                    "source": "refactor_pandas_in_memory",
                    "orders_rows": len(m1_result.get("orders_df", pd.DataFrame())),
                    "shipments_rows": len(shipments),
                },
                "m3_source": "empty (first day)" if m3_frame is None else f"{DB_SCHEMA}.{M3_TABLE}:{previous_key}",
                "m3_rows": 0 if m3_frame is None else len(m3_frame),
                "m4_runtime_state": {
                    "line_state_lines": len(line_states),
                    "allocated_capacity_keys": len(allocated),
                    "persistence": "DbRuntimeState -> sim_checkpoint.orch_state_json",
                },
                "comparisons": comparisons,
            })
    finally:
        db.close()

    report = {
        "database_schema": DB_SCHEMA,
        "historical_run_id": HISTORICAL_RUN_ID,
        "config_schema": CONFIG_SCHEMA,
        "config_name": CONFIG_NAME,
        "engine": "pandas",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "database_writes": False,
        "m1_input_policy": "refactor_pandas_in_memory_daily",
        "m3_input_policy": "first_day_empty_then_database_previous_day",
        "discovered_module4_tables": discovered_m4_tables,
        "unmapped_module4_tables": unmapped_m4_tables,
        "m4_runtime_state_policy": (
            "DbRuntimeState only during the run; serialized as db_runtime_state "
            "inside sim_checkpoint.orch_state_json for resume, not persisted as daily orchestrator rows"
        ),
        "compared_tables": list(M4_OUTPUT_TABLES.values()),
        "days": days,
    }
    report_path = _write_report(report)
    report["report_path"] = report_path
    return report


def test_m4_refactor_db_compare_completes_five_days(db_compare_report: dict) -> None:
    # 测试目的：验证“m4、refactor、db、compare、completes、five、days”场景下生产排程、产能分配与生产结果的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `all()`，再通过 7 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止生产排程、产能分配与生产结果的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """强制验证回放契约；业务不一致由报告承载。"""
    assert len(db_compare_report["days"]) == 5
    assert db_compare_report["m1_input_policy"] == "refactor_pandas_in_memory_daily"
    assert db_compare_report["days"][0]["m3_source"] == "empty (first day)"
    assert all(
        day["m3_source"].startswith(f"{DB_SCHEMA}.{M3_TABLE}:")
        for day in db_compare_report["days"][1:]
    )
    assert set(db_compare_report["compared_tables"]) == set(M4_OUTPUT_TABLES.values())
    assert "DbRuntimeState" in db_compare_report["m4_runtime_state_policy"]
    assert db_compare_report["report_path"].exists()


def test_m4_refactor_db_compare_covers_every_output_table(db_compare_report: dict) -> None:
    # 测试目的：验证“m4、refactor、db、compare、covers、every、output、table”场景下生产排程、产能分配与生产结果的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `set()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止生产排程、产能分配与生产结果的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """每一天都必须比较全部四张 M4 输出表。"""
    expected = set(db_compare_report["compared_tables"])
    for day in db_compare_report["days"]:
        actual = {item["table"].split(".", 1)[1] for item in day["comparisons"]}
        assert actual == expected, day["date"]
        assert all(Path(item["details_file"]).exists() for item in day["comparisons"])

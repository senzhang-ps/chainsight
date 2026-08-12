"""真实数据库配置下的主项目旧 M1 与 Decoupling M1 五日性能对比。

旧实现必须从 ``chainsight-main`` 的数据库模式调用链执行；重构实现和全部
报告产物均位于 ``ChainSight-Decoupling``。两端使用独立状态逐日回放，业务
差异仅写入报告，不使测试失败。

运行：
    conda run -n work pytest tests/test_m1_two_way_compare.py -s -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
from src.modules.demand_planning.integration_refactor import ModuleOne
from src.modules.state_context import StateContext
from tests.compare_utils import compare_dataframes_by_key


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MAIN_PROJECT_ROOT = PROJECT_ROOT.parent / "chainsight-main"
CONFIG_NAME = "OC_Paste_S1_20251224_repare"
START_DATE = "2025-12-15"
END_DATE = "2025-12-19"
REPORT_DIR = PROJECT_ROOT / "outputs" / "m1_two_way_compare"
PROGRESS_PATH = REPORT_DIR / "m1_two_way_compare.progress.log"
LEGACY_RESULT_PATH = REPORT_DIR / "m1_legacy_result.json"
LEGACY_CONFIG_PATH = REPORT_DIR / "m1_database_config.json"
LEGACY_RUNNER_PATH = PROJECT_ROOT / "tests" / "m1_legacy_runner.py"
M1_FRAME_KEYS = (
    "orders_df",
    "shipment_df",
    "cut_df",
    "supply_demand_df",
    "summary_df",
)


def _progress(message: str) -> None:
    """绕过项目 logger 对 stdout 的重定向，持续输出长任务进度。"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [M1 compare] {message}\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("a", encoding="utf-8") as progress_file:
        progress_file.write(line)
    sys.__stdout__.write(line)
    sys.__stdout__.flush()


def _db() -> DatabaseConnection:
    config = get_database_config()
    return DatabaseConnection(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        user=config["user"],
        password=config["password"],
        schema=config.get("default_schema", "public"),
        auto_create_schema=False,
    )


def _load_prepared_config(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    raw = _load_config_from_database(db, CONFIG_NAME)
    if not raw:
        pytest.skip(f"数据库中找不到配置: {CONFIG_NAME}")
    return prepare_configuration(load_configuration_from_dict(raw, CONFIG_NAME))


def _new_context(config: dict[str, pd.DataFrame], *, engine: str) -> tuple[Orch, StateContext]:
    """为指定重构后端创建独立 Orch / StateContext，避免共享可变库存。"""
    orch = Orch(
        start_date=START_DATE,
        end_date=END_DATE,
        config_dict={name: frame.copy() for name, frame in config.items()},
        output_path=str(REPORT_DIR / f"refactor_{engine}_scratch"),
        engine=engine,
        skip_dq=True,
    )
    context = StateContext(simulation_date=START_DATE, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _write_database_config(config: dict[str, pd.DataFrame]) -> None:
    """将数据库读取并准备完成的配置快照交给主项目独立进程。"""
    payload = {
        name: frame.to_json(orient="split", date_format="iso", default_handler=str)
        for name, frame in config.items()
    }
    LEGACY_CONFIG_PATH.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_main_legacy() -> dict:
    """通过独立进程调用 chainsight-main 的 DB 模式 M1 入口。

    独立解释器避免两个工作区同名 ``src`` 包相互覆盖。计时在运行器内完成，
    因此不包含子进程启动、数据库读取和配置初始化。
    """
    if not MAIN_PROJECT_ROOT.is_dir():
        pytest.skip(f"找不到主项目目录: {MAIN_PROJECT_ROOT}")
    environment = os.environ.copy()
    environment.update({
        "CHAINSIGHT_MAIN_ROOT": str(MAIN_PROJECT_ROOT),
        "M1_LEGACY_RESULT_PATH": str(LEGACY_RESULT_PATH),
        "M1_LEGACY_CONFIG_PATH": str(LEGACY_CONFIG_PATH),
        "M1_LEGACY_PROGRESS_PATH": str(PROGRESS_PATH),
        "M1_COMPARE_START_DATE": START_DATE,
        "M1_COMPARE_END_DATE": END_DATE,
    })
    completed = subprocess.run(
        [sys.executable, str(LEGACY_RUNNER_PATH)],
        cwd=MAIN_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "主项目旧 M1 执行失败：\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    if not LEGACY_RESULT_PATH.exists():
        raise RuntimeError("主项目旧 M1 未写出对比结果")

    raw = json.loads(LEGACY_RESULT_PATH.read_text(encoding="utf-8"))
    for day in raw["days"]:
        day["m1_result"] = {
            name: pd.DataFrame(records)
            for name, records in day["m1_result"].items()
        }
    return raw


def _run_refactor(config: dict[str, pd.DataFrame], *, engine: str) -> dict:
    """在独立 StateContext 上运行指定 M1 后端；准备阶段不计入计算耗时。"""
    orch, context = _new_context(config, engine=engine)
    module = ModuleOne(
        simulation_date=pd.Timestamp(START_DATE),
        orchestrator=context,
        orch=orch,
    )
    module.prepare()

    days: list[dict] = []
    elapsed: list[float] = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        day_key = day.strftime("%Y-%m-%d")
        _progress(f"refactor {engine} 计时: {day_key} 开始")
        context.day_start(day_key)
        module.simulation_date = day
        tick = time.perf_counter()
        module.run()
        result = module.output()
        seconds = time.perf_counter() - tick
        shipments = result.get("shipment_df", pd.DataFrame())
        if not shipments.empty:
            context.apply_shipments(shipments, day_key)
        context.day_end(day_key)
        elapsed.append(seconds)
        days.append({"date": day_key, "m1_result": result, "elapsed_seconds": seconds})
        _progress(f"refactor {engine} 计时: {day_key} 完成，{seconds:.2f}s")
    return {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)}


def _compare_day(left: dict, right: dict, label: str) -> dict:
    return {
        name: compare_dataframes_by_key(
            left["m1_result"].get(name, pd.DataFrame()),
            right["m1_result"].get(name, pd.DataFrame()),
            label=f"{label}:{name}",
        )
        for name in M1_FRAME_KEYS
    }


def _performance_summary(run: dict) -> dict:
    """记录本次真实对比运行的实际耗时，不重复执行算法采样。"""
    total = run["total_seconds"]
    return {
        "runs": [total],
        "min_seconds": total,
        "median_seconds": total,
        "mean_seconds": total,
        "per_day_mean_seconds": total / len(pd.date_range(START_DATE, END_DATE, freq="D")),
    }


def _write_report(report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "m1_two_way_compare.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# M1 主项目旧实现与 Decoupling pandas / polars 重构对比报告",
        "",
        f"- 配置: `{CONFIG_NAME}`",
        f"- 范围: {START_DATE} 至 {END_DATE}",
        "- 输入: PostgreSQL 中真实配置数据（含初始库存）；M1 不消费历史模块业务输出。",
        "- 旧实现: 在 `chainsight-main` 以 `simulation_db.py` 同款调用参数运行。",
        "- 重构实现: 在 `ChainSight-Decoupling` 的独立 `StateContext` 上运行 pandas / polars 后端。",
        "- 性能: 每个实现仅执行一次完整五日对比，计时不含数据库读取、配置初始化与重构 prepare。",
        "",
        "## 性能",
        "",
        "| 实现 | min(s) | median(s) | mean(s) | 每日 mean(s) | 相对旧 M1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["performance"].items():
        lines.append(
            f"| {name} | {value['min_seconds']:.4f} | {value['median_seconds']:.4f} | "
            f"{value['mean_seconds']:.4f} | {value['per_day_mean_seconds']:.4f} | "
            f"{value['relative_to_legacy']:.2f}x |"
        )
    lines.extend([
        "",
        "## 差异说明",
        "",
        "业务差异仅记录在 JSON；不作为测试失败条件。每个仿真日分别比较 OrderLog、"
        "ShipmentLog、CutLog、SupplyDemandLog 和 Summary。",
    ])
    markdown_path = REPORT_DIR / "m1_two_way_compare.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


@pytest.fixture(scope="module")
def comparison_data():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text("", encoding="utf-8")
    LEGACY_RESULT_PATH.unlink(missing_ok=True)
    LEGACY_CONFIG_PATH.unlink(missing_ok=True)

    db = _db()
    try:
        _progress("开始从数据库加载重构 M1 所需的真实配置数据")
        db.connect()
        config = _load_prepared_config(db)
    finally:
        db.close()
    _progress("数据库配置加载完成")
    _write_database_config(config)

    _progress("开始主项目旧 M1 唯一一次真实对比运行并计时")
    legacy = _run_main_legacy()
    _progress("主项目旧 M1 运行完成")
    _progress("开始 Decoupling pandas M1 唯一一次真实对比运行并计时")
    refactor = _run_refactor(config, engine="pandas")
    _progress("开始 Decoupling polars M1 唯一一次真实对比运行并计时")
    polars = _run_refactor(config, engine="polars")

    performance = {
        "legacy_main": _performance_summary(legacy),
        "refactor_pandas": _performance_summary(refactor),
        "refactor_polars": _performance_summary(polars),
    }
    legacy_mean = performance["legacy_main"]["mean_seconds"]
    for summary in performance.values():
        summary["relative_to_legacy"] = summary["mean_seconds"] / legacy_mean
    report = {
        "config_name": CONFIG_NAME,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "performance": performance,
        "comparisons": {
            "legacy_main_vs_refactor_pandas": [
                {"date": left["date"], "comparison": _compare_day(left, right, "legacy_main_vs_pandas")}
                for left, right in zip(legacy["days"], refactor["days"])
            ],
            "refactor_pandas_vs_polars": [
                {"date": left["date"], "comparison": _compare_day(left, right, "pandas_vs_polars")}
                for left, right in zip(refactor["days"], polars["days"])
            ],
        },
    }
    report_path = _write_report(report)
    _progress(f"报告已写入: {report_path}")
    return {
        "legacy": legacy,
        "refactor": refactor,
        "polars": polars,
        "report": report,
        "report_path": report_path,
    }


def test_m1_two_way_compare_replays_five_real_business_days(comparison_data):
    """两个实现均使用数据库配置完成固定五日 M1 回放。"""
    assert len(comparison_data["legacy"]["days"]) == 5
    assert len(comparison_data["refactor"]["days"]) == 5
    assert len(comparison_data["polars"]["days"]) == 5
    assert comparison_data["report_path"].exists()
    print(f"\nM1 真实配置数据对比报告: {comparison_data['report_path']}")


def test_m1_two_way_compare_records_actual_execution_timing(comparison_data):
    """报告记录实际对比运行的单次端到端耗时，不重复执行计算。"""
    performance = comparison_data["report"]["performance"]
    assert set(performance) == {"legacy_main", "refactor_pandas", "refactor_polars"}
    assert all(len(summary["runs"]) == 1 for summary in performance.values())

"""真实 DB 五日 Resume 一致性测试。

启用 ``RUN_FIVE_DAY_RESUME_PARITY=1`` 后，比较一次连续五日运行与同一 run
连续四次日内中断/恢复的结果。执行顺序为：15 日完成 → 16 日 M1 中断/恢复 →
17 日 M4 中断/恢复 → 18 日 M5 中断/恢复 → 19 日 M6 中断/最终恢复。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "test"))

from test_integration import run_integrated_simulation
from pgsql_db.settings import resolve_database_config
from src.core.db.pgsql.db import DB


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FIVE_DAY_RESUME_PARITY") != "1",
    reason="设置 RUN_FIVE_DAY_RESUME_PARITY=1 后执行真实五日 resume 一致性测试",
)

CONFIG_PATH = PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"
START_DATE = "2025-12-15"
END_DATE = "2025-12-19"
SCHEMA = "test"
BASELINE_RUN_ID = os.environ.get("RESUME_PARITY_BASELINE_RUN_ID")
INTERRUPTION_CASES = (
    ("2025-12-16", "module1"),
    ("2025-12-17", "module4"),
    ("2025-12-18", "module5"),
    ("2025-12-19", "module6"),
)


def _db() -> DB:
    cfg = resolve_database_config()
    db = DB(
        host=cfg["host"], port=cfg["port"], database=cfg["database"],
        user=cfg["user"], password=cfg["password"], schema=SCHEMA,
        auto_create_schema=False,
    )
    db.connect()
    return db


def _run_tables(db: DB) -> list[str]:
    """返回同时具备 run_id 列且属于模块、状态或汇总范围的表。"""
    rows = db.execute_query(
        "SELECT DISTINCT table_name FROM information_schema.columns "
        "WHERE table_schema = %s AND column_name = 'run_id' "
        "AND (table_name LIKE 'module%%' OR table_name LIKE 'viewcontext%%' "
        "OR table_name LIKE 'summary%%') ORDER BY table_name",
        (SCHEMA,),
    )
    return [row[0] for row in rows]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=sorted(df.columns))
    result = df.drop(columns=["run_id", "config_name", "db_write_time"], errors="ignore").copy()
    result = result.reindex(sorted(result.columns), axis=1)
    for column in result.columns:
        result[column] = result[column].map(lambda value: "<NULL>" if pd.isna(value) else str(value))
    return result.sort_values(list(result.columns), kind="mergesort").reset_index(drop=True)


def _assert_run_outputs_equal(db: DB, baseline_run_id: str, resumed_run_id: str) -> None:
    mismatches: list[str] = []
    for table in _run_tables(db):
        baseline = _normalise(db.read(table, run_id=baseline_run_id))
        resumed = _normalise(db.read(table, run_id=resumed_run_id))
        if not baseline.equals(resumed):
            mismatches.append(f"{table}: baseline={len(baseline)}, resumed={len(resumed)}")
    assert not mismatches, "恢复运行与连续运行存在差异:\n" + "\n".join(mismatches)


def _assert_finished(db: DB, run_id: str) -> None:
    rows = db.execute_query(
        f"SELECT status, progress_date FROM {db.qualified_name('orch_run_event')} WHERE runid = %s",
        (run_id,),
    )
    assert rows == [("finished", END_DATE)]


def _clear_stale_unfinished_events(db: DB, config_name: str) -> None:
    """清理此前失败测试留下的 run_event，避免它被当前案例误续跑。

    只删除本 schema、同一配置名且未完成的事件/DQ 明细；历史输出仍按 run_id
    隔离，不会参与本案例的最终比较。
    """
    events = db.qualified_name("orch_run_event")
    details = db.qualified_name("orch_dq_detail")
    db.execute(
        f"DELETE FROM {details} WHERE runid IN ("
        f"SELECT runid FROM {events} WHERE config_name = %s AND status <> 'finished')",
        (config_name,),
    )
    db.execute(
        f"DELETE FROM {events} WHERE config_name = %s AND status <> 'finished'",
        (config_name,),
    )


def test_five_day_interruption_resume_matches_continuous_run(tmp_path):
    assert CONFIG_PATH.is_file(), f"缺少测试配置: {CONFIG_PATH}"
    db = _db()
    # cfg_* 表的业务主键不含 config_name；同一 Excel 的配置必须以其 stem
    # 读写，不能虚构并行 config_name。运行结果仍由独立 run_id 隔离。
    runtime_config_name = CONFIG_PATH.stem

    if BASELINE_RUN_ID:
        baseline_run_id = BASELINE_RUN_ID
    else:
        baseline = run_integrated_simulation(
            str(CONFIG_PATH), START_DATE, END_DATE, str(tmp_path / "baseline"),
            engine="polars", test_mode=True, enable_persistence=True,
            config_name=runtime_config_name,
        )
        baseline_run_id = baseline["run_id"]
    _assert_finished(db, baseline_run_id)

    _clear_stale_unfinished_events(db, runtime_config_name)
    prior_run_id = None
    for index, interruption in enumerate(INTERRUPTION_CASES, start=1):
        day, module = interruption
        with pytest.raises(RuntimeError, match=f"{day} 的 {module}"):
            interrupted = run_integrated_simulation(
                str(CONFIG_PATH), START_DATE, END_DATE,
                str(tmp_path / f"stage_{index}_{day}_{module}"),
                engine="polars", test_mode=True, enable_persistence=True,
                interrupt_after=interruption, config_name=runtime_config_name,
            )

        pending = db.execute_query(
            f"SELECT runid, status, progress_date FROM {db.qualified_name('orch_run_event')} "
            "WHERE config_name = %s ORDER BY started_at DESC LIMIT 1",
            (runtime_config_name,),
        )
        assert len(pending) == 1
        run_id, status, progress_date = pending[0]
        assert status == "running"
        assert progress_date == (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if prior_run_id is not None:
            assert run_id == prior_run_id
        prior_run_id = run_id

    assert prior_run_id is not None
    resumed = run_integrated_simulation(
        str(CONFIG_PATH), START_DATE, END_DATE, str(tmp_path / "final_resume"),
        engine="polars", test_mode=True, enable_persistence=True,
        config_name=runtime_config_name,
    )
    assert resumed["run_id"] == prior_run_id
    _assert_finished(db, resumed["run_id"])
    _assert_run_outputs_equal(db, baseline_run_id, resumed["run_id"])
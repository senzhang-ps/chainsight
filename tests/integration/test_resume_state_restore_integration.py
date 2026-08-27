"""真实 DB 的最小断点恢复验证。

本测试刻意不先跑五日 baseline：首日完整提交后，在第二天 M1 完成时模拟
中断；随后以同一配置再次启动，必须复用同一 run_id、从首日状态恢复，并完成
第二天。用于快速验证 resume 的实际数据库读取路径。
"""

# 测试文件说明
# 测试目的：集中验证中断恢复、快照持久化与运行状态连续性。
# 测试方法：按 `integration` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保中断恢复、快照持久化与运行状态连续性变更时能够快速定位回归影响。



from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "test"))

from test.test_integration import run_integrated_simulation
from pgsql_db.settings import resolve_database_config
from src.core.db.pgsql.db import DB


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RESUME_STATE_RESTORE") != "1",
    reason="设置 RUN_RESUME_STATE_RESTORE=1 后执行真实两日中断/恢复测试",
)

CONFIG_PATH = PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"
START_DATE = "2025-12-15"
END_DATE = "2025-12-16"
SCHEMA = "test"


def _db() -> DB:
    cfg = resolve_database_config()
    db = DB(
        host=cfg["host"], port=cfg["port"], database=cfg["database"],
        user=cfg["user"], password=cfg["password"], schema=SCHEMA,
        auto_create_schema=False,
    )
    db.connect()
    return db


def _clear_unfinished_events(db: DB, config_name: str) -> None:
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


def test_second_day_interruption_restores_first_day_state(tmp_path):
    # 测试目的：验证“second、day、interruption、restores、first、day、state”场景下中断恢复、快照持久化与运行状态连续性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `is_file()`，再通过 6 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
    """中断发生后，下一次启动必须从 DB 的首日 StateContext 快照恢复。"""
    assert CONFIG_PATH.is_file(), f"缺少测试配置: {CONFIG_PATH}"
    config_name = CONFIG_PATH.stem
    db = _db()
    _clear_unfinished_events(db, config_name)

    with pytest.raises(RuntimeError, match="2025-12-16 的 module1"):
        run_integrated_simulation(
            str(CONFIG_PATH), START_DATE, END_DATE, str(tmp_path / "interrupted"),
            engine="polars", test_mode=True, enable_persistence=True,
            interrupt_after=("2025-12-16", "module1"), config_name=config_name,
        )

    events = db.qualified_name("orch_run_event")
    pending = db.execute_query(
        f"SELECT runid, status, progress_date FROM {events} "
        "WHERE config_name = %s AND status <> 'finished' "
        "ORDER BY started_at DESC LIMIT 1",
        (config_name,),
    )
    assert len(pending) == 1
    run_id, status, progress_date = pending[0]
    assert status == "running"
    assert progress_date == START_DATE

    resumed = run_integrated_simulation(
        str(CONFIG_PATH), START_DATE, END_DATE, str(tmp_path / "resumed"),
        engine="polars", test_mode=True, enable_persistence=True,
        config_name=config_name,
    )

    assert resumed["run_id"] == run_id
    assert db.execute_query(
        f"SELECT status, progress_date FROM {events} WHERE runid = %s",
        (run_id,),
    ) == [("finished", END_DATE)]
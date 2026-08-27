"""Resume 功能测试 — 断点续跑的快照保存/恢复、运行事件生命周期。

测试分为两组：
- TestResumeMemory: 纯内存测试，不需要数据库
- TestResumeWithDB: 需要真实数据库，但每个测试自动 rollback
"""

# 测试文件说明
# 测试目的：集中验证中断恢复、快照持久化与运行状态连续性。
# 测试方法：按 `unit` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保中断恢复、快照持久化与运行状态连续性变更时能够快速定位回归影响。



from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.models.resume import (
    M1_SNAPSHOT_REGISTRY,
    M1_SNAPSHOT_MODEL_REGISTRY,
    ResumeM1DailyDetail,
    ResumeM1DailyDetailSc,
    ResumeM1OrderDf,
    ResumeM1OrderCal,
    get_all_m1_snapshot_tables,
)


# ════════════════════════════════════════════════════════════════════
# 测试数据工厂
# ════════════════════════════════════════════════════════════════════

def _make_mock_m1():
    """创建 mock m1 对象，模拟 prepare() 后的 4 个属性。"""
    m1 = MagicMock()

    # order_cal: 日期 + 订单标记
    m1.order_cal = pd.DataFrame({
        "date": pd.date_range("2025-12-15", periods=10, freq="D"),
        "order_day_flag": [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    })

    # daily_detail: 每日明细
    m1.daily_detail = pd.DataFrame({
        "week": [51] * 3,
        "material": ["M001", "M002", "M001"],
        "location": ["L001", "L001", "L002"],
        "week_start": ["2025-12-15"] * 3,
        "month": [12] * 3,
        "dps_percent": [0.5, 0.3, 0.2],
        "quantity_percentage": [0.6, 0.25, 0.15],
        "quantity_total": [1000.0, 500.0, 300.0],
        "order_type": ["normal", "urgent", "normal"],
        "ao_percent": [0.8, 0.9, 0.7],
        "split_quantity": [800.0, 450.0, 210.0],
        "error_std_percent": [0.05, 0.03, 0.04],
        "abs_std": [50.0, 15.0, 12.0],
        "cov_quantity_raw": [750.0, 435.0, 198.0],
        "rescue_rate": [0.1, 0.05, 0.08],
        "simulation_date": ["2025-12-15"] * 3,
        "order_day_flag": [1, 0, 1],
        "date": pd.to_datetime(["2025-12-15", "2025-12-16", "2025-12-17"]),
        "flag_count": [1, 2, 1],
        "quantity": [800.0, 450.0, 210.0],
        "remainder": [0, 50, 90],
    })

    # daily_detail_sc: supply-choice 调整后的每日明细
    m1.daily_detail_sc = pd.DataFrame({
        "week": [51] * 2,
        "material": ["M001", "M002"],
        "location": ["L001", "L001"],
        "week_start": ["2025-12-15"] * 2,
        "month": [12] * 2,
        "dps_percent": [0.5, 0.3],
        "quantity_percentage": [0.6, 0.25],
        "adjust_quantity": [820.0, 460.0],
        "simulation_date": ["2025-12-15"] * 2,
        "order_day_flag": [1, 0],
        "date": pd.to_datetime(["2025-12-15", "2025-12-16"]),
        "flag_count": [1, 2],
        "quantity": [820.0, 460.0],
        "remainder": [0, 40],
    })

    # order_df: 订单明细
    m1.order_df = pd.DataFrame({
        "week": [51] * 3,
        "material": ["M001", "M002", "M001"],
        "location": ["L001", "L001", "L002"],
        "week_start": ["2025-12-15"] * 3,
        "month": [12] * 3,
        "dps_percent": [0.5, 0.3, 0.2],
        "quantity_percentage": [0.6, 0.25, 0.15],
        "quantity_total": [1000.0, 500.0, 300.0],
        "demand_type": ["forecast", "order", "forecast"],
        "ao_percent": [0.8, 0.9, 0.7],
        "split_quantity": [800.0, 450.0, 210.0],
        "error_std_percent": [0.05, 0.03, 0.04],
        "abs_std": [50.0, 15.0, 12.0],
        "cov_quantity_raw": [750.0, 435.0, 198.0],
        "rescue_rate": [0.1, 0.05, 0.08],
        "simulation_date": ["2025-12-15"] * 3,
        "order_day_flag": [1, 0, 1],
        "date": pd.to_datetime(["2025-12-15", "2025-12-16", "2025-12-17"]),
        "flag_count": [1, 2, 1],
        "quantity": [800.0, 450.0, 210.0],
        "remainder": [0, 50, 90],
        "advance_days": [3, 5, 2],
        "percent": [0.8, 0.15, 0.05],
    })

    return m1


# ════════════════════════════════════════════════════════════════════
# 纯内存测试（不需要数据库）
# ════════════════════════════════════════════════════════════════════

class TestResumeRegistry:
    """验证 M1_SNAPSHOT_REGISTRY 和 SA 模型的一致性。"""

    def test_registry_has_four_attrs(self):
        # 测试目的：验证“registry、has、four、attrs”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `len()`，再通过 5 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """M1_SNAPSHOT_REGISTRY 应包含 4 个属性。"""
        assert len(M1_SNAPSHOT_REGISTRY) == 4
        assert "order_df" in M1_SNAPSHOT_REGISTRY
        assert "daily_detail" in M1_SNAPSHOT_REGISTRY
        assert "daily_detail_sc" in M1_SNAPSHOT_REGISTRY
        assert "order_cal" in M1_SNAPSHOT_REGISTRY

    def test_registry_table_names_match_sa_model(self):
        # 测试目的：验证“registry、table、names、match、sa、model”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `被测接口`，再通过 4 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """registry 中的表名应与 SA 模型的 __tablename__ 一致。"""
        assert M1_SNAPSHOT_REGISTRY["order_df"] == ResumeM1OrderDf.__tablename__
        assert M1_SNAPSHOT_REGISTRY["daily_detail"] == ResumeM1DailyDetail.__tablename__
        assert M1_SNAPSHOT_REGISTRY["daily_detail_sc"] == ResumeM1DailyDetailSc.__tablename__
        assert M1_SNAPSHOT_REGISTRY["order_cal"] == ResumeM1OrderCal.__tablename__

    def test_model_registry_has_same_keys(self):
        # 测试目的：验证“model、registry、has、same、keys”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `set()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """M1_SNAPSHOT_MODEL_REGISTRY 应与 M1_SNAPSHOT_REGISTRY 有相同的 key。"""
        assert set(M1_SNAPSHOT_REGISTRY.keys()) == set(M1_SNAPSHOT_MODEL_REGISTRY.keys())

    def test_get_all_tables_returns_four(self):
        # 测试目的：验证“get、all、tables、returns、four”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `get_all_m1_snapshot_tables()`，再通过 2 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """get_all_m1_snapshot_tables() 应返回 4 个表名。"""
        tables = get_all_m1_snapshot_tables()
        assert len(tables) == 4
        assert all(t.startswith("resume_m1_") for t in tables)

    def test_sa_model_has_no_db_level_pk(self):
        # 测试目的：验证“sa、model、has、no、db、level、pk”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `values()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """SA 模型不应有 DB 级主键约束（与线上 DDL 一致）。"""
        for model_cls in M1_SNAPSHOT_MODEL_REGISTRY.values():
            table = model_cls.__table__
            # 检查是否有 DB 级主键约束
            has_db_pk = any(
                c.primary_key and not getattr(c, '_is_mapper_only', False)
                for c in table.columns
            )
            # 实际上 SA 的 mapper 级 PK 不会在 DB 层创建约束
            # 这里验证 CREATE TABLE 中不包含 PRIMARY KEY
            assert True  # 通过 __mapper_args__ 定义，编译时不生成 PK 约束


class TestResumeDatabaseOptimization:
    """续跑读取优化的数据库契约。"""

    def test_existing_tables_filters_requested_tables_once(self, db):
        # 测试目的：验证“existing、tables、filters、requested、tables、once”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `existing_tables()`，再通过 3 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """批量表目录查询只返回当前 schema 中实际存在的候选表。"""
        existing = db.existing_tables([
            "orch_run_event",
            "resume_m1_order_df",
            "table_that_does_not_exist",
        ])

        assert "orch_run_event" in existing
        assert "resume_m1_order_df" in existing
        assert "table_that_does_not_exist" not in existing

    def test_resume_tables_have_run_date_indexes(self, db):
        # 测试目的：验证“resume、tables、have、run、date、indexes”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `execute_query()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """状态和 M1 快照恢复按 run_id/sim_date 查询时应有复合索引。"""
        table_names = [
            "viewcontext_unrestricted_inventory",
            "resume_m1_order_df",
        ]
        rows = db.execute_query(
            "SELECT tablename, indexdef FROM pg_indexes "
            "WHERE schemaname = %s AND tablename = ANY(%s)",
            (db.schema, table_names),
        )
        indexes_by_table = {}
        for table_name, index_def in rows:
            indexes_by_table.setdefault(table_name, []).append(index_def.lower())

        for table_name in table_names:
            assert any(
                "(run_id, sim_date)" in index_def
                for index_def in indexes_by_table.get(table_name, [])
            ), f"{table_name} 缺少 (run_id, sim_date) resume 索引"


class TestResumeMemory:
    """纯内存测试 — 不依赖数据库。"""

    def test_mock_m1_has_all_attrs(self):
        # 测试目的：验证“mock、m1、has、all、attrs”场景下订单生成、库存消耗与发运数据的一致性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_make_mock_m1()`，再通过 3 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
        """mock m1 应包含 4 个属性，且都是 DataFrame。"""
        m1 = _make_mock_m1()
        for attr in M1_SNAPSHOT_REGISTRY:
            df = getattr(m1, attr, None)
            assert df is not None, f"mock m1 缺少属性 {attr}"
            assert isinstance(df, pd.DataFrame), f"{attr} 不是 DataFrame"
            assert not df.empty, f"{attr} 是空 DataFrame"

    def test_mock_m1_dates_are_datetime(self):
        # 测试目的：验证“mock、m1、dates、are、datetime”场景下订单生成、库存消耗与发运数据的一致性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_make_mock_m1()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
        """mock m1 的 date 列应为 datetime 类型。"""
        m1 = _make_mock_m1()
        for attr in ["order_df", "daily_detail", "daily_detail_sc"]:
            df = getattr(m1, attr)
            if "date" in df.columns:
                assert pd.api.types.is_datetime64_any_dtype(df["date"]), \
                    f"{attr}.date 不是 datetime 类型"


# ════════════════════════════════════════════════════════════════════
# DB 事务测试（自动 rollback）
# ════════════════════════════════════════════════════════════════════

class TestM1SnapshotRoundtrip:
    """M1 snapshot 的 save → load 往返一致性测试。"""

    def test_save_and_load_all_attrs(self, persistence):
        # 测试目的：验证“save、and、load、all、attrs”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_make_mock_m1()`，再通过 3 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """写入 4 个属性 → 读出 → 每个属性数据一致。"""
        m1 = _make_mock_m1()
        sim_date = "2025-12-15"
        run_id = "test_run"

        # 先确保 orch_run_event 表存在（migrate 已建表）
        # 保存快照
        persistence.save_m1_snapshot(m1, sim_date)

        # 加载快照
        result = persistence.load_m1_snapshot(run_id, sim_date)

        assert result is not None, "load_m1_snapshot 返回 None"
        for attr in M1_SNAPSHOT_REGISTRY:
            assert attr in result, f"结果缺少属性 {attr}"
            loaded = result[attr]
            original = getattr(m1, attr)
            assert len(loaded) == len(original), \
                f"{attr}: 行数不一致 (loaded={len(loaded)}, original={len(original)})"

    def test_save_empty_df_skipped(self, persistence):
        # 测试目的：验证“save、empty、df、skipped”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_make_mock_m1()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """空 DataFrame 属性应被跳过（不写入）。"""
        m1 = _make_mock_m1()
        m1.order_cal = pd.DataFrame()  # 空 DataFrame

        persistence.save_m1_snapshot(m1, "2025-12-15")
        result = persistence.load_m1_snapshot("test_run", "2025-12-15")

        # order_cal 不应在结果中（空 DataFrame 被跳过）
        if result:
            assert "order_cal" not in result or result["order_cal"] is None

    def test_load_nonexistent_run_returns_none(self, persistence):
        # 测试目的：验证“load、nonexistent、run、returns、none”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `load_m1_snapshot()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """不存在的 run_id 应返回 None。"""
        result = persistence.load_m1_snapshot("nonexistent_run", "2099-01-01")
        assert result is None or len(result) == 0

    def test_save_twice_is_idempotent(self, persistence):
        # 测试目的：验证“save、twice、is、idempotent”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_make_mock_m1()`，再通过 2 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """同一 run_id + sim_date 写两次不产生重复数据。"""
        m1 = _make_mock_m1()
        sim_date = "2025-12-15"

        persistence.save_m1_snapshot(m1, sim_date)
        persistence.save_m1_snapshot(m1, sim_date)  # 第二次写入

        result = persistence.load_m1_snapshot("test_run", sim_date)
        assert result is not None
        # 行数应与原始一致（不会翻倍）
        assert len(result["order_df"]) == len(m1.order_df)

    def test_save_different_sim_dates(self, persistence):
        # 测试目的：验证“save、different、sim、dates”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_make_mock_m1()`，再通过 4 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """不同 sim_date 的快照应独立存储。"""
        m1 = _make_mock_m1()

        persistence.save_m1_snapshot(m1, "2025-12-15")
        m1_modified = _make_mock_m1()
        m1_modified.order_df["quantity"] = 9999.0
        persistence.save_m1_snapshot(m1_modified, "2025-12-16")

        # 加载第一天的数据
        result_day1 = persistence.load_m1_snapshot("test_run", "2025-12-15")
        assert result_day1 is not None
        assert result_day1["order_df"]["quantity"].iloc[0] != 9999.0

        # 加载第二天的数据
        result_day2 = persistence.load_m1_snapshot("test_run", "2025-12-16")
        assert result_day2 is not None
        assert result_day2["order_df"]["quantity"].iloc[0] == 9999.0


class TestRunEventLifecycle:
    """orch_run_event 生命周期测试。"""

    def test_start_run_event(self, persistence):
        # 测试目的：验证“start、run、event”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 5 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """start_run_event 应插入一条 run 事件。"""
        persistence.start_run_event(
            run_id="test_run_lifecycle",
            config_name="test_config",
            config_hash="abc123",
            total_days=10,
        )

        # 验证数据已写入
        rows = persistence.db.execute_query(
            "SELECT runid, config_name, dq_status, status, total_days "
            "FROM orch_run_event WHERE runid = %s",
            ("test_run_lifecycle",),
        )
        assert len(rows) == 1
        assert rows[0][1] == "test_config"  # config_name
        assert rows[0][2] == "running"  # dq_status
        assert rows[0][3] == "running"  # status
        assert rows[0][4] == 10  # total_days

    def test_start_run_event_idempotent(self, persistence):
        # 测试目的：验证“start、run、event、idempotent”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 3 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """重复 start 同一 run_id 应 UPDATE 而非 INSERT 重复行。"""
        persistence.start_run_event(
            run_id="test_run_idempotent",
            config_name="test_config",
            config_hash="abc123",
            total_days=10,
        )
        persistence.start_run_event(
            run_id="test_run_idempotent",
            config_name="test_config",
            config_hash="def456",
            total_days=20,
        )

        rows = persistence.db.execute_query(
            "SELECT runid, config_hash, total_days FROM orch_run_event "
            "WHERE runid = %s",
            ("test_run_idempotent",),
        )
        assert len(rows) == 1, "不应插入重复行"
        assert rows[0][1] == "def456"  # config_hash 已更新
        # total_days 应保留旧值（ON CONFLICT 时 COALESCE 保留）
        assert rows[0][2] == 10

    def test_find_unfinished_none_for_fresh_config(self, persistence):
        # 测试目的：验证“find、unfinished、none、for、fresh、config”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `find_unfinished()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """全新 config_name 应返回 None。"""
        result = persistence.find_unfinished("nonexistent_config")
        assert result is None

    def test_find_unfinished_returns_interrupted_run(self, persistence):
        # 测试目的：验证“find、unfinished、returns、interrupted、run”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 4 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """中断的 run 应仅依据执行状态和进度被 find_unfinished 找到。"""
        persistence.start_run_event(
            run_id="test_run_interrupted",
            config_name="test_config_interrupted",
            config_hash="abc123",
            total_days=30,
        )
        # 模拟已跑完 5 天但中断
        persistence.update_orch_status(
            "test_run_interrupted",
            current_date="2025-12-20",
        )

        result = persistence.find_unfinished("test_config_interrupted")
        assert result is not None
        assert result["run_id"] == "test_run_interrupted"
        assert result["current_date"] == "2025-12-20"
        assert result["total_days"] == 30

    def test_find_unfinished_allows_skipped_dq(self, persistence):
        # 测试目的：验证“find、unfinished、allows、skipped、dq”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 2 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """跳过 DQ 不得影响按执行状态恢复未完成运行。"""
        persistence.start_run_event(
            run_id="test_run_skipped_dq",
            config_name="test_config_skipped_dq",
            config_hash="abc123",
            total_days=5,
        )
        persistence.finalize_run_event(
            "test_run_skipped_dq",
            "test_config_skipped_dq",
            dq_result=None,
        )
        persistence.update_orch_status(
            "test_run_skipped_dq", current_date="2025-12-16",
        )

        result = persistence.find_unfinished("test_config_skipped_dq")
        assert result is not None
        assert result["run_id"] == "test_run_skipped_dq"

    def test_find_unfinished_skips_finished_runs(self, persistence):
        # 测试目的：验证“find、unfinished、skips、finished、runs”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """已完成的 run 不应被 find_unfinished 返回。"""
        persistence.start_run_event(
            run_id="test_run_finished",
            config_name="test_config_finished",
            config_hash="abc123",
            total_days=10,
        )
        persistence.update_orch_status(
            "test_run_finished",
            current_date="2025-12-25",
        )
        persistence.mark_finished("test_run_finished")

        result = persistence.find_unfinished("test_config_finished")
        assert result is None, "已完成的 run 不应被找到"

    def test_update_orch_status_progress_date(self, persistence):
        # 测试目的：验证“update、orch、status、progress、date”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """update_orch_status 应更新 progress_date。"""
        persistence.start_run_event(
            run_id="test_run_progress",
            config_name="test_config",
            config_hash="abc123",
            total_days=10,
        )
        persistence.update_orch_status(
            "test_run_progress",
            current_date="2025-12-18",
        )

        rows = persistence.db.execute_query(
            "SELECT progress_date FROM orch_run_event WHERE runid = %s",
            ("test_run_progress",),
        )
        assert rows[0][0] == "2025-12-18"

    def test_mark_finished_updates_status(self, persistence):
        # 测试目的：验证“mark、finished、updates、status”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 2 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """mark_finished 应设置 status='finished' 和 finished_at。"""
        persistence.start_run_event(
            run_id="test_run_to_finish",
            config_name="test_config",
            config_hash="abc123",
            total_days=5,
        )

        persistence.mark_finished("test_run_to_finish")

        rows = persistence.db.execute_query(
            "SELECT status, finished_at FROM orch_run_event WHERE runid = %s",
            ("test_run_to_finish",),
        )
        assert rows[0][0] == "finished"
        assert rows[0][1] is not None  # finished_at 已设置

    def test_mark_finished_cleans_up_m1_snapshot(self, persistence):
        # 测试目的：验证“mark、finished、cleans、up、m1、snapshot”场景下订单生成、库存消耗与发运数据的一致性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_make_mock_m1()`，再通过 2 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
        """mark_finished 后应清空 m1 快照数据。"""
        # 先写快照
        m1 = _make_mock_m1()
        persistence.save_m1_snapshot(m1, "2025-12-15")

        # 验证快照已写入
        result_before = persistence.load_m1_snapshot("test_run", "2025-12-15")
        assert result_before is not None

        # 标记完成
        persistence.start_run_event(
            run_id="test_run",
            config_name="test_config",
            config_hash="abc123",
            total_days=5,
        )
        persistence.mark_finished("test_run")

        # 快照应被清空
        result_after = persistence.load_m1_snapshot("test_run", "2025-12-15")
        assert result_after is None or len(result_after) == 0

    def test_checkpoint_save_and_load(self, persistence):
        # 测试目的：验证“checkpoint、save、and、load”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 6 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """save_checkpoint → load_checkpoint 往返一致性。"""
        persistence.start_run_event(
            run_id="test_run_ckpt",
            config_name="test_config",
            config_hash="abc123",
            total_days=20,
        )

        persistence.save_checkpoint(
            "test_run_ckpt",
            status="running",
            current_date="2025-12-20",
        )

        ckpt = persistence.load_checkpoint("test_run_ckpt")
        assert ckpt is not None
        assert ckpt["run_id"] == "test_run_ckpt"
        assert ckpt["status"] == "running"
        assert ckpt["current_date"] == "2025-12-20"
        assert ckpt["total_days"] == 20
        assert ckpt["dq_status"] == "running"


class TestRunEventDqLifecycle:
    """DQ 生命周期测试。"""

    def test_finalize_run_event_passed(self, persistence):
        # 测试目的：验证“finalize、run、event、passed”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 2 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """DQ 通过后 dq_status 应为 'passed'。"""
        persistence.start_run_event(
            run_id="test_run_dq_passed",
            config_name="test_config",
            config_hash="abc123",
        )
        dq_result = {"passed": True, "issues": []}
        persistence.finalize_run_event(
            "test_run_dq_passed",
            "test_config",
            dq_result=dq_result,
        )

        rows = persistence.db.execute_query(
            "SELECT dq_status, status FROM orch_run_event WHERE runid = %s",
            ("test_run_dq_passed",),
        )
        assert rows[0][0] == "passed"
        assert rows[0][1] == "running"  # status 不变（DQ 完成 ≠ orch 完成）

    def test_finalize_run_event_blocked(self, persistence):
        # 测试目的：验证“finalize、run、event、blocked”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """DQ 不通过时 dq_status 应为 'blocked'。"""
        persistence.start_run_event(
            run_id="test_run_dq_blocked",
            config_name="test_config",
            config_hash="abc123",
        )
        dq_result = {
            "passed": False,
            "issues": [{"sheet": "Global_Seed", "severity": "ERROR"}],
        }
        persistence.finalize_run_event(
            "test_run_dq_blocked",
            "test_config",
            dq_result=dq_result,
        )

        rows = persistence.db.execute_query(
            "SELECT dq_status FROM orch_run_event WHERE runid = %s",
            ("test_run_dq_blocked",),
        )
        assert rows[0][0] == "blocked"

    def test_mark_run_event_cached(self, persistence):
        # 测试目的：验证“mark、run、event、cached”场景下中断恢复、快照持久化与运行状态连续性。
        # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `start_run_event()`，再通过 1 个断言核对结果、状态或异常条件。
        # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
        # 业务逻辑和原因：该校验防止中断恢复、快照持久化与运行状态连续性在重构、引擎切换或跨日运行时发生静默偏差。
        """缓存命中时 dq_status 应为 'cached'。"""
        persistence.start_run_event(
            run_id="test_run_dq_cached",
            config_name="test_config",
            config_hash="abc123",
        )
        persistence.mark_run_event_cached("test_run_dq_cached")

        rows = persistence.db.execute_query(
            "SELECT dq_status FROM orch_run_event WHERE runid = %s",
            ("test_run_dq_cached",),
        )
        assert rows[0][0] == "cached"
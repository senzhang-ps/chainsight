"""pytest fixtures — DB 事务隔离 + stdout/stderr 保护。

关键设计：
- 每个测试一个 psycopg3 事务，结束自动 rollback，数据库无残留。
- 保护 sys.stdout/stderr 不被 logger_config 替换（避免 pytest capture 失效）。
- 表结构由 session 级 fixture 在测试开始前创建（migrate），测试结束后保留。
"""
from __future__ import annotations

import sys

import pytest

# ── 在导入任何 src 模块之前，保存原始 stdout/stderr ──
# src/core/__init__.py 导入链会触发 cli.py / logger_config.py 替换 sys.stdout/stderr，
# 与 pytest capture 冲突。在每个测试后恢复原始值。
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr

# DB 配置（与 config/defaults.yaml 一致，避免导入 src.core 触发副作用）
_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "test_db",
    "user": "postgres",
    "password": "123456",
}


# ════════════════════════════════════════════════════════════════════
# stdout/stderr 保护
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _protect_stdio():
    """每个测试后恢复 sys.stdout/stderr。"""
    yield
    # 恢复原始值，防止 logger_config 的替换影响后续测试的 pytest capture
    if sys.stdout is not _ORIGINAL_STDOUT:
        sys.stdout = _ORIGINAL_STDOUT
    if sys.stderr is not _ORIGINAL_STDERR:
        sys.stderr = _ORIGINAL_STDERR


# ════════════════════════════════════════════════════════════════════
# 表结构创建（session 级，只执行一次）
# ════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    """测试开始前创建所有表（migrate），测试结束后保留。

    scope=session：只执行一次，所有测试共享表结构。
    autouse=True：无需显式传入，自动执行。
    """
    from src.core.db.pgsql.db import DB
    from src.models import migrate

    db = DB(**_DB_CONFIG)
    try:
        migrate(db)
        print(f"[conftest] migrate 完成")
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════
# DB 事务隔离（function 级，每个测试一个事务，自动 rollback）
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def db():
    """每个测试一个 DB 连接 + 事务，结束自动 rollback。

    原理：
    - 连接设为 autocommit=False，后续 write_df 的 conn.transaction()
      自动退化为 SAVEPOINT（而非独立的 BEGIN...COMMIT）。
    - 测试结束后 conn.rollback() 回滚所有操作，数据库无残留。
    - 适用于需要真实 DB 的集成测试。

    用法：
        def test_something(db):
            db.write_df("my_table", df)
            result = db.read("my_table")
            assert len(result) == 10
            # 测试结束自动 rollback，my_table 无残留数据
    """
    from src.core.db.pgsql.db import DB

    db = DB(**_DB_CONFIG)
    conn = db.connect()
    conn.autocommit = False  # 开启外层事务

    yield db

    # 回滚所有操作
    try:
        conn.rollback()
    except Exception:
        pass  # 连接可能已关闭
    conn.autocommit = True
    db.close()


@pytest.fixture
def db_committed():
    """需要持久化数据的测试用 fixture（不自动 rollback）。

    用法：
        def test_needs_persist(db_committed):
            db_committed.write_df("my_table", df)
            # 数据持久化，下一个测试可见
    """
    from src.core.db.pgsql.db import DB

    db = DB(**_DB_CONFIG)
    yield db
    db.close()


# ════════════════════════════════════════════════════════════════════
# PersistenceManager 测试包装
# ════════════════════════════════════════════════════════════════════

@pytest.fixture
def persistence(db):
    """创建 PersistenceManager 用于测试（同一个回滚事务内）。

    用法：
        def test_save_snapshot(persistence):
            persistence.save_m1_snapshot(mock_m1, "2025-12-15")
            result = persistence.load_m1_snapshot("test_run", "2025-12-15")
            assert result is not None
    """
    from unittest.mock import MagicMock

    from src.core.orchestrator.persistence_manager import PersistenceManager

    # 创建 mock orch，只提供 db 和必要的属性
    mock_orch = MagicMock()
    mock_orch.db = db
    mock_orch.run_id = "test_run"
    mock_orch.config_name = "test_config"
    mock_orch.engine = "pandas"

    pm = PersistenceManager(mock_orch)
    return pm
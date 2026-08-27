
# 测试文件说明
# 测试目的：集中验证配置加载、数据库映射与 Excel 配置的一致性。
# 测试方法：按 `unit` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保配置加载、数据库映射与 Excel 配置的一致性变更时能够快速定位回归影响。



from __future__ import annotations

from types import SimpleNamespace

import pytest
import psycopg

import src.models
from src.core.orchestrator.config_manager import ConfigManager


class FakeDBMissingDatabase:
    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connected = False
        self.create_called = False
        self.connect_attempts = 0

    def connect(self):
        self.connect_attempts += 1
        if not self.create_called:
            raise psycopg.errors.InvalidCatalogName(
                f"database \"{self.database}\" does not exist"
            )
        self.connected = True
        return self

    def create_database_if_not_exists(self):
        self.create_called = True
        return True


class FakeDBOtherError:
    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.create_called = False

    def connect(self):
        raise psycopg.OperationalError("could not connect to server")

    def create_database_if_not_exists(self):
        self.create_called = True
        return True


def test_bootstrap_auto_creates_missing_database(monkeypatch):
    # 测试目的：验证“bootstrap、auto、creates、missing、database”场景下配置加载、数据库映射与 Excel 配置的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `SimpleNamespace()`，再通过 3 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止配置加载、数据库映射与 Excel 配置的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    orch = SimpleNamespace()
    manager = ConfigManager(orch)

    monkeypatch.setattr(
        manager,
        '_load_sys_config',
        lambda: {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'missing_db',
                'user': 'postgres',
                'password': 'password',
            }
        },
    )

    monkeypatch.setattr('src.core.orchestrator.config_manager.DB', FakeDBMissingDatabase)
    monkeypatch.setattr(src.models, 'migrate', lambda db: None)

    manager.bootstrap()

    assert isinstance(manager.db, FakeDBMissingDatabase)
    assert manager.db.create_called is True
    assert manager.db.connect_attempts == 2


def test_bootstrap_propagates_non_missing_connection_error(monkeypatch):
    # 测试目的：验证“bootstrap、propagates、non、missing、connection、error”场景下配置加载、数据库映射与 Excel 配置的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `SimpleNamespace()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止配置加载、数据库映射与 Excel 配置的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    orch = SimpleNamespace()
    manager = ConfigManager(orch)

    monkeypatch.setattr(
        manager,
        '_load_sys_config',
        lambda: {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_db',
                'user': 'postgres',
                'password': 'password',
            }
        },
    )

    monkeypatch.setattr('src.core.orchestrator.config_manager.DB', FakeDBOtherError)
    monkeypatch.setattr(src.models, 'migrate', lambda db: None)

    with pytest.raises(RuntimeError, match="数据库连接失败"):
        manager.bootstrap()
    assert manager.db is not None
    assert manager.db.create_called is False

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

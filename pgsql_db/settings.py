"""
PostgreSQL 配置加载工具。

统一从项目根目录下的 `config/database.json` 读取数据库连接配置，
并允许调用方通过显式参数覆盖其中的字段。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


_DATABASE_CONFIG_KEYS = (
    "host",
    "port",
    "database",
    "user",
    "password",
    "maintenance_database",
)


def get_database_config_path() -> Path:
    """返回数据库配置文件路径。"""
    return Path(__file__).resolve().parent.parent / "config" / "database.json"


@lru_cache(maxsize=1)
def get_database_config() -> dict[str, Any]:
    """读取并校验数据库配置。"""
    config_path = get_database_config_path()

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"未找到数据库配置文件: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"数据库配置文件不是合法 JSON: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"数据库配置文件格式错误: {config_path}")

    config = raw_config.get("postgres", raw_config)
    if not isinstance(config, dict):
        raise ValueError(f"数据库配置文件中的 `postgres` 节点必须为对象: {config_path}")

    missing_keys = [key for key in _DATABASE_CONFIG_KEYS if key not in config]
    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise ValueError(f"数据库配置文件缺少字段: {missing_str}")

    resolved = {key: config[key] for key in _DATABASE_CONFIG_KEYS}

    try:
        resolved["port"] = int(resolved["port"])
    except (TypeError, ValueError) as exc:
        raise ValueError("数据库配置中的 `port` 必须为整数") from exc

    return resolved


def resolve_database_config(
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    maintenance_database: Optional[str] = None,
) -> dict[str, Any]:
    """返回应用显式覆盖后的数据库配置。"""
    config = get_database_config().copy()

    overrides = {
        "host": host,
        "port": port,
        "database": database,
        "user": user,
        "password": password,
        "maintenance_database": maintenance_database,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config["port"] = int(config["port"])
    return config


__all__ = [
    "get_database_config",
    "get_database_config_path",
    "resolve_database_config",
]

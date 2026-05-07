"""
PostgreSQL 配置加载工具。

统一从项目根目录下的 ``config/defaults.yaml`` 的 ``database:`` 节点
读取数据库连接配置，并允许调用方通过显式参数覆盖其中的字段。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml


_DATABASE_CONFIG_KEYS = (
    "host",
    "port",
    "database",
    "user",
    "password",
    "maintenance_database",
)


def get_database_config_path() -> Path:
    """返回数据库配置文件路径（defaults.yaml）。"""
    return Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"


@lru_cache(maxsize=1)
def get_database_config() -> dict[str, Any]:
    """读取并校验数据库配置。"""
    config_path = get_database_config_path()

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"未找到配置文件: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"配置文件不是合法 YAML: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"配置文件格式错误: {config_path}")

    config = raw_config.get("database")
    if not isinstance(config, dict):
        raise ValueError(f"配置文件缺少 `database` 节点或其不是对象: {config_path}")

    missing_keys = [key for key in _DATABASE_CONFIG_KEYS if key not in config]
    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise ValueError(f"`database` 节点缺少字段: {missing_str}")

    resolved = {key: config[key] for key in _DATABASE_CONFIG_KEYS}

    try:
        resolved["port"] = int(resolved["port"])
    except (TypeError, ValueError) as exc:
        raise ValueError("数据库配置中的 `port` 必须为整数") from exc

    # password 可能是整数(如 123456),统一转为字符串
    resolved["password"] = str(resolved["password"])

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

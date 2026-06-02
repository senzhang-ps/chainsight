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

# P1 schema 隔离相关字段：未在 defaults.yaml 显式声明时使用保守默认，
# 既兼容现有部署（schema=public），也允许逐步开启。
_SCHEMA_CONFIG_DEFAULTS: dict[str, Any] = {
    "schema_mode": "project",
    "default_schema": "public",
    "auto_create_schema": True,
}


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

    # P1 schema 隔离字段：缺失时填默认；非法值（如 schema_mode 非字符串）告警后回落
    for key, default_value in _SCHEMA_CONFIG_DEFAULTS.items():
        if key in config and config[key] is not None:
            resolved[key] = config[key]
        else:
            resolved[key] = default_value

    # 规范化 auto_create_schema 为 bool（YAML 解析后通常已是 bool，
    # 但允许用户写 "true"/"false" 字符串时也能识别）
    raw_auto = resolved.get("auto_create_schema", True)
    if isinstance(raw_auto, str):
        resolved["auto_create_schema"] = raw_auto.strip().lower() in {"1", "true", "yes", "on"}
    else:
        resolved["auto_create_schema"] = bool(raw_auto)

    return resolved


def resolve_database_config(
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    maintenance_database: Optional[str] = None,
    schema_mode: Optional[str] = None,
    default_schema: Optional[str] = None,
    auto_create_schema: Optional[bool] = None,
) -> dict[str, Any]:
    """返回应用显式覆盖后的数据库配置。

    schema_mode/default_schema/auto_create_schema 为 P1 schema 隔离字段，
    调用方（如 ``DatabaseConnection.__init__``）可显式覆盖来源于 defaults.yaml 的值。
    """
    config = get_database_config().copy()

    overrides = {
        "host": host,
        "port": port,
        "database": database,
        "user": user,
        "password": password,
        "maintenance_database": maintenance_database,
        "schema_mode": schema_mode,
        "default_schema": default_schema,
        "auto_create_schema": auto_create_schema,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config["port"] = int(config["port"])
    config["auto_create_schema"] = bool(config.get("auto_create_schema", True))
    return config


__all__ = [
    "get_database_config",
    "get_database_config_path",
    "resolve_database_config",
]

"""配置表数据库建表元数据适配模块。

本模块从固定 Python schema 中提取 PostgreSQL 建表所需的表注释、
字段注释、字段类型、主键和索引配置。
"""
from __future__ import annotations

from typing import Any

from pgsql_db.config_table_schema import get_config_table_schemas

_PRIMARY_KEY_MARKERS = {"Y", "YES", "TRUE", "1", "是"}


def is_primary_key_marker(value: Any) -> bool:
    """判断字段级主键标记是否表示该字段应设为主键。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.upper() in _PRIMARY_KEY_MARKERS


def _append_unique(columns: list[str], column: Any) -> None:
    """向字段列表追加去重后的非空字段名。"""
    text = str(column).strip() if column is not None else ""
    if text and text not in columns:
        columns.append(text)


def load_config_table_comment_map(
    mapping_config_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """从固定 schema 读取 PostgreSQL 表级和字段级建表元数据。

    Args:
        mapping_config_path: 已废弃参数。传入非空值时抛出异常，避免继续
            依赖 YAML 字段映射。

    Returns:
        以物理表名为 key 的建表元数据映射，例如 ``cfg_global_seed``。

    Raises:
        ValueError: 当调用方继续传入已废弃的 mapping_config_path 时抛出。
    """
    if mapping_config_path is not None:
        raise ValueError(
            "mapping_config_path is no longer supported; config table database "
            "schemas are fixed in pgsql_db.config_table_schema"
        )

    tables = get_config_table_schemas()
    comment_map: dict[str, dict[str, Any]] = {}
    for key, table_cfg in tables.items():
        if not isinstance(table_cfg, dict):
            continue
        db_table = str(table_cfg.get("db_table") or key)
        local_sheet = str(table_cfg.get("local_sheet") or key)
        table_comment = (
            table_cfg.get("comment")
            or table_cfg.get("description")
            or f"配置表：{local_sheet}"
        )

        column_comments: dict[str, str] = {}
        column_types: dict[str, str] = {}
        primary_key_columns: list[str] = []
        index_columns: list[list[str]] = []
        table_primary_key = table_cfg.get("primary_key") or []
        if isinstance(table_primary_key, (list, tuple)):
            for column in table_primary_key:
                _append_unique(primary_key_columns, column)
        elif table_primary_key:
            _append_unique(primary_key_columns, table_primary_key)

        all_fields = list(table_cfg.get("fields") or []) + list(
            table_cfg.get("system_fields") or []
        )
        for field in all_fields:
            if not isinstance(field, dict):
                continue
            db_name = field.get("db_name") or field.get("local_name")
            comment = field.get("comment")
            if db_name and comment:
                column_comments[str(db_name)] = str(comment)
            db_type = field.get("db_type")
            if db_name and db_type:
                column_types[str(db_name)] = str(db_type)
            if db_name and is_primary_key_marker(field.get("primary_key")):
                _append_unique(primary_key_columns, db_name)

        has_config_name = any(
            field.get("db_name") == "config_name"
            for field in table_cfg.get("system_fields") or []
        )
        if has_config_name:
            index_columns.append(["config_name"])

        comment_map[db_table] = {
            "table_comment": str(table_comment) if table_comment else "",
            "column_comments": column_comments,
            "column_types": column_types,
            "primary_key_columns": primary_key_columns,
            "index_columns": index_columns,
        }

    return comment_map


__all__ = ["is_primary_key_marker", "load_config_table_comment_map"]

"""统一输出写入器 — DBWriter / ExcelWriter / MemoryWriter / CompositeWriter / NoopWriter。

所有输出写入器的统一接口 ``DataWriter(Protocol)``，模块不需要知道写到哪。
本期只把 PersistenceManager 的 DB 写入切到 DBWriter；模块入口签名不动。

DBWriter 封装现有 ``db.write_df`` / ``db.delete_where``（不引入 SA Session，
留待 Phase 2），保持运行时 psycopg COPY 批量写入路径不变。

用法：
    writer = DBWriter(orch.db)
    writer.write("module1_output_orderlog", df)
    writer.delete("cfg_global_seed", {"config_name": "test"})
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, TYPE_CHECKING, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from ..core.db.pgsql.db import DB

logger = logging.getLogger(__name__)


@runtime_checkable
class DataWriter(Protocol):
    """所有输出写入器的统一接口。"""

    def write(self, table_key: str, df: pd.DataFrame) -> None: ...
    def delete(self, table_name: str, conditions: dict) -> int: ...


class DBWriter:
    """写入 PostgreSQL（主路径）。

    封装现有 ``db.write_df`` / ``db.delete_where``，不引入 SA Session。
    运行时仍走 psycopg COPY 批量写入——完全保留现有行为。
    """

    def __init__(self, db: "DB"):
        self._db = db

    def write(self, table_key: str, df: pd.DataFrame) -> None:
        """将 DataFrame 写入指定表（COPY 协议批量写入）。

        Args:
            table_key: 数据库表名（如 ``"module1_output_orderlog"`` 或 ``"cfg_global_seed"``）。
            df: 已注入元数据列的 DataFrame。
        """
        if self._db is None:
            logger.debug("DBWriter: 无 DB 连接，跳过写入")
            return
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            logger.debug(f"DBWriter: DataFrame 为空，跳过写入 {table_key}")
            return
        self._db.write_df(table_key, df)

    def delete(self, table_name: str, conditions: dict) -> int:
        """从指定表删除满足条件的行。

        Args:
            table_name: 目标表名。
            conditions: ``{列名: 值}`` 字典，多列之间用 AND 连接。

        Returns:
            被删除的行数。
        """
        if self._db is None:
            return 0
        return self._db.delete_where(table_name, conditions)


class ExcelWriter:
    """写入本地 Excel（legacy / also-local 模式）。

    替代各模块自己的 output_writer.py，但本期暂不接入（模块入口签名不动）。
    """

    def __init__(self, output_dir: Path, date_str: str):
        self._output_dir = output_dir
        self._date_str = date_str

    def write(self, table_key: str, df: pd.DataFrame) -> None:
        """将 DataFrame 写入本地 Excel sheet。"""
        if df is None or df.empty:
            return
        file_path = self._output_dir / f"{self._date_str}.xlsx"
        with pd.ExcelWriter(str(file_path), engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=table_key, index=False)

    def delete(self, table_name: str, conditions: dict) -> int:
        """Excel 写入器不支持删除操作。"""
        logger.warning("ExcelWriter.delete: 不支持删除操作，跳过")
        return 0


class MemoryWriter:
    """写入内存 dict（测试用）。

    用法：在测试中注入 MemoryWriter 替代 DBWriter，模块可独立测试（不需要数据库）。
    """

    def __init__(self):
        self._store: dict[str, pd.DataFrame] = {}

    def write(self, table_key: str, df: pd.DataFrame) -> None:
        """将 DataFrame 存入内存 dict。"""
        if df is not None and not df.empty:
            self._store[table_key] = df.copy()

    def delete(self, table_name: str, conditions: dict) -> int:
        """从内存 dict 删除满足条件的行（简易实现）。"""
        if table_name not in self._store:
            return 0
        df = self._store[table_name]
        mask = pd.Series(True, index=df.index)
        for col, val in conditions.items():
            if col in df.columns:
                mask &= (df[col] == val)
        count = int(mask.sum())
        self._store[table_name] = df[~mask]
        return count

    def get(self, table_key: str) -> pd.DataFrame | None:
        """获取内存中的 DataFrame（测试断言用）。"""
        return self._store.get(table_key)

    @property
    def store(self) -> dict[str, pd.DataFrame]:
        """全部已写入的 DataFrame（测试断言用）。"""
        return dict(self._store)


class CompositeWriter:
    """组合多个 writer（同时写 DB + 本地 Excel）。"""

    def __init__(self, *writers: DataWriter):
        self._writers = list(writers)

    def write(self, table_key: str, df: pd.DataFrame) -> None:
        """依次写入所有 writer。"""
        for w in self._writers:
            w.write(table_key, df)

    def delete(self, table_name: str, conditions: dict) -> int:
        """依次删除（仅 DBWriter 实际删除，其它忽略）。"""
        total = 0
        for w in self._writers:
            total += w.delete(table_name, conditions)
        return total


class NoopWriter:
    """跳过写入（纯计算模式）。"""

    def write(self, table_key: str, df: pd.DataFrame) -> None:
        pass

    def delete(self, table_name: str, conditions: dict) -> int:
        return 0


__all__ = [
    "DataWriter",
    "DBWriter",
    "ExcelWriter",
    "MemoryWriter",
    "CompositeWriter",
    "NoopWriter",
]

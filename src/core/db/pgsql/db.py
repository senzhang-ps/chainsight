"""纯数据库读写层。

职责：连接管理 + SQL 执行 + DataFrame / Module 写入。
不负责：数据清洗、类型推断、元数据注入（由调用方在传入前完成）。
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import psycopg
from psycopg import sql

from .snapshot import Snapshot

logger = logging.getLogger(__name__)


class DB:
    """纯粹的数据库读写。"""

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._connection = None
        self.snapshot = Snapshot(self)

    # ── 连接管理 ──────────────────────────────

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def connect(self) -> psycopg.Connection:
        """建立连接（autocommit=True，确保事务语义正确）。"""
        if self._connection is not None and not self._connection.closed:
            return self._connection
        last_err = None
        for attempt in range(3):
            try:
                self._connection = psycopg.connect(
                    host=self.host,
                    port=self.port,
                    dbname=self.database,
                    user=self.user,
                    password=self.password,
                    client_encoding='UTF8',
                    autocommit=True,
                    connect_timeout=30,
                )
                return self._connection
            except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout) as e:
                last_err = e
                self._connection = None
                if attempt < 2:
                    time.sleep(5 * (2 ** attempt))
        raise last_err

    def close(self):
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None

    @contextmanager
    def get_cursor(self, commit: bool = True):
        """获取游标上下文管理器。

        Args:
            commit: True → 包裹在 BEGIN...COMMIT 中；False → autocommit 直接执行
        """
        conn = self.connect()
        if commit:
            with conn.transaction():
                cursor = conn.cursor()
                try:
                    yield cursor
                finally:
                    cursor.close()
        else:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    # ── 读取 ─────────────────────────────────

    def read(self, table: str, run_id: str = None,
             sim_date: str = None, config_name: str = None,
             filters: dict = None) -> pd.DataFrame:
        """按条件读取一张表，返回 DataFrame。"""
        conditions = []
        params_list = []

        if run_id is not None:
            conditions.append(sql.SQL("{} = %s").format(sql.Identifier('run_id')))
            params_list.append(run_id)
        if sim_date is not None:
            conditions.append(sql.SQL("{} = %s").format(sql.Identifier('sim_date')))
            params_list.append(sim_date)
        if config_name is not None:
            conditions.append(sql.SQL("{} = %s").format(sql.Identifier('config_name')))
            params_list.append(config_name)
        if filters:
            for col, val in filters.items():
                if val is None:
                    conditions.append(sql.SQL("{} IS NULL").format(sql.Identifier(col)))
                else:
                    conditions.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
                    params_list.append(val)

        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
        if conditions:
            query = sql.SQL("SELECT * FROM {} WHERE {}").format(
                sql.Identifier(table),
                sql.SQL(" AND ").join(conditions),
            )

        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, tuple(params_list) if params_list else None)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)

    # ── 写入 ─────────────────────────────────

    def write(self, source, run_id: str = None, sim_date: str = None,
              table_name: str = None):
        """写入数据到数据库。支持两种调用方式：

        1. Module 模式（向后兼容）：
             write(module, run_id, sim_date)
           调用 module.output() 获取 {table_name: DataFrame}。

        2. DataFrame 模式：
             write(df, table_name="my_table")
           将单个 DataFrame 写入指定表。

        Args:
            source: Module 实例或 pandas DataFrame。
            run_id: 运行标识（Module 模式必填）。
            sim_date: 仿真日期字符串（Module 模式必填）。
            table_name: 目标表名（DataFrame 模式必填）。
        """
        if isinstance(source, pd.DataFrame):
            # DataFrame 模式
            if table_name is None:
                raise ValueError("DataFrame 模式下 table_name 为必填参数")
            self.write_df(table_name, source)
        else:
            # Module 模式（原有逻辑不变）
            if run_id is None or sim_date is None:
                raise ValueError("Module 模式下 run_id 和 sim_date 为必填参数")
            results = source.output() if callable(source.output) else source.output
            if not results:
                return
            for tbl, df in results.items():
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    continue
                self.write_df(tbl, df)

    def write_df(self, table_name: str, df: pd.DataFrame):
        """将 DataFrame 写入指定表（COPY 协议批量写入）。公开方法。

        调用方应确保元数据列（run_id / sim_date / config_name / db_write_time）
        已在传入前注入完毕。

        列对齐策略（以库表为权威）：
        - 列名统一 strip + 小写，避免大小写/空格导致的 COPY 失败。
        - 表已存在：与库表真实列求交集，df 多余列丢弃（warning），交集为空跳过；
          并按库表列类型做类型转换（数值/日期/文本），让 psycopg 以原生类型写入，
          避免 ``bigint`` 列收到 ``"42.0"`` 之类的字符串导致类型错。
        - 表不存在（动态输出表）：按 df 当前列 CREATE TABLE IF NOT EXISTS 全 TEXT，
          值按文本写入。
        """
        # 统一小写 + 去空白
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        columns = list(df.columns)
        conn = self.connect()

        with conn.transaction():
            with conn.cursor() as cursor:
                rows = None
                if self.table_exists(table_name):
                    # 已存在表：按真实列求交集，丢弃多余列
                    col_types = self._get_table_columns(cursor, table_name)
                    keep = [c for c in columns if c in col_types]
                    dropped = [c for c in columns if c not in col_types]
                    if dropped:
                        logger.warning(
                            f"write_df {table_name}: 丢弃 df 中库表不存在的列 {dropped}"
                        )
                    if not keep:
                        logger.warning(
                            f"write_df {table_name}: 无可写列（df 与库表列无交集），跳过"
                        )
                        return
                    rows, columns = self._prepare_typed_rows(
                        df[keep], {c: col_types[c] for c in keep}
                    )
                    if not rows:
                        logger.info(f"写入 {table_name}: 0 行")
                        return
                else:
                    # 动态建表（输出表），用 df 当前列（已小写）
                    create_cols = [
                        sql.SQL('{} TEXT').format(sql.Identifier(c)) for c in columns
                    ]
                    cursor.execute(sql.SQL(
                        'CREATE TABLE IF NOT EXISTS {} ({})'
                    ).format(sql.Identifier(table_name), sql.SQL(', ').join(create_cols)))

                # COPY 批量写入
                copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
                    sql.Identifier(table_name),
                    sql.SQL(', ').join(sql.Identifier(c) for c in columns),
                )
                with cursor.copy(copy_sql) as copy:
                    if rows is not None:
                        # 已按库表类型转换：原生值，COPY 直接写
                        for row in rows:
                            copy.write_row(list(row))
                    else:
                        # 动态 TEXT 表：按文本写入
                        for row in df.itertuples(index=False, name=None):
                            copy.write_row(
                                [str(v) if v is not None else None for v in row]
                            )

        logger.info(f"写入 {table_name}: {len(df)} 行")

    def _get_table_columns(self, cursor, table_name: str) -> dict[str, str]:
        """查询库表真实列名 → data_type 映射（列名小写）。调用方需在事务游标内执行。"""
        cursor.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s",
            (table_name,),
        )
        return {row[0].lower(): (row[1] or "").lower() for row in cursor.fetchall()}

    # db data_type → 归类集合，供 _prepare_typed_rows 选择转换方式
    _INT_DB_TYPES = {"bigint", "integer", "smallint", "serial", "bigserial"}
    _FLOAT_DB_TYPES = {
        "double precision", "real", "numeric", "decimal", "money",
    }
    _DATETIME_DB_TYPES = {
        "timestamp without time zone", "timestamp with time zone",
        "timestamp", "date",
    }

    def _prepare_typed_rows(
        self, df: pd.DataFrame, col_types: dict[str, str]
    ) -> tuple[list[tuple], list[str]]:
        """按库表列 data_type 把 df 各列转为原生 Python 值，空值规整为 None。

        使 COPY 以原生类型（int/float/datetime/str）写入，规避 ``str()`` 把数值
        变成 ``"42.0"`` 写 ``bigint`` 失败的问题。

        Args:
            df: 仅含库表存在列的 DataFrame。
            col_types: ``{列名: data_type}``。

        Returns:
            ``(rows, columns)``：rows 为按列类型转换后的 Python 原生值元组列表
            （空值统一为 None）；columns 为列顺序。
        """
        columns = list(col_types.keys())
        col_values: list[list] = []
        for c in columns:
            dtype = col_types[c]
            series = df[c]
            if dtype in self._INT_DB_TYPES:
                s = pd.to_numeric(series, errors="coerce").astype("Int64")
                col_values.append(
                    [None if pd.isna(v) else int(v) for v in s.tolist()]
                )
            elif dtype in self._FLOAT_DB_TYPES:
                s = pd.to_numeric(series, errors="coerce")
                col_values.append(
                    [None if pd.isna(v) else float(v) for v in s.tolist()]
                )
            elif dtype in self._DATETIME_DB_TYPES:
                s = pd.to_datetime(series, errors="coerce")
                col_values.append(
                    [None if pd.isna(v) else v.to_pydatetime() for v in s.tolist()]
                )
            else:
                # 文本/布尔等：保持字符串（与 TEXT 列一致）
                col_values.append(
                    [None if pd.isna(v) else str(v) for v in series.tolist()]
                )
        # 列 → 行
        n = len(df)
        rows = [tuple(col_values[j][i] for j in range(len(columns)))
                for i in range(n)]
        return rows, columns

    # ── 通用 SQL ──────────────────────────────

    def delete_where(
        self, table_name: str, conditions: dict, *, safe: bool = True,
    ) -> int:
        """从指定表删除满足条件的行。

        Args:
            table_name: 目标表名。
            conditions: ``{列名: 值}`` 字典，多列之间用 AND 连接。
            safe: 为 True 时，表不存在则静默返回 0 而非报错。

        Returns:
            被删除的行数。
        """
        if safe and not self.table_exists(table_name):
            return 0
        clauses = []
        params = []
        for col, val in conditions.items():
            if val is None:
                clauses.append(sql.SQL("{} IS NULL").format(sql.Identifier(col)))
            else:
                clauses.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
                params.append(val)
        stmt = sql.SQL("DELETE FROM {} WHERE {}").format(
            sql.Identifier(table_name),
            sql.SQL(" AND ").join(clauses),
        )
        with self.get_cursor() as cursor:
            cursor.execute(stmt, tuple(params))
            return cursor.rowcount

    def execute(self, query: str, params: tuple = None):
        """执行任意 SQL。"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)

    def execute_query(self, query: str, params: tuple = None) -> list:
        """执行查询返回行列表。"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_query_df(self, query: str, params: tuple = None) -> pd.DataFrame:
        """执行查询返回 DataFrame。"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return pd.DataFrame(cursor.fetchall(), columns=columns)

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在。"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (table_name,)
            )
            return cursor.fetchone()[0]

    def get_all_tables(self) -> list[str]:
        """获取当前数据库中所有用户表名。"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            return [row[0] for row in cursor.fetchall()]

    # ── 向后兼容 ──────────────────────────────

    _write_df = write_df

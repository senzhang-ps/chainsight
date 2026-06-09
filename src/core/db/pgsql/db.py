"""纯数据库读写层。

职责：连接管理 + SQL 执行 + 按 Module 对象写入。
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

    def write(self, module, run_id: str, sim_date: str):
        """传入 Module 对象，按 module.output() 写入对应的表。

        module.output() 返回 {table_name: DataFrame}，
        DataFrame 应由调用方确保元数据列已注入且数据已清洗。
        """
        results = module.output() if callable(module.output) else module.output
        if not results:
            return
        for table_name, df in results.items():
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue
            self._write_df(table_name, df)

    def _write_df(self, table_name: str, df: pd.DataFrame):
        """将 DataFrame 写入指定表（COPY 协议批量写入）。

        调用方应确保元数据列（run_id / sim_date / config_name / db_write_time）
        已在传入前注入完毕。
        """
        columns = list(df.columns)
        conn = self.connect()

        with conn.transaction():
            with conn.cursor() as cursor:
                # 确保表存在
                col_list = sql.SQL(', ').join(sql.Identifier(c) for c in columns)
                placeholder = sql.SQL(', ').join(sql.SQL('%s') for _ in columns)
                create_cols = []
                for c in columns:
                    create_cols.append(sql.SQL('{} TEXT').format(sql.Identifier(c)))
                cursor.execute(sql.SQL(
                    'CREATE TABLE IF NOT EXISTS {} ({})'
                ).format(sql.Identifier(table_name), sql.SQL(', ').join(create_cols)))

                # COPY 批量写入
                copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
                    sql.Identifier(table_name),
                    sql.SQL(', ').join(sql.Identifier(c) for c in columns),
                )
                with cursor.copy(copy_sql) as copy:
                    for row in df.itertuples(index=False, name=None):
                        copy.write_row([str(v) if v is not None else None for v in row])

        logger.info(f"写入 {table_name}: {len(df)} 行")

    # ── 通用 SQL ──────────────────────────────

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

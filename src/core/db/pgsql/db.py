"""纯数据库读写层。

职责：连接管理 + SQL 执行 + DataFrame / Module 写入。
不负责：数据清洗、类型推断、元数据注入（由调用方在传入前完成）。
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

logger = logging.getLogger(__name__)


class DB:
    """纯粹的数据库读写。"""

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str, *,
                 schema: str = "public",
                 auto_create_schema: bool = True):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        # P1 schema 隔离：所有表落到 self.schema（默认 public，向后兼容旧调用方）。
        self.schema = schema
        self.auto_create_schema = auto_create_schema
        self._connection = None
        self._schema_ensured = False
        self._col_type_cache: dict[str, dict[str, str]] = {}

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
                # 新连接需重新 ensure schema（CREATE SCHEMA IF NOT EXISTS）。
                self._schema_ensured = False
                self._ensure_schema_ready()
                return self._connection
            except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout) as e:
                last_err = e
                self._connection = None
                error_text = str(e).lower()
                if 'does not exist' in error_text and attempt == 0:
                    try:
                        self.create_database_if_not_exists()
                        continue
                    except Exception:
                        pass
                if attempt < 2:
                    time.sleep(5 * (2 ** attempt))
        raise last_err

    def close(self):
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None

    # ── schema 隔离 ──────────────────────────

    def _ensure_schema_ready(self) -> None:
        """在当前连接上 ``CREATE SCHEMA IF NOT EXISTS``，仅执行一次（每条新连接重置）。

        - 仅当 ``auto_create_schema=True`` 时创建。
        - 用 ``sql.Identifier`` 引用 schema 名，兼容 ``bc-dev`` 这类含 ``-`` 的名字。
        - 不依赖 ``search_path``：表 SQL 一律使用 schema-qualified identifier。
        """
        if self._schema_ensured or not self.auto_create_schema:
            return
        try:
            with self._connection.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema)
                    )
                )
            self._schema_ensured = True
        except Exception as e:
            logger.warning(
                f"[schema] CREATE SCHEMA IF NOT EXISTS {self.schema!r} 失败：{e}"
            )
            # 不抛出：后续 SQL 若 schema 真缺失，会以更明确的错误暴露。

    def _qualified(self, table_name: str) -> sql.Composed:
        """返回 schema-qualified 的 ``sql.Identifier``，供 ``sql.SQL().format()`` 拼接。

        当 schema 含 ``-`` 等需 quoting 的字符时，``sql.Identifier`` 会自动加引号。
        """
        return sql.Identifier(self.schema, table_name)

    def qualified_name(self, table_name: str) -> str:
        """返回 ``"schema"."table"`` 形式字符串，供 raw-SQL 调用方拼接。

        要求 ``table_name`` 为代码可控的可信标识符（同旧层 ``db_connection`` 契约）；
        schema 已由 ``resolve_project_schema`` 校验为 ``[a-z0-9_-]+``。
        """
        return f'"{self.schema}"."{table_name}"'

    def database_exists(self) -> bool:
        """检查目标数据库是否存在。"""
        temp_conn = None
        try:
            temp_conn = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname='postgres',
                user=self.user,
                password=self.password,
                client_encoding='UTF8',
                connect_timeout=30,
            )
            with temp_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (self.database,),
                )
                return cursor.fetchone() is not None
        finally:
            if temp_conn and not temp_conn.closed:
                temp_conn.close()

    def create_database_if_not_exists(self) -> bool:
        """如果目标数据库不存在，则创建它。"""
        if self.database_exists():
            return True
        temp_conn = None
        try:
            temp_conn = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname='postgres',
                user=self.user,
                password=self.password,
                client_encoding='UTF8',
                autocommit=True,
                connect_timeout=30,
            )
            with temp_conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {} ").format(
                        sql.Identifier(self.database)
                    )
                )
            return True
        finally:
            if temp_conn and not temp_conn.closed:
                temp_conn.close()

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

        query = sql.SQL("SELECT * FROM {}").format(self._qualified(table))
        if conditions:
            query = sql.SQL("SELECT * FROM {} WHERE {}").format(
                self._qualified(table),
                sql.SQL(" AND ").join(conditions),
            )

        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, tuple(params_list) if params_list else None)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)

    def read_polars(self, table: str, run_id: str = None,
                    sim_date: str = None, config_name: str = None,
                    filters: dict = None):
        """按条件直接读取为 Polars DataFrame。

        主要供 resume 的百万行 M1 快照使用，避免 ``read()`` 先构造 pandas
        DataFrame、再转换为 Polars 所造成的双份内存与逐列转换开销。
        """
        import polars as pl

        conditions = []
        params_list = []
        for column, value in (
            ('run_id', run_id), ('sim_date', sim_date),
            ('config_name', config_name),
        ):
            if value is not None:
                conditions.append(sql.SQL("{} = %s").format(sql.Identifier(column)))
                params_list.append(value)
        if filters:
            for col, val in filters.items():
                if val is None:
                    conditions.append(sql.SQL("{} IS NULL").format(sql.Identifier(col)))
                else:
                    conditions.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
                    params_list.append(val)

        query = sql.SQL("SELECT * FROM {} ").format(self._qualified(table))
        if conditions:
            query += sql.SQL("WHERE {} ").format(sql.SQL(" AND ").join(conditions))

        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, tuple(params_list) if params_list else None)
            columns = [desc[0] for desc in cursor.description]
            return pl.DataFrame(cursor.fetchall(), schema=columns, orient='row')

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

    def write_df(self, table_name: str, df: pd.DataFrame, *, cursor=None):
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

        Args:
            table_name: 目标表名。
            df: 待写入的 DataFrame。
            cursor: 外部事务游标。传入时跳过内部 conn.transaction()，
                   由调用方管理事务边界（用于批量事务优化）。
        """
        # 统一小写 + 去空白
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        columns = list(df.columns)

        if cursor is not None:
            # 外部事务模式：复用传入游标，不自行管理事务
            self._do_write_df(cursor, table_name, df, columns)
        else:
            conn = self.connect()
            with conn.transaction():
                with conn.cursor() as cur:
                    self._do_write_df(cur, table_name, df, columns)

        logger.info(f"写入 {table_name}: {len(df)} 行")

    def _do_write_df(self, cursor, table_name: str, df: pd.DataFrame,
                     columns: list[str]):
        """write_df 核心逻辑（在给定游标内执行 COPY）。"""
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
            ).format(self._qualified(table_name), sql.SQL(', ').join(create_cols)))
            # 清除缓存（新表列类型由 CREATE TABLE 决定）
            self._col_type_cache.pop(table_name, None)

        # COPY 批量写入
        copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
            self._qualified(table_name),
            sql.SQL(', ').join(sql.Identifier(c) for c in columns),
        )
        with cursor.copy(copy_sql) as copy:
            if rows is not None:
                # 已按库表类型转换：原生值，COPY 逐行写入
                for row in rows:
                    copy.write_row(list(row))
            else:
                # 动态 TEXT 表：按文本写入
                for row in df.itertuples(index=False, name=None):
                    copy.write_row(
                        [str(v) if v is not None else None for v in row]
                    )

    def _get_table_columns(self, cursor, table_name: str) -> dict[str, str]:
        """查询库表真实列名 → data_type 映射（列名小写）。调用方需在事务游标内执行。

        缓存结果，避免每次写入都查询 information_schema。
        """
        if table_name in self._col_type_cache:
            return self._col_type_cache[table_name]
        cursor.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (self.schema, table_name),
        )
        result = {row[0].lower(): (row[1] or "").lower() for row in cursor.fetchall()}
        self._col_type_cache[table_name] = result
        return result

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

        向量化版本：用 pandas/numpy 批量操作替代逐 cell 循环，
        对 50K+ 行 DataFrame 提速 5-10x。

        Args:
            df: 仅含库表存在列的 DataFrame。
            col_types: ``{列名: data_type}``。

        Returns:
            ``(rows, columns)``：rows 为按列类型转换后的 Python 原生值元组列表
            （空值统一为 None）；columns 为列顺序。
        """
        columns = list(col_types.keys())
        col_arrays: list[np.ndarray] = []
        for c in columns:
            dtype = col_types[c]
            s = df[c]
            if dtype in self._INT_DB_TYPES:
                s = pd.to_numeric(s, errors="coerce").astype("Int64")
                arr = s.to_numpy(dtype=object).copy()
                arr[pd.isna(s)] = None
                col_arrays.append(arr)
            elif dtype in self._FLOAT_DB_TYPES:
                s = pd.to_numeric(s, errors="coerce")
                arr = s.to_numpy(dtype=object).copy()
                arr[pd.isna(s)] = None
                col_arrays.append(arr)
            elif dtype in self._DATETIME_DB_TYPES:
                s = pd.to_datetime(s, errors="coerce")
                arr = np.empty(len(s), dtype=object)
                mask = pd.isna(s)
                arr[mask] = None
                non_na = s[~mask]
                arr[~mask] = [v.to_pydatetime() for v in non_na]
                col_arrays.append(arr)
            else:
                # 文本/布尔等：保持字符串
                # 某些 pandas ExtensionArray 返回只读 object 数组；后续需就地
                # 标准化空值和文本，因此显式复制为可写数组。
                arr = s.to_numpy(dtype=object).copy()
                mask = pd.isna(s)
                arr[mask] = None
                arr[~mask] = [str(v) for v in arr[~mask]]
                col_arrays.append(arr)

        # 列 → 行：zip 比逐索引构造快
        rows = list(zip(*col_arrays))
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
            self._qualified(table_name),
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
        """检查表是否存在（限定到 ``self.schema``）。"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s)",
                (self.schema, table_name)
            )
            return cursor.fetchone()[0]

    def existing_tables(self, table_names: list[str] | None = None) -> set[str]:
        """一次查询当前 schema 中存在的表名。

        Args:
            table_names: 可选的候选表名。传入时仅返回这些表中实际存在的项，
                用于避免续跑恢复时为每张状态表单独访问 ``information_schema``。

        Returns:
            当前 schema 中存在的表名集合。
        """
        query = (
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s"
        )
        params: tuple = (self.schema,)
        if table_names is not None:
            candidates = list(dict.fromkeys(table_names))
            if not candidates:
                return set()
            query += " AND table_name = ANY(%s)"
            params = (self.schema, candidates)

        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            return {row[0] for row in cursor.fetchall()}

    def get_all_tables(self) -> list[str]:
        """获取当前 schema 中所有用户表名。"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s ORDER BY table_name",
                (self.schema,),
            )
            return [row[0] for row in cursor.fetchall()]

    # ── 向后兼容 ──────────────────────────────

    _write_df = write_df

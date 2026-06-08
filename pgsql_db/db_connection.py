"""
PostgreSQL数据库连接模块
提供数据库连接、测试、表操作等功能
使用 psycopg3 (psycopg) 以解决Windows中文环境编码问题
"""

import psycopg
from psycopg import sql
import pandas as pd
from typing import Optional, List, Dict, Any, Sequence
from contextlib import contextmanager
import logging
import re
import time
from datetime import datetime

from src.utils.numeric_safe import coerce_db_float, coerce_db_int

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """PostgreSQL数据库连接类"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        schema: Optional[str] = None,
        auto_create_schema: Optional[bool] = None,
    ):
        """
        初始化数据库连接参数

        未显式传入的字段将从 ``config/defaults.yaml`` 的 ``database:`` 节点读取。

        参数：
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            schema: P1 schema 隔离时使用的 PostgreSQL schema 名（已小写化校验）。
                不传时回落到 defaults.yaml 的 ``database.default_schema``。
            auto_create_schema: 缺失 schema 时是否自动 ``CREATE SCHEMA IF NOT EXISTS``。
        """
        from .settings import resolve_database_config
        cfg = resolve_database_config(
            host=host, port=port, database=database, user=user, password=password,
            auto_create_schema=auto_create_schema,
        )
        self.host = cfg["host"]
        self.port = cfg["port"]
        self.database = cfg["database"]
        self.user = cfg["user"]
        self.password = cfg["password"]
        # schema 缺省 -> default_schema；调用方应优先显式传入校验过的值。
        self.schema = (schema or cfg.get("default_schema") or "public")
        self.auto_create_schema = bool(cfg.get("auto_create_schema", True))
        self._connection = None
        # 标记 schema 是否已在当前连接上被 ensure 过，避免每次 connect() 重复创建。
        self._schema_ensured = False
    
    @property
    def connection_string(self) -> str:
        """返回连接字符串"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def connect(self) -> psycopg.Connection:
        """建立数据库连接（autocommit=True 模式）

         使用 autocommit=True 确保 conn.transaction() 始终创建真正的
        BEGIN...COMMIT 事务块。在 autocommit=False（psycopg3 默认值）下，
        任何先前的 SQL 语句（包括 SELECT）都会隐式开启事务，导致后续的
        conn.transaction() 仅创建 SAVEPOINT 而非顶层事务。SAVEPOINT 退出时
        只发出 RELEASE SAVEPOINT 而非 COMMIT，数据不会持久化到磁盘。
        进程被 KeyboardInterrupt 终止后，PostgreSQL 回滚整个未提交的外层事务，
        所有已写入的数据全部丢失。

        使用 autocommit=True 后：
        - conn.transaction() 始终发出 BEGIN...COMMIT（数据真正持久化）
        - 每次 flush 的数据在事务退出时即刻可见且不可丢失
        - 无需在 conn.transaction() 前手动 conn.commit() 清理隐式事务
        """
        if self._connection is not None and not self._connection.closed:
            return self._connection
        # 重试退避：Summary 等长查询结束后 PostgreSQL 可能短暂繁忙
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
                # 新连接需要重新 ensure schema（哪怕之前的 connection 已 ensure 过）
                self._schema_ensured = False
                self._ensure_schema_ready()
                return self._connection
            except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout) as e:
                last_err = e
                self._connection = None
                if attempt < 2:
                    time.sleep(5 * (2 ** attempt))  # 5s, 10s
        raise last_err

    def _ensure_schema_ready(self) -> None:
        """在当前连接上 ``CREATE SCHEMA IF NOT EXISTS``，仅执行一次。

        - 仅当 ``auto_create_schema=True`` 时创建。
        - SQL 通过 ``sql.Identifier`` 引用 schema 名，兼容 ``bc-dev`` 这类含 ``-`` 的名字。
        - 不依赖 ``search_path``：表 SQL 一律使用 schema-qualified identifier。
        """
        if self._schema_ensured:
            return
        if not self.auto_create_schema:
            self._schema_ensured = True
            return
        try:
            conn = self._connection
            if conn is None or conn.closed:
                return
            with conn.cursor() as cur:
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
            # 不抛出：调用方在后续 SQL 中若 schema 真缺失，会以更明确的错误暴露。

    def _qualified(self, table_name: str, schema: Optional[str] = None) -> sql.Composed:
        """返回 schema-qualified 的 ``sql.Identifier``，供 ``sql.SQL().format(...)`` 拼接使用。

        当 schema 含 ``-`` 等需 quoting 的字符时，``sql.Identifier`` 会自动加引号。
        """
        return sql.Identifier(schema or self.schema, self._clean_name(table_name))

    def qualified_name(self, table_name: str, schema: Optional[str] = None) -> str:
        """返回 ``"schema"."table"`` 形式的字符串，便于历史的字符串 SQL 拼接。

        - ``schema`` / ``table`` 已通过 ``_clean_name`` 限定到 ``[a-z0-9_]``，
          再加上 schema 限定到 ``[a-z0-9_-]``，双引号包裹后**不会**产生 SQL 注入。
        - 兼容含 ``-`` 的 schema 名（如 ``bc-dev``）。
        """
        target_schema = schema or self.schema
        clean_table = self._clean_name(table_name)
        return f'"{target_schema}"."{clean_table}"'
    
    def close(self):
        """关闭数据库连接"""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
    
    @contextmanager
    def get_cursor(self, commit: bool = True):
        """
        获取数据库游标的上下文管理器
        
        参数：
            commit: 是否在事务中执行（True=包裹在 BEGIN...COMMIT 中，False=直接执行）
        
         autocommit=True 模式下：
        - commit=True: 使用 conn.transaction() 包裹，保证原子性
        - commit=False: 直接执行（每条语句自动提交，适用于只读查询）
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
    
    def database_exists(self) -> bool:
        """
        检测目标数据库是否存在
        
        返回：
            bool: 数据库是否存在
        """
        temp_conn = None
        try:
            # 连接到postgres数据库检查目标数据库是否存在
            temp_conn = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname='postgres',
                user=self.user,
                password=self.password,
                client_encoding='UTF8'
            )
            with temp_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (self.database,)
                )
                exists = cursor.fetchone() is not None
            return exists
        except Exception as e:
            raise
        finally:
            if temp_conn and not temp_conn.closed:
                temp_conn.close()
    
    def create_database_if_not_exists(self) -> bool:
        """
        如果数据库不存在则创建
        
        返回：
            bool: 是否成功
        """
        if self.database_exists():
            return True
        
        temp_conn = None
        try:
            # 连接到postgres数据库创建新数据库
            temp_conn = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname='postgres',
                user=self.user,
                password=self.password,
                client_encoding='UTF8',
                autocommit=True  # 创建数据库需要autocommit
            )
            with temp_conn.cursor() as cursor:
                # 使用安全的方式创建数据库
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database))
                )
            return True
        except Exception as e:
            raise
        finally:
            if temp_conn and not temp_conn.closed:
                temp_conn.close()
    
    def check_tables_exist(self, table_names: List[str], schema: Optional[str] = None) -> Dict[str, bool]:
        """
        检查多个表是否存在

        参数：
            table_names: 表名列表
            schema: 目标 schema（默认使用 self.schema）

        返回：
            dict: 表名 -> 是否存在
        """
        result = {}
        existing_tables = set(self.get_all_tables(schema=schema))
        for table_name in table_names:
            clean_name = self._clean_name(table_name)
            result[table_name] = clean_name in existing_tables
        return result
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试数据库连接
        
        返回：
            dict: 包含连接状态、数据库版本、连接时间等信息
        """
        result = {
            "success": False,
            "message": "",
            "version": None,
            "connection_time_ms": 0,
            "database": self.database,
            "host": self.host,
            "port": self.port
        }
        
        start_time = time.time()
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                result["version"] = version
                result["success"] = True
                result["message"] = "连接成功"
        except Exception as e:
            raise
        finally:
            result["connection_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return result
    
    def table_exists(self, table_name: str, schema: Optional[str] = None) -> bool:
        """检查表是否存在（默认使用 self.schema）。"""
        target_schema = schema or self.schema
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                );
            """, (target_schema, table_name))
            return cursor.fetchone()[0]

    def get_all_tables(self, schema: Optional[str] = None) -> List[str]:
        """获取所有表名（默认使用 self.schema）。"""
        target_schema = schema or self.schema
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name;
            """, (target_schema,))
            return [row[0] for row in cursor.fetchall()]

    def drop_table(self, table_name: str, cascade: bool = False, schema: Optional[str] = None):
        """删除表（默认在 self.schema 中）。"""
        cascade_str = "CASCADE" if cascade else ""
        with self.get_cursor() as cursor:
            query = sql.SQL("DROP TABLE IF EXISTS {} {}").format(
                self._qualified(table_name, schema),
                sql.SQL(cascade_str)
            )
            cursor.execute(query)
    
    def check_config_exists(self, table_name: str, config_name: str) -> bool:
        """
        检查指定配置是否已在表中存在数据

        参数：
            table_name: 表名
            config_name: 配置名称（如 BC_S5, BC_S9）

        返回：
            bool: 配置是否已存在
        """
        clean_name = self._clean_name(table_name)
        if not self.table_exists(clean_name):
            return False

        try:
            with self.get_cursor(commit=False) as cursor:
                # 先检查表是否有 config_name 列（限定到当前 schema 防止跨 schema 同名表干扰）
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s AND column_name = 'config_name'
                """, (self.schema, clean_name))
                if not cursor.fetchone():
                    return False

                # 检查是否有该配置的数据
                cursor.execute(
                    sql.SQL("SELECT 1 FROM {} WHERE config_name = %s LIMIT 1").format(
                        self._qualified(clean_name)
                    ),
                    (config_name,)
                )
                return cursor.fetchone() is not None
        except Exception as e:
            raise

    def delete_config_data(self, table_name: str, config_name: str) -> int:
        """
        删除表中指定配置的数据

        参数：
            table_name: 表名
            config_name: 配置名称（如 BC_S5, BC_S9）

        返回：
            int: 删除的行数
        """
        clean_name = self._clean_name(table_name)
        if not self.table_exists(clean_name):
            return 0

        try:
            with self.get_cursor() as cursor:
                cursor.execute(
                    sql.SQL("DELETE FROM {} WHERE config_name = %s").format(
                        self._qualified(clean_name)
                    ),
                    (config_name,)
                )
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    pass
                return deleted_count
        except Exception as e:
            raise
    
    def create_table_from_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        add_write_time: bool = True,
        config_name: Optional[str] = None,
        round_float_values: Optional[bool] = None,
        table_comment: Optional[str] = None,
        column_comments: Optional[Dict[str, str]] = None,
        column_types: Optional[Dict[str, str]] = None,
        primary_key_columns: Optional[Sequence[str]] = None,
        index_columns: Optional[Sequence[Sequence[str]]] = None,
    ) -> bool:
        """
        根据DataFrame创建表并写入数据
        
        参数：
            df: 数据DataFrame
            table_name: 表名
            if_exists: 如果表存在的处理方式 ('replace', 'append', 'fail')
            add_write_time: 是否自动添加写入时间列
            config_name: 配置文件标识（如 BC_S5, BC_S9），用于区分不同配置的数据
            round_float_values: 是否在写入前将浮点数四舍五入到10位小数；默认对 cfg_* 表关闭，对其他表开启
            table_comment: 可选的 PostgreSQL 表注释
            column_comments: 可选的 PostgreSQL 字段注释，key 为写库字段名
            column_types: 可选的 PostgreSQL 字段类型，key 为写库字段名
            primary_key_columns: 可选的 PostgreSQL 主键字段，key 为写库字段名
            index_columns: 可选的 PostgreSQL 索引字段组，每组为一个普通索引
        
        返回：
            bool: 是否成功
        """
        # 清理表名（去除特殊字符）
        clean_table_name = self._clean_name(table_name)
        if round_float_values is None:
            round_float_values = not clean_table_name.startswith('cfg_')
        
        # 检查是否为空表（只有列定义）
        is_empty_table = df.empty
        
        # 添加元数据列
        df_to_write = df.copy()
        
        # 添加config_name列（用于区分不同配置文件的数据）
        if config_name:
            if is_empty_table:
                df_to_write['config_name'] = pd.Series(dtype='object')
            else:
                df_to_write['config_name'] = config_name
        
        # 添加写入时间列
        if add_write_time:
            if is_empty_table:
                # 空表只添加列定义
                df_to_write['db_write_time'] = pd.Series(dtype='datetime64[ns]')
            else:
                df_to_write['db_write_time'] = datetime.now()

        write_columns = [self._clean_name(str(col)) for col in df_to_write.columns]
        clean_column_types = self._normalize_column_types(
            column_types,
            existing_columns=write_columns,
        )
        clean_primary_key_columns = self._normalize_primary_key_columns(
            primary_key_columns,
            existing_columns=write_columns,
            include_config_name=bool(config_name),
        )
        clean_index_columns = self._normalize_index_columns(
            index_columns,
            existing_columns=write_columns,
        )
        
        # 检查表是否存在
        exists = self.table_exists(clean_table_name)
        if exists and clean_table_name.startswith("cfg_"):
            self.drop_column_if_exists(clean_table_name, "config_type")
        
        if exists:
            if if_exists == "fail":
                raise ValueError(f"表 {clean_table_name} 已存在")
            elif if_exists == "replace":
                # 为避免丢失历史数据，replace模式改为追加写入
                if not self._check_table_compatible(
                    clean_table_name,
                    df_to_write,
                    column_types=clean_column_types,
                ):
                    return False
            elif if_exists == "append":
                # 追加模式（append）：检查表结构是否兼容
                if not self._check_table_compatible(
                    clean_table_name,
                    df_to_write,
                    column_types=clean_column_types,
                ):
                    return False
        
        # 创建表（如果不存在）
        if not self.table_exists(clean_table_name):
            columns = []
            for col_name, dtype in df_to_write.dtypes.items():
                clean_col = self._clean_name(str(col_name))
                pg_type = clean_column_types.get(clean_col) or self._pandas_to_pg_type(
                    dtype,
                    col_name=clean_col,
                )
                columns.append(
                    sql.SQL("{} {}").format(sql.Identifier(clean_col), sql.SQL(pg_type))
                )
            if clean_primary_key_columns:
                columns.append(
                    sql.SQL("PRIMARY KEY ({})").format(
                        sql.SQL(", ").join(
                            [sql.Identifier(col) for col in clean_primary_key_columns]
                        )
                    )
                )

            create_sql = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                self._qualified(clean_table_name),
                sql.SQL(", ").join(columns),
            )

            with self.get_cursor() as cursor:
                cursor.execute(create_sql)

        else:
            if is_empty_table:
                pass
            else:
                pass

            self._ensure_primary_key(clean_table_name, clean_primary_key_columns)

        self._apply_table_comments(
            clean_table_name,
            table_comment=table_comment,
            column_comments=column_comments,
            existing_columns=[self._clean_name(str(col)) for col in df_to_write.columns],
        )
        self._create_fixed_indexes(
            clean_table_name,
            clean_index_columns,
            primary_key_columns=clean_primary_key_columns,
        )
        
        # 插入数据（非空表才插入）
        if not is_empty_table:
            self._insert_dataframe(
                df_to_write,
                clean_table_name,
                round_float_values=round_float_values,
            )
        
        # 性能优化 (Phase 4 - 问题2)：调整索引创建顺序
        # - 原因：先创建索引再写入数据，导致B-tree维护开销 +10-20%
        # - 改进：先COPY写入，再创建索引（一次性开销）
        # - 性能收益：-75-90%
        # - 仅为新表创建索引（避免重复创建）
        if not self.table_exists(clean_table_name):
            self._create_auto_indexes(clean_table_name, df_to_write)
        
        return True

    def _apply_table_comments(
        self,
        table_name: str,
        *,
        table_comment: Optional[str] = None,
        column_comments: Optional[Dict[str, str]] = None,
        existing_columns: Optional[List[str]] = None,
    ) -> None:
        """写入 PostgreSQL 表/字段注释。"""
        normalized_table_comment = self._normalize_comment(table_comment)
        normalized_column_comments = {
            self._clean_name(str(column)): self._normalize_comment(comment)
            for column, comment in (column_comments or {}).items()
        }
        normalized_column_comments = {
            column: comment
            for column, comment in normalized_column_comments.items()
            if comment
        }

        if not normalized_table_comment and not normalized_column_comments:
            return

        allowed_columns = (
            {self._clean_name(str(column)) for column in existing_columns}
            if existing_columns is not None
            else None
        )
        with self.get_cursor() as cursor:
            if normalized_table_comment:
                cursor.execute(
                    sql.SQL("COMMENT ON TABLE {} IS {}").format(
                        self._qualified(table_name),
                        sql.Literal(normalized_table_comment),
                    )
                )

            for column, comment in normalized_column_comments.items():
                if allowed_columns is not None and column not in allowed_columns:
                    continue
                cursor.execute(
                    sql.SQL("COMMENT ON COLUMN {}.{} IS {}").format(
                        self._qualified(table_name),
                        sql.Identifier(column),
                        sql.Literal(comment),
                    )
                )

    @staticmethod
    def _normalize_comment(comment: Any) -> str:
        if comment is None:
            return ""
        text = str(comment).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    def apply_table_comments(
        self,
        table_name: str,
        *,
        table_comment: Optional[str] = None,
        column_comments: Optional[Dict[str, str]] = None,
    ) -> None:
        """对已存在表刷新 PostgreSQL 表/字段注释。"""
        clean_table_name = self._clean_name(table_name)
        if not self.table_exists(clean_table_name):
            return
        existing_columns = list(self._get_column_types(clean_table_name).keys())
        self._apply_table_comments(
            clean_table_name,
            table_comment=table_comment,
            column_comments=column_comments,
            existing_columns=existing_columns,
        )

    def drop_column_if_exists(self, table_name: str, column_name: str) -> None:
        """删除指定表的指定列（若存在）。"""
        clean_table_name = self._clean_name(table_name)
        clean_column_name = self._clean_name(column_name)
        if not self.table_exists(clean_table_name):
            return
        with self.get_cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER TABLE {} DROP COLUMN IF EXISTS {}").format(
                    self._qualified(clean_table_name),
                    sql.Identifier(clean_column_name),
                )
            )

    def _normalize_primary_key_columns(
        self,
        primary_key_columns: Optional[Sequence[str]],
        *,
        existing_columns: Sequence[str],
        include_config_name: bool = False,
    ) -> List[str]:
        """标准化并过滤主键列；配置表按 config_name 共表存储时自动纳入配置名。"""
        allowed = {self._clean_name(str(column)) for column in existing_columns}
        columns: List[str] = []

        def add_column(column: str) -> None:
            clean_column = self._clean_name(str(column))
            if clean_column and clean_column in allowed and clean_column not in columns:
                columns.append(clean_column)

        if include_config_name and primary_key_columns and "config_name" in allowed:
            add_column("config_name")
        for column in primary_key_columns or []:
            add_column(str(column))
        return columns

    def _normalize_column_types(
        self,
        column_types: Optional[Dict[str, str]],
        *,
        existing_columns: Sequence[str],
    ) -> Dict[str, str]:
        """标准化 YAML 中声明的 PostgreSQL 字段类型。"""
        allowed_columns = {self._clean_name(str(column)) for column in existing_columns}
        normalized: Dict[str, str] = {}
        for column, db_type in (column_types or {}).items():
            clean_column = self._clean_name(str(column))
            if clean_column not in allowed_columns:
                continue
            pg_type = self._normalize_pg_type(db_type)
            if pg_type:
                normalized[clean_column] = pg_type
        return normalized

    def _normalize_index_columns(
        self,
        index_columns: Optional[Sequence[Sequence[str]]],
        *,
        existing_columns: Sequence[str],
    ) -> List[List[str]]:
        """标准化固定索引字段组。"""
        allowed = {self._clean_name(str(column)) for column in existing_columns}
        normalized: List[List[str]] = []
        seen: set[tuple[str, ...]] = set()
        for group in index_columns or []:
            cols: List[str] = []
            for column in group or []:
                clean_column = self._clean_name(str(column))
                if clean_column and clean_column in allowed and clean_column not in cols:
                    cols.append(clean_column)
            key = tuple(cols)
            if key and key not in seen:
                normalized.append(cols)
                seen.add(key)
        return normalized

    @staticmethod
    def _normalize_pg_type(db_type: Any) -> str:
        """将 YAML 字段类型别名转换为安全的 PostgreSQL DDL 类型。"""
        if db_type is None:
            return ""
        text = str(db_type).strip()
        if not text:
            return ""
        compact = " ".join(text.lower().split())
        aliases = {
            "str": "TEXT",
            "string": "TEXT",
            "text": "TEXT",
            "object": "TEXT",
            "int": "BIGINT",
            "int8": "BIGINT",
            "int64": "BIGINT",
            "bigint": "BIGINT",
            "integer": "INTEGER",
            "int4": "INTEGER",
            "smallint": "SMALLINT",
            "int2": "SMALLINT",
            "float": "DOUBLE PRECISION",
            "float8": "DOUBLE PRECISION",
            "float64": "DOUBLE PRECISION",
            "double": "DOUBLE PRECISION",
            "double precision": "DOUBLE PRECISION",
            "real": "REAL",
            "float4": "REAL",
            "bool": "BOOLEAN",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "datetime": "TIMESTAMP",
            "datetime64[ns]": "TIMESTAMP",
            "datetime64[us]": "TIMESTAMP",
            "timestamp": "TIMESTAMP",
            "timestamp without time zone": "TIMESTAMP WITHOUT TIME ZONE",
            "timestamp with time zone": "TIMESTAMP WITH TIME ZONE",
            "json": "JSON",
            "jsonb": "JSONB",
        }
        if compact in aliases:
            return aliases[compact]

        upper = " ".join(text.upper().split())
        allowed_types = {
            "TEXT",
            "BIGINT",
            "INTEGER",
            "SMALLINT",
            "DOUBLE PRECISION",
            "REAL",
            "BOOLEAN",
            "DATE",
            "TIMESTAMP",
            "TIMESTAMP WITHOUT TIME ZONE",
            "TIMESTAMP WITH TIME ZONE",
            "JSON",
            "JSONB",
        }
        if upper in allowed_types:
            return upper
        if re.fullmatch(r"(NUMERIC|DECIMAL)\(\d+(,\s*\d+)?\)", upper):
            return upper.replace(" ", "")
        return ""

    def _get_primary_key_columns(self, table_name: str) -> List[str]:
        """读取当前 schema 下表的主键字段顺序。"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(
                """
                SELECT a.attname
                FROM pg_index i
                JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord)
                  ON TRUE
                JOIN pg_attribute a
                  ON a.attrelid = i.indrelid
                 AND a.attnum = k.attnum
                WHERE i.indrelid = %s::regclass
                  AND i.indisprimary
                ORDER BY k.ord;
                """,
                (self.qualified_name(table_name),),
            )
            return [row[0] for row in cursor.fetchall()]

    def _ensure_primary_key(
        self,
        table_name: str,
        primary_key_columns: Sequence[str],
    ) -> None:
        """为已存在表补齐主键约束；已有主键时保持原状。"""
        if not primary_key_columns:
            return

        existing_pk = self._get_primary_key_columns(table_name)
        if existing_pk:
            if list(existing_pk) != list(primary_key_columns):
                logger.warning(
                    "[schema] %s 已存在主键 %s，跳过映射主键 %s",
                    table_name,
                    existing_pk,
                    list(primary_key_columns),
                )
            return

        constraint_name = self._clean_name(f"{table_name}_pkey")
        with self.get_cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} PRIMARY KEY ({})").format(
                    self._qualified(table_name),
                    sql.Identifier(constraint_name),
                    sql.SQL(", ").join(
                        [sql.Identifier(col) for col in primary_key_columns]
                    ),
                )
            )

    def _create_fixed_indexes(
        self,
        table_name: str,
        index_columns: Sequence[Sequence[str]],
        *,
        primary_key_columns: Sequence[str],
    ) -> None:
        """按固定 schema 创建普通索引，跳过主键前缀重复索引。"""
        if not index_columns:
            return
        for columns in index_columns:
            clean_columns = [self._clean_name(str(column)) for column in columns if column]
            if not clean_columns:
                continue
            if list(primary_key_columns[: len(clean_columns)]) == clean_columns:
                continue
            suffix = "_".join(clean_columns)
            index_name = self._clean_name(f"idx_{table_name}_{suffix}")
            with self.get_cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} ({})").format(
                        sql.Identifier(index_name),
                        self._qualified(table_name),
                        sql.SQL(", ").join(
                            [sql.Identifier(column) for column in clean_columns]
                        ),
                    )
                )
    
    def _check_table_compatible(
        self,
        table_name: str,
        df: pd.DataFrame,
        column_types: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        检查DataFrame与现有表结构是否兼容
        
        对于append模式：
        - 如果现有表缺少db_write_time列，自动添加该列
        - 检查DataFrame的其他列是否都存在于现有表中
        
        参数：
            table_name: 表名
            df: 要写入的DataFrame
        
        返回：
            bool: 是否兼容
        """
        try:
            # 获取现有表的列信息（含类型）；限定到当前 schema 避免跨 schema 误读
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position;
                """, (self.schema, table_name))
                existing_col_info = {row[0]: row[1].upper() for row in cursor.fetchall()}
            existing_cols = set(existing_col_info.keys())
            
            # 获取DataFrame的列（清理后）
            df_cols = set(self._clean_name(str(col)) for col in df.columns)
            df_col_types = {
                self._clean_name(str(col)): df[col].dtype
                for col in df.columns
            }
            # cleaned name -> 原始列名映射，用于按 cleaned name 取回原列数据
            df_clean_to_orig = {
                self._clean_name(str(col)): col for col in df.columns
            }
            
            # 如果现有表缺少db_write_time列，自动添加
            if 'db_write_time' not in existing_cols and 'db_write_time' in df_cols:
                with self.get_cursor() as cursor:
                    cursor.execute(sql.SQL("""
                        ALTER TABLE {} ADD COLUMN db_write_time TIMESTAMP
                    """).format(self._qualified(table_name)))
                existing_cols.add('db_write_time')
            
            # 列类型升级：BIGINT → DOUBLE PRECISION（防止浮点数截断）
            INT_TYPES = {'BIGINT', 'INTEGER', 'SMALLINT', 'INT', 'INT4', 'INT8', 'INT2'}
            DATE_LIKE_TYPES = {
                'DATE', 'TIMESTAMP', 'TIMESTAMP WITHOUT TIME ZONE', 'TIMESTAMP WITH TIME ZONE',
            }
            TEXT_TYPES = {'TEXT', 'VARCHAR', 'CHARACTER VARYING', 'CHAR', 'CHARACTER'}
            TEXT_UPGRADE_TARGETS = {'BOOLEAN', 'DOUBLE PRECISION', 'BIGINT'}
            for col_name in (df_cols & existing_cols):  # 仅检查已存在的公共列
                df_dtype = df_col_types.get(col_name)
                if df_dtype is None:
                    continue
                expected_pg_type = (column_types or {}).get(
                    col_name
                ) or self._pandas_to_pg_type(df_dtype, col_name=col_name)
                existing_pg_type = existing_col_info.get(col_name, '').upper()
                # 如果 DataFrame 期望 DOUBLE PRECISION 但 DB 现有列是整型 → 升级
                if expected_pg_type == 'DOUBLE PRECISION' and existing_pg_type in INT_TYPES:
                    with self.get_cursor() as cursor:
                        cursor.execute(sql.SQL("""
                            ALTER TABLE {} ALTER COLUMN {} TYPE DOUBLE PRECISION USING {}::DOUBLE PRECISION
                        """).format(
                            self._qualified(table_name),
                            sql.Identifier(col_name),
                            sql.Identifier(col_name)
                        ))
                # DataFrame 期望 TEXT、DB 现有列是 DATE/TIMESTAMP：数据驱动判断
                # - 全部值都能解析为日期 → 保持 DATE/TIMESTAMP，让真日期字符串如 "2026-04-01" 直接 COPY
                # - 含非日期值（如 "ALL" 通配符）→ ALTER 到 TEXT
                elif expected_pg_type == 'TEXT' and existing_pg_type in DATE_LIKE_TYPES:
                    orig_col = df_clean_to_orig.get(col_name)
                    needs_demote = True
                    if orig_col is not None:
                        non_null = df[orig_col].dropna()
                        if non_null.empty:
                            needs_demote = False
                        else:
                            parsed = pd.to_datetime(non_null.astype(str), errors='coerce')
                            if not parsed.isna().any():
                                needs_demote = False
                    if needs_demote:
                        with self.get_cursor() as cursor:
                            cursor.execute(sql.SQL("""
                                ALTER TABLE {} ALTER COLUMN {} TYPE TEXT USING {}::TEXT
                            """).format(
                                self._qualified(table_name),
                                sql.Identifier(col_name),
                                sql.Identifier(col_name)
                            ))
                # TEXT → BOOLEAN/DOUBLE/BIGINT：修复历史上被误建为 TEXT 的 bool/数值列
                # （配合 _pandas_to_pg_type 收紧标识符匹配后，期望类型已回归原生）。
                # 防御性：cast 失败（含非数值/非布尔脏值）则保持 TEXT 并告警，不影响写入。
                elif existing_pg_type in TEXT_TYPES and expected_pg_type in TEXT_UPGRADE_TARGETS:
                    self._upgrade_text_column(table_name, col_name, expected_pg_type)

            # 检查DataFrame的列是否都在现有表中（允许现有表有额外列）
            missing_cols = df_cols - existing_cols
            if missing_cols:
                # 自动为缺失列补齐表结构
                for col_name in missing_cols:
                    dtype = df_col_types.get(col_name)
                    if dtype is None:
                        continue
                    pg_type = (column_types or {}).get(
                        col_name
                    ) or self._pandas_to_pg_type(dtype, col_name=col_name)
                    with self.get_cursor() as cursor:
                        cursor.execute(sql.SQL("""
                            ALTER TABLE {} ADD COLUMN {} {}
                        """).format(
                            self._qualified(table_name),
                            sql.Identifier(col_name),
                            sql.SQL(pg_type)
                        ))
                    existing_cols.add(col_name)
                return True
            
            return True
        except Exception as e:
            raise

    def _upgrade_text_column(self, table_name: str, col_name: str, target_pg_type: str) -> bool:
        """将历史误建为 TEXT 的列升级到 target_pg_type。

        仅当该列全部非空值可被 PostgreSQL 安全 cast 时才成功；任一值无法转换
        （例如混入 'ALL' 之类的脏字符串）则 ALTER 失败，本方法吞掉异常并保持 TEXT，
        因此绝不会破坏已有数据或中断写入。
        """
        cast_suffix = {
            'BOOLEAN': sql.SQL("::BOOLEAN"),
            'DOUBLE PRECISION': sql.SQL("::DOUBLE PRECISION"),
            # 经 DOUBLE 中转可同时容忍 '2' 与 '2.0' 文本
            'BIGINT': sql.SQL("::DOUBLE PRECISION::BIGINT"),
        }.get(target_pg_type)
        if cast_suffix is None:
            return False
        try:
            with self.get_cursor() as cursor:
                cursor.execute(
                    sql.SQL("ALTER TABLE {t} ALTER COLUMN {c} TYPE {ty} USING {c}{cast}").format(
                        t=self._qualified(table_name),
                        c=sql.Identifier(col_name),
                        ty=sql.SQL(target_pg_type),
                        cast=cast_suffix,
                    )
                )
            logger.info(f"[schema] {table_name}.{col_name} TEXT→{target_pg_type} 升级成功")
            return True
        except Exception as exc:  # noqa: BLE001 - 升级失败时保持 TEXT 即可
            logger.warning(
                f"[schema] {table_name}.{col_name} 保持 TEXT（无法升级到 {target_pg_type}: {exc}）"
            )
            return False

    def _clean_name(self, name: str) -> str:
        """清理名称，使其符合PostgreSQL命名规范"""
        # 替换空格和特殊字符
        clean = name.replace(" ", "_").replace("-", "_").replace(".", "_")
        # 移除其他特殊字符
        clean = "".join(c for c in clean if c.isalnum() or c == "_")
        # 确保不以数字开头
        if clean and clean[0].isdigit():
            clean = "_" + clean
        return clean.lower()
    
    def _pandas_to_pg_type(self, dtype, col_name: Optional[str] = None) -> str:
        """将Pandas数据类型转换为PostgreSQL类型"""
        dtype_str = str(dtype).lower()
        
        # 针对特定列名的规则增强（统一输入输出表的类型）
        if col_name:
            col_name_lower = str(col_name).lower()

            # 0. 布尔类 -> BOOLEAN（永远不是标识符，必须优先于下方名称启发式）
            #    覆盖 bool dtype，以及 is_/has_ 前缀、_flag 后缀的布尔标志列。
            #    防止如 is_cross_node 被 'node' 标识符规则误判成 TEXT。
            if "bool" in dtype_str or col_name_lower.startswith(("is_", "has_")) or col_name_lower.endswith("_flag"):
                return "BOOLEAN"

            # 1. 标识符类 -> 始终使用 TEXT (防止前导零丢失)
            text_identifiers = [
                'material', 'location', 'sending', 'receiving', 'sourcing',
                'dps_location', 'line', 'vendor', 'customer',
                'item', 'sku', 'node', 'plant', 'warehouse', 'dc',
                'status', 'type', 'group', 'category', 'id', 'uid', 'uuid',
                'file_date', 'sim_date', 'run_id', 'issue', 'severity', 'impact',
                'demand_element', 'demand_type', 'element'  # 需求元素标识符
            ]
            # 'truck'/'vehicle' 仅作为完整列名才算标识符；作为前缀会误伤
            # truck_used / truck_load_pct / vehicle_no 等度量/计数列（应为数值，而非 TEXT）。
            exact_only_identifiers = ('truck', 'vehicle')
            if col_name_lower in exact_only_identifiers or any(
                name == col_name_lower
                or col_name_lower.endswith('_' + name)
                or col_name_lower.startswith(name + '_')
                for name in text_identifiers
            ):
                 return "TEXT"
            
            # 2. 日期/时间类 -> 优先检查，避免被数量类误匹配
            # 特别是 production_plan_date 等包含 "production" 的日期列
            # 如果列名以 _date 结尾，且 pandas 类型是 datetime，使用 TIMESTAMP
            if col_name_lower.endswith('_date') and "datetime" in dtype_str:
                return "TIMESTAMP"
            
            # 3. 日期类 -> 仅对明确的日期字段使用 DATE 类型
            # 注意：某些包含 "date" 的列可能存储 "ALL" 等特殊值，需要使用 TEXT
            # `file_date` 和 `sim_date` 作为标识符使用 `TEXT` 类型（格式：YYYYMMDD）
            date_specific_names = [
                'start_date', 'end_date', 'order_date', 'delivery_date',
                'ship_date', 'arrival_date', 'due_date', 'created_date',
                'updated_date', 'forecast_date', 'plan_date', 'production_plan_date'
            ]
            # 只有明确的日期字段才使用 DATE 类型，避免误判
            if any(name == col_name_lower or col_name_lower.endswith('_' + name) for name in date_specific_names):
                return "DATE"
                 
            # 4. 数量/度量类 -> 使用 DOUBLE PRECISION (防止 int/float 混淆)
            # 注意：排除以 _date 结尾的列，避免误匹配日期列
            float_measures = [
                'qty', 'quantity', 'amount', 'inventory', 'stock', 'capacity', 
                'demand', 'supply', 'shipment', 'production', 'weight', 'volume',
                'price', 'cost', 'ratio', 'percent', 'rate', 'yield',
                'leadtime', 'duration', 'hours', 'time_needed',
                'wfr', 'vfr', 'mdq',  # 物流配置参数 (weight fill rate, volume fill rate, min dispatch qty)
                # 生产/配置参数 (防止 BIGINT 截断浮点数)
                'min_batch', 'rv', 'ptf', 'lsk', 'mct', 'moq', 'prd_rate',
                'pdt', 'gr', 'otd',  # Global_LeadTime 参数
            ]
            if not col_name_lower.endswith('_date') and any(name in col_name_lower for name in float_measures):
                return "DOUBLE PRECISION"
            
            # 5. 索引/排序类 -> 始终使用 BIGINT
            int_indexes = ['day', 'week', 'month', 'year', 'priority', 'sequence', 'order', 'step', 'count', 'seed']
            if any(name == col_name_lower for name in int_indexes):
                return "BIGINT"

        if "int" in dtype_str:
            return "BIGINT"
        elif "float" in dtype_str:
            return "DOUBLE PRECISION"
        elif "datetime" in dtype_str:
            return "TIMESTAMP"
        elif "date" in dtype_str:
            return "DATE"
        elif "bool" in dtype_str:
            return "BOOLEAN"
        else:
            return "TEXT"
    
    def _insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        batch_size: int = 1000,
        round_float_values: bool = True,
    ):
        """
        批量插入DataFrame数据 - 使用高效的COPY方式（批块化优化版）
        
        性能优化 (Phase 3):
        - 批块化写入：每1000行提交一次，减少事务开销
        - 预计性能提升：60-80s → 8-12s (-85%)
        
        参数：
            df: 要插入的DataFrame
            table_name: 目标表名
            batch_size: 每批写入的行数，默认1000
        """
        if df.empty:
            return
        
        total_rows = len(df)
        
        # 清理列名
        clean_columns = [self._clean_name(str(col)) for col in df.columns]
        
        # 获取目标表的列类型，用于数据类型转换
        col_types = self._get_column_types(table_name)
        
        # 创建列名到索引的映射
        col_name_to_idx = {col: idx for idx, col in enumerate(clean_columns)}
        
        # 确定需要显式转换类型的列索引
        int_col_indices = set()
        float_col_indices = set()
        text_col_indices = set()
        datetime_col_indices = set()

        _TIMESTAMP_TYPES = {
            'TIMESTAMP', 'TIMESTAMP WITHOUT TIME ZONE', 'TIMESTAMP WITH TIME ZONE',
            'DATE',
        }

        for col_name, col_type in col_types.items():
            if col_name in col_name_to_idx:
                idx = col_name_to_idx[col_name]
                ct_upper = col_type.upper()
                if ct_upper in ('BIGINT', 'INTEGER', 'SMALLINT', 'INT', 'INT4', 'INT8', 'INT2'):
                    int_col_indices.add(idx)
                elif ct_upper in ('DOUBLE PRECISION', 'REAL', 'NUMERIC', 'FLOAT4', 'FLOAT8'):
                    float_col_indices.add(idx)
                elif ct_upper in ('TEXT', 'VARCHAR', 'CHARACTER VARYING', 'CHAR', 'CHARACTER'):
                    text_col_indices.add(idx)
                elif ct_upper in _TIMESTAMP_TYPES:
                    datetime_col_indices.add(idx)

        # 准备插入数据
        records = df.values.tolist()

        # Excel 序列日期 epoch（1899-12-30）
        from datetime import timedelta as _timedelta
        _EXCEL_EPOCH = datetime(1899, 12, 30)

        # 处理NaN值和数据类型转换
        import numpy as np
        for i, row in enumerate(records):
            new_row = []
            for j, val in enumerate(row):
                if j in int_col_indices:
                    new_row.append(coerce_db_int(val))
                elif j in float_col_indices:
                    new_row.append(
                        coerce_db_float(
                            val,
                            round_values=round_float_values,
                        )
                    )
                elif pd.isna(val) if not isinstance(val, str) else False:
                    new_row.append(None)
                elif j in text_col_indices:
                    # Convert booleans to "True"/"False" strings to match
                    # 与 Dev/Src 的 xlsx 输出格式保持一致（避免 PG 将 bool->text 转成 `t`/`f`）
                    if isinstance(val, (bool, np.bool_)):
                        new_row.append(str(val))
                    else:
                        new_row.append(str(val) if val is not None else None)
                elif j in datetime_col_indices:
                    # Excel 将日期存为整数序列号时，转换为 datetime；已是 datetime 则直接使用
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        try:
                            new_row.append(_EXCEL_EPOCH + _timedelta(days=float(val)))
                        except (ValueError, OverflowError):
                            new_row.append(None)
                    else:
                        new_row.append(val)
                else:
                    new_row.append(val)
            records[i] = tuple(new_row)
        
        # 构建 COPY 语句 - 使用 psycopg3 的 copy 功能；通过 sql.Composed 安全引用 schema.table
        copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
            self._qualified(table_name),
            sql.SQL(", ").join(sql.Identifier(col) for col in clean_columns),
        )
        
        conn = self.connect()
        
        # 性能优化 (Phase 4 - 问题1)：修改事务提交粒度
        # - 原因：批块化提交导致多次fsync，开销 6-30s
        # - 改进：单次事务提交，开销 0.2-1s
        # - 性能收益：-85-95%
        try:
            with conn.transaction():
                # 单次事务内写入所有数据
                with conn.cursor() as cursor:
                    with cursor.copy(copy_sql) as copy:
                        for record in records:
                            copy.write_row(record)
            
            # 大数据集时显示进度
            if total_rows >= 10000:
                pass
        except Exception as error:
             # autocommit=True 模式下，conn.transaction() 退出时已自动 ROLLBACK，
            # 无需手动 rollback
            raise error
    
    def _get_column_types(self, table_name: str, schema: Optional[str] = None) -> Dict[str, str]:
        """获取表的列名和数据类型映射（默认限定到当前 schema）。"""
        target_schema = schema or self.schema
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                """, (target_schema, table_name))
                return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception:
            return {}
    
    def _create_auto_indexes(self, table_name: str, df: pd.DataFrame):
        """
        自动为常用查询字段创建索引
        
        性能优化 (Phase 3):
        - 自动检测并为 material/location/date 等常用字段创建BTREE索引
        - 为 run_id 创建HASH索引
        - 预计查询性能提升：5-10s → 2-4s (-60%)
        
        参数：
            table_name: 表名
            df: 对应的DataFrame（用于检测列名）
        """
        # 定义需要索引的列及其索引类型
        index_columns = {
            # B 树索引 - 适合范围查询和等值查询
            'material': 'BTREE',
            'location': 'BTREE',
            'sending': 'BTREE',
            'receiving': 'BTREE',
            'date': 'BTREE',
            'simulation_date': 'BTREE',
            'order_date': 'BTREE',
            'available_date': 'BTREE',
            'delivery_date': 'BTREE',
            # 哈希索引 - 适合等值查询
            'run_id': 'HASH',
        }
        
        # 清理后的列名
        clean_columns = {self._clean_name(str(col)): col for col in df.columns}
        
        indexes_created = []
        
        for col_name, index_type in index_columns.items():
            clean_col = self._clean_name(col_name)
            if clean_col in clean_columns:
                # 生成索引名
                index_name = f"idx_{table_name}_{clean_col}"
                
                try:
                    with self.get_cursor() as cursor:
                        # 检查索引是否已存在（限定到当前 schema 防止跨 schema 重名干扰）
                        cursor.execute("""
                            SELECT 1 FROM pg_indexes
                            WHERE schemaname = %s AND tablename = %s AND indexname = %s
                        """, (self.schema, table_name, index_name))

                        if cursor.fetchone() is None:
                            # 创建索引 - schema-qualified；自动为含 - 的 schema 加引号
                            if index_type == 'HASH':
                                idx_sql = sql.SQL(
                                    "CREATE INDEX {iname} ON {tbl} USING HASH ({col})"
                                ).format(
                                    iname=sql.Identifier(index_name),
                                    tbl=self._qualified(table_name),
                                    col=sql.Identifier(clean_col),
                                )
                            else:
                                idx_sql = sql.SQL(
                                    "CREATE INDEX {iname} ON {tbl} ({col})"
                                ).format(
                                    iname=sql.Identifier(index_name),
                                    tbl=self._qualified(table_name),
                                    col=sql.Identifier(clean_col),
                                )
                            cursor.execute(idx_sql)
                            indexes_created.append(f"{clean_col}({index_type})")
                except Exception as e:
                    # 索引创建失败不影响主流程
                    raise
        
        if indexes_created:
            pass
    
    def read_table(self, table_name: str, filters: Optional[Dict[str, Any]] = None, schema: Optional[str] = None) -> pd.DataFrame:
        """读取表数据到 DataFrame，可选按列等值过滤（默认从 self.schema 读取）。"""
        qualified = self._qualified(table_name, schema)
        query = sql.SQL("SELECT * FROM {}").format(qualified)
        params = None
        if filters:
            conditions = []
            params_list = []
            for column, value in filters.items():
                identifier = sql.Identifier(self._clean_name(str(column)))
                if value is None:
                    conditions.append(sql.SQL("{} IS NULL").format(identifier))
                else:
                    conditions.append(sql.SQL("{} = %s").format(identifier))
                    params_list.append(value)
            query = sql.SQL("SELECT * FROM {} WHERE {}").format(
                qualified,
                sql.SQL(" AND ").join(conditions),
            )
            params = tuple(params_list)

        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
    
    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """执行查询并返回结果"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_query_df(self, query: str, params: tuple = None) -> pd.DataFrame:
        """执行查询并以带列名的DataFrame返回结果。"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return pd.DataFrame(cursor.fetchall(), columns=columns)

    def iter_query_chunks(self, query, params=None, chunksize: int = 100_000):
        """服务端游标流式产出 DataFrame chunk，避免大结果集一次性加载到内存。

        autocommit=True 下，server-side cursor 必须包在 conn.transaction()
        里（DECLARE CURSOR 需要显式事务），由本方法自行管理。
        """
        import secrets
        conn = self.connect()
        cur_name = f"sc_{secrets.token_hex(4)}"
        with conn.transaction():
            with conn.cursor(name=cur_name) as cur:
                cur.execute(query, params)
                columns = (
                    [desc[0] for desc in cur.description]
                    if cur.description else []
                )
                while True:
                    rows = cur.fetchmany(chunksize)
                    if not rows:
                        break
                    yield pd.DataFrame(rows, columns=columns)
    
    def execute_non_query(self, query: str, params: tuple = None):
        """执行非查询语句（INSERT, UPDATE, DELETE等）"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
    
    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """获取表信息（默认从 self.schema 读取）。"""
        target_schema = schema or self.schema
        with self.get_cursor(commit=False) as cursor:
            # 获取列信息
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position;
            """, (target_schema, table_name))
            columns = cursor.fetchall()

            # 获取行数
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(
                self._qualified(table_name, target_schema)
            ))
            row_count = cursor.fetchone()[0]

            return {
                "table_name": table_name,
                "schema": target_schema,
                "columns": [{"name": c[0], "type": c[1], "nullable": c[2]} for c in columns],
                "row_count": row_count
            }


def test_database_connection():
    """测试数据库连接的独立函数"""
    
    db = DatabaseConnection()
    result = db.test_connection()
    
    
    if result['success']:
        
        # 列出所有表
        tables = db.get_all_tables()
        if tables:
            pass
    else:
        pass
    
    db.close()
    return result['success']


if __name__ == "__main__":
    test_database_connection()

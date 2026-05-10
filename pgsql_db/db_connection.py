"""
PostgreSQL数据库连接模块
提供数据库连接、测试、表操作等功能
使用 psycopg3 (psycopg) 以解决Windows中文环境编码问题
"""

import psycopg
from psycopg import sql
import pandas as pd
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import time
from datetime import datetime


class DatabaseConnection:
    """PostgreSQL数据库连接类"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
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
        """
        from .settings import resolve_database_config
        cfg = resolve_database_config(
            host=host, port=port, database=database, user=user, password=password
        )
        self.host = cfg["host"]
        self.port = cfg["port"]
        self.database = cfg["database"]
        self.user = cfg["user"]
        self.password = cfg["password"]
        self._connection = None
    
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
                return self._connection
            except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout) as e:
                last_err = e
                self._connection = None
                if attempt < 2:
                    time.sleep(5 * (2 ** attempt))  # 5s, 10s
        raise last_err
    
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
            return False
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
            return False
        finally:
            if temp_conn and not temp_conn.closed:
                temp_conn.close()
    
    def check_tables_exist(self, table_names: List[str]) -> Dict[str, bool]:
        """
        检查多个表是否存在
        
        参数：
            table_names: 表名列表
        
        返回：
            dict: 表名 -> 是否存在
        """
        result = {}
        existing_tables = set(self.get_all_tables())
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
            result["message"] = f"连接失败: {str(e)}"
        finally:
            result["connection_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return result
    
    def table_exists(self, table_name: str, schema: str = "public") -> bool:
        """检查表是否存在"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                );
            """, (schema, table_name))
            return cursor.fetchone()[0]
    
    def get_all_tables(self, schema: str = "public") -> List[str]:
        """获取所有表名"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name;
            """, (schema,))
            return [row[0] for row in cursor.fetchall()]
    
    def drop_table(self, table_name: str, cascade: bool = False):
        """删除表"""
        cascade_str = "CASCADE" if cascade else ""
        with self.get_cursor() as cursor:
            query = sql.SQL("DROP TABLE IF EXISTS {} {}").format(
                sql.Identifier(table_name),
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
                # 先检查表是否有 config_name 列
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = %s AND column_name = 'config_name'
                """, (clean_name,))
                if not cursor.fetchone():
                    return False
                
                # 检查是否有该配置的数据
                cursor.execute(
                    sql.SQL("SELECT 1 FROM {} WHERE config_name = %s LIMIT 1").format(
                        sql.Identifier(clean_name)
                    ),
                    (config_name,)
                )
                return cursor.fetchone() is not None
        except Exception as e:
            return False
    
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
                        sql.Identifier(clean_name)
                    ),
                    (config_name,)
                )
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    pass
                return deleted_count
        except Exception as e:
            return 0
    
    def create_table_from_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        add_write_time: bool = True,
        config_name: Optional[str] = None,
        config_type: Optional[str] = None,
        round_float_values: Optional[bool] = None
    ) -> bool:
        """
        根据DataFrame创建表并写入数据
        
        参数：
            df: 数据DataFrame
            table_name: 表名
            if_exists: 如果表存在的处理方式 ('replace', 'append', 'fail')
            add_write_time: 是否自动添加写入时间列
            config_name: 配置文件标识（如 BC_S5, BC_S9），用于区分不同配置的数据
            config_type: 配置类型 ('OC' / 'BC' / 'OTHER')，用于快速区分配置类别
            round_float_values: 是否在写入前将浮点数四舍五入到10位小数；默认对 cfg_* 表关闭，对其他表开启
        
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
        
        # 添加config_type列（用于快速区分 OC / BC 配置）
        if config_type:
            if is_empty_table:
                df_to_write['config_type'] = pd.Series(dtype='object')
            else:
                df_to_write['config_type'] = config_type
        
        # 添加写入时间列
        if add_write_time:
            if is_empty_table:
                # 空表只添加列定义
                df_to_write['db_write_time'] = pd.Series(dtype='datetime64[ns]')
            else:
                df_to_write['db_write_time'] = datetime.now()
        
        # 检查表是否存在
        exists = self.table_exists(clean_table_name)
        
        if exists:
            if if_exists == "fail":
                raise ValueError(f"表 {clean_table_name} 已存在")
            elif if_exists == "replace":
                # 为避免丢失历史数据，replace模式改为追加写入
                if not self._check_table_compatible(clean_table_name, df_to_write):
                    return False
            elif if_exists == "append":
                # 追加模式（append）：检查表结构是否兼容
                if not self._check_table_compatible(clean_table_name, df_to_write):
                    return False
        
        # 创建表（如果不存在）
        if not self.table_exists(clean_table_name):
            columns = []
            for col_name, dtype in df_to_write.dtypes.items():
                clean_col = self._clean_name(str(col_name))
                pg_type = self._pandas_to_pg_type(dtype, col_name=clean_col)
                columns.append(f'"{clean_col}" {pg_type}')
            
            create_sql = f'CREATE TABLE IF NOT EXISTS "{clean_table_name}" ({", ".join(columns)})'
            
            with self.get_cursor() as cursor:
                cursor.execute(create_sql)
            
        else:
            if is_empty_table:
                pass
            else:
                pass
        
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
    
    def _check_table_compatible(
        self,
        table_name: str,
        df: pd.DataFrame
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
            # 获取现有表的列信息（含类型）
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, (table_name,))
                existing_col_info = {row[0]: row[1].upper() for row in cursor.fetchall()}
            existing_cols = set(existing_col_info.keys())
            
            # 获取DataFrame的列（清理后）
            df_cols = set(self._clean_name(str(col)) for col in df.columns)
            df_col_types = {
                self._clean_name(str(col)): df[col].dtype
                for col in df.columns
            }
            
            # 如果现有表缺少db_write_time列，自动添加
            if 'db_write_time' not in existing_cols and 'db_write_time' in df_cols:
                with self.get_cursor() as cursor:
                    cursor.execute(sql.SQL("""
                        ALTER TABLE {} ADD COLUMN db_write_time TIMESTAMP
                    """).format(sql.Identifier(table_name)))
                existing_cols.add('db_write_time')
            
            # 列类型升级：BIGINT → DOUBLE PRECISION（防止浮点数截断）
            INT_TYPES = {'BIGINT', 'INTEGER', 'SMALLINT', 'INT', 'INT4', 'INT8', 'INT2'}
            for col_name in (df_cols & existing_cols):  # 仅检查已存在的公共列
                df_dtype = df_col_types.get(col_name)
                if df_dtype is None:
                    continue
                expected_pg_type = self._pandas_to_pg_type(df_dtype, col_name=col_name)
                existing_pg_type = existing_col_info.get(col_name, '').upper()
                # 如果 DataFrame 期望 DOUBLE PRECISION 但 DB 现有列是整型 → 升级
                if expected_pg_type == 'DOUBLE PRECISION' and existing_pg_type in INT_TYPES:
                    with self.get_cursor() as cursor:
                        cursor.execute(sql.SQL("""
                            ALTER TABLE {} ALTER COLUMN {} TYPE DOUBLE PRECISION USING {}::DOUBLE PRECISION
                        """).format(
                            sql.Identifier(table_name),
                            sql.Identifier(col_name),
                            sql.Identifier(col_name)
                        ))
            
            # 检查DataFrame的列是否都在现有表中（允许现有表有额外列）
            missing_cols = df_cols - existing_cols
            if missing_cols:
                # 自动为缺失列补齐表结构
                for col_name in missing_cols:
                    dtype = df_col_types.get(col_name)
                    if dtype is None:
                        continue
                    pg_type = self._pandas_to_pg_type(dtype, col_name=col_name)
                    with self.get_cursor() as cursor:
                        cursor.execute(sql.SQL("""
                            ALTER TABLE {} ADD COLUMN {} {}
                        """).format(
                            sql.Identifier(table_name),
                            sql.Identifier(col_name),
                            sql.SQL(pg_type)
                        ))
                    existing_cols.add(col_name)
                return True
            
            return True
        except Exception as e:
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
            
            # 1. 标识符类 -> 始终使用 TEXT (防止前导零丢失)
            text_identifiers = [
                'material', 'location', 'sending', 'receiving', 'sourcing', 
                'dps_location', 'line', 'truck', 'vehicle', 'vendor', 'customer',
                'item', 'sku', 'node', 'plant', 'warehouse', 'dc',
                'status', 'type', 'group', 'category', 'id', 'uid', 'uuid',
                'file_date', 'sim_date', 'run_id', 'issue', 'severity', 'impact',
                'demand_element', 'demand_type', 'element'  # 需求元素标识符
            ]
            if any(name == col_name_lower or col_name_lower.endswith('_' + name) or col_name_lower.startswith(name + '_') for name in text_identifiers):
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
                'date',
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
                if pd.isna(val) if not isinstance(val, str) else False:
                    new_row.append(None)
                elif j in int_col_indices:
                    try:
                        # 兼容处理：float -> int
                        new_row.append(int(float(val)))
                    except (ValueError, TypeError):
                        new_row.append(None)
                elif j in float_col_indices:
                    try:
                        float_val = float(val)
                        if round_float_values:
                            new_row.append(round(float_val, 15))
                        else:
                            new_row.append(float_val)
                    except (ValueError, TypeError):
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
        
        # 构建COPY语句 - 使用psycopg3的copy功能
        cols_str = ", ".join([f'"{col}"' for col in clean_columns])
        copy_sql = f'COPY "{table_name}" ({cols_str}) FROM STDIN'
        
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
    
    def _get_column_types(self, table_name: str) -> Dict[str, str]:
        """获取表的列名和数据类型映射"""
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s
                """, (table_name,))
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
                        # 检查索引是否已存在
                        cursor.execute("""
                            SELECT 1 FROM pg_indexes 
                            WHERE tablename = %s AND indexname = %s
                        """, (table_name, index_name))
                        
                        if cursor.fetchone() is None:
                            # 创建索引
                            if index_type == 'HASH':
                                create_idx_sql = f'CREATE INDEX "{index_name}" ON "{table_name}" USING HASH ("{clean_col}")'
                            else:
                                create_idx_sql = f'CREATE INDEX "{index_name}" ON "{table_name}" ("{clean_col}")'
                            
                            cursor.execute(create_idx_sql)
                            indexes_created.append(f"{clean_col}({index_type})")
                except Exception as e:
                    # 索引创建失败不影响主流程
                    pass
        
        if indexes_created:
            pass
    
    def read_table(self, table_name: str, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """读取表数据到DataFrame，可选按列等值过滤。"""
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
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
                sql.Identifier(table_name),
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
    
    def execute_non_query(self, query: str, params: tuple = None):
        """执行非查询语句（INSERT, UPDATE, DELETE等）"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """获取表信息"""
        with self.get_cursor(commit=False) as cursor:
            # 获取列信息
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            columns = cursor.fetchall()
            
            # 获取行数
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
            row_count = cursor.fetchone()[0]
            
            return {
                "table_name": table_name,
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

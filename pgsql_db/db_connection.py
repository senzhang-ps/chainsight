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
        host: str = "localhost",
        port: int = 5432,
        database: str = "test_db",
        user: str = "postgres",
        password: str = "123456"
    ):
        """
        初始化数据库连接参数
        
        Args:
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._connection = None
    
    @property
    def connection_string(self) -> str:
        """返回连接字符串"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def connect(self) -> psycopg.Connection:
        """建立数据库连接"""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
                client_encoding='UTF8'
            )
        return self._connection
    
    def close(self):
        """关闭数据库连接"""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
    
    @contextmanager
    def get_cursor(self, commit: bool = True):
        """
        获取数据库游标的上下文管理器
        
        Args:
            commit: 是否自动提交
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试数据库连接
        
        Returns:
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
            print(f"✅已删除表: {table_name}")
    
    def create_table_from_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        add_write_time: bool = True
    ) -> bool:
        """
        根据DataFrame创建表并写入数据
        
        Args:
            df: 数据DataFrame
            table_name: 表名
            if_exists: 如果表存在的处理方式 ('replace', 'append', 'fail')
            add_write_time: 是否自动添加写入时间列
        
        Returns:
            bool: 是否成功
        """
        if df.empty:
            print(f"⚠️DataFrame为空，跳过创建表: {table_name}")
            return False
        
        # 清理表名（去除特殊字符）
        clean_table_name = self._clean_name(table_name)
        
        # 添加写入时间列
        df_to_write = df.copy()
        if add_write_time:
            df_to_write['db_write_time'] = datetime.now()
        
        # 检查表是否存在
        exists = self.table_exists(clean_table_name)
        
        if exists:
            if if_exists == "fail":
                raise ValueError(f"表 {clean_table_name} 已存在")
            elif if_exists == "replace":
                self.drop_table(clean_table_name)
                print(f"✅已删除表: {clean_table_name}")
            elif if_exists == "append":
                # append模式：检查表结构是否兼容
                if not self._check_table_compatible(clean_table_name, df_to_write):
                    print(f"⚠️表结构不兼容，跳过追加: {clean_table_name}")
                    return False
        
        # 创建表（如果不存在）
        if not self.table_exists(clean_table_name):
            columns = []
            for col_name, dtype in df_to_write.dtypes.items():
                clean_col = self._clean_name(str(col_name))
                pg_type = self._pandas_to_pg_type(dtype)
                columns.append(f'"{clean_col}" {pg_type}')
            
            create_sql = f'CREATE TABLE IF NOT EXISTS "{clean_table_name}" ({", ".join(columns)})'
            
            with self.get_cursor() as cursor:
                cursor.execute(create_sql)
            
            print(f"✅已创建表: {clean_table_name} ({len(df_to_write)} 行, {len(df_to_write.columns)} 列)")
        else:
            print(f"✅追加数据到表: {clean_table_name} (+{len(df_to_write)} 行)")
        
        # 插入数据
        self._insert_dataframe(df_to_write, clean_table_name)
        
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
        
        Args:
            table_name: 表名
            df: 要写入的DataFrame
        
        Returns:
            bool: 是否兼容
        """
        try:
            # 获取现有表的列信息
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, (table_name,))
                existing_cols = set(row[0] for row in cursor.fetchall())
            
            # 获取DataFrame的列（清理后）
            df_cols = set(self._clean_name(str(col)) for col in df.columns)
            
            # 如果现有表缺少db_write_time列，自动添加
            if 'db_write_time' not in existing_cols and 'db_write_time' in df_cols:
                print(f"🔧 为表 {table_name} 添加 db_write_time 列")
                with self.get_cursor() as cursor:
                    cursor.execute(sql.SQL("""
                        ALTER TABLE {} ADD COLUMN db_write_time TIMESTAMP
                    """).format(sql.Identifier(table_name)))
                existing_cols.add('db_write_time')
            
            # 检查DataFrame的列是否都在现有表中（允许现有表有额外列）
            missing_cols = df_cols - existing_cols
            if missing_cols:
                print(f"⚠️DataFrame包含现有表中不存在的列: {missing_cols}")
                return False
            
            return True
        except Exception as e:
            print(f"⚠️检查表结构时出错: {e}")
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
    
    def _pandas_to_pg_type(self, dtype) -> str:
        """将Pandas数据类型转换为PostgreSQL类型"""
        dtype_str = str(dtype)
        
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
    
    def _insert_dataframe(self, df: pd.DataFrame, table_name: str):
        """批量插入DataFrame数据 - 使用高效的COPY方式"""
        if df.empty:
            return
        
        # 清理列名
        clean_columns = [self._clean_name(str(col)) for col in df.columns]
        
        # 获取目标表的列类型，用于数据类型转换
        col_types = self._get_column_types(table_name)
        
        # 创建列名到索引的映射
        col_name_to_idx = {col: idx for idx, col in enumerate(clean_columns)}
        
        # 确定需要转换为整数的列索引
        int_col_indices = set()
        for col_name, col_type in col_types.items():
            if col_name in col_name_to_idx and col_type.upper() in ('BIGINT', 'INTEGER', 'SMALLINT', 'INT', 'INT4', 'INT8', 'INT2'):
                int_col_indices.add(col_name_to_idx[col_name])
        
        # 准备插入数据
        records = df.values.tolist()
        
        # 处理NaN值和数据类型转换
        for i, row in enumerate(records):
            new_row = []
            for j, val in enumerate(row):
                if pd.isna(val):
                    new_row.append(None)
                elif j in int_col_indices and isinstance(val, float):
                    # 将浮点数转换为整数（如 0.0 -> 0）
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            records[i] = tuple(new_row)
        
        # 构建COPY语句 - 使用psycopg3的copy功能
        cols_str = ", ".join([f'"{col}"' for col in clean_columns])
        copy_sql = f'COPY "{table_name}" ({cols_str}) FROM STDIN'
        
        conn = self.connect()
        with conn.cursor() as cursor:
            with cursor.copy(copy_sql) as copy:
                for record in records:
                    copy.write_row(record)
        conn.commit()
    
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
    
    def read_table(self, table_name: str) -> pd.DataFrame:
        """读取表数据到DataFrame"""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name)))
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
    print("=" * 60)
    print("PostgreSQL数据库连接测试")
    print("=" * 60)
    
    db = DatabaseConnection()
    result = db.test_connection()
    
    print(f"主机: {result['host']}:{result['port']}")
    print(f"数据库: {result['database']}")
    print(f"连接状态: {'✅成功' if result['success'] else '❌失败'}")
    print(f"连接耗时: {result['connection_time_ms']}ms")
    
    if result['success']:
        print(f"数据库版本: {result['version'][:50]}...")
        
        # 列出所有表
        tables = db.get_all_tables()
        print(f"现有表数量: {len(tables)}")
        if tables:
            print(f"表列表: {', '.join(tables[:10])}" + ("..." if len(tables) > 10 else ""))
    else:
        print(f"错误信息: {result['message']}")
    
    db.close()
    print("=" * 60)
    return result['success']


if __name__ == "__main__":
    test_database_connection()

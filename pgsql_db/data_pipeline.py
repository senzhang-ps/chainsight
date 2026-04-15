"""
数据处理管道
组合DuckDB数据处理器和PostgreSQL数据库连接
"""

from .duckdb_processor import DuckDBProcessor
from .db_connection import DatabaseConnection
from typing import Dict, Any


class DataPipeline:
    """数据处理管道 - 组合DuckDB和PostgreSQL"""
    
    def __init__(
        self,
        pg_host: str = "localhost",
        pg_port: int = 5432,
        pg_database: str = "test_db",
        pg_user: str = "postgres",
        pg_password: str = "123456",
        duck_db_path: str = ":memory:"
    ):
        """
        初始化数据处理管道
        
        参数：
            pg_host: PostgreSQL主机
            pg_port: PostgreSQL端口
            pg_database: PostgreSQL数据库名
            pg_user: PostgreSQL用户名
            pg_password: PostgreSQL密码
            duck_db_path: DuckDB数据库路径，默认内存模式
        """
        self.pg = DatabaseConnection(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password
        )
        self.duck = DuckDBProcessor(db_path=duck_db_path)
        
        # 统计信息
        self._stats = {
            'tables_processed': 0,
            'rows_processed': 0,
            'tables_written': 0,
            'rows_written': 0
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False
    
    def close(self):
        """关闭所有连接"""
        self.duck.close()
        self.pg.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        return self._stats.copy()
    
    def print_stats(self):
        """打印处理统计"""
    
    def update_stats(
        self, 
        tables_processed: int = 0,
        rows_processed: int = 0,
        tables_written: int = 0,
        rows_written: int = 0
    ):
        """更新统计信息"""
        self._stats['tables_processed'] += tables_processed
        self._stats['rows_processed'] += rows_processed
        self._stats['tables_written'] += tables_written
        self._stats['rows_written'] += rows_written

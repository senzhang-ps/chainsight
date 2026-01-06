"""
PostgreSQL数据库连接组件
用于测试数据库连接、数据导入导出等操作

组件:
- DatabaseConnection: PostgreSQL数据库连接
- ExcelImporter: Excel文件导入器
- ModuleDataWriter: 模块数据写入器
- DuckDBProcessor: DuckDB数据处理器
- DataPipeline: 数据处理管道（DuckDB + PostgreSQL）
- DatabaseInitializer: 数据库初始化器（自动检测、创建数据库和表）
- table_mapping: 表映射配置
"""

from .db_connection import DatabaseConnection
from .excel_importer import ExcelImporter
from .module_data_writer import ModuleDataWriter
from .duckdb_processor import DuckDBProcessor
from .data_pipeline import DataPipeline
from .db_initializer import DatabaseInitializer, initialize_database
from . import table_mapping

__all__ = [
    'DatabaseConnection', 
    'ExcelImporter', 
    'ModuleDataWriter',
    'DuckDBProcessor',
    'DataPipeline',
    'DatabaseInitializer',
    'initialize_database',
    'table_mapping'
]

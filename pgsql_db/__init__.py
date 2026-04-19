"""
PostgreSQL数据库连接组件
用于测试数据库连接、数据导入导出等操作

组件:
- DatabaseConnection: PostgreSQL数据库连接
- ExcelImporter: Excel文件导入器
- ModuleDataWriter: 模块数据写入器
- DatabaseInitializer: 数据库初始化器（自动检测、创建数据库和表）
- OptimizedDataProcessor: 高性能数据处理器（DuckDB向量化）
- ModuleCalculationEngine: 模块计算引擎
- OptimizedSimulationRunner: 优化仿真运行器
- Module3Optimizer/Module5Optimizer/Module6Optimizer: 模块优化器
- PerformanceDashboard: 性能监控仪表盘
- IncrementalProcessor: 增量数据处理器
- table_mapping: 表映射配置
"""

from .db_connection import DatabaseConnection
from .excel_importer import ExcelImporter
from .module_data_writer import ModuleDataWriter
from .db_initializer import DatabaseInitializer, initialize_database
from .optimized_processor import OptimizedDataProcessor
from .module_engine import ModuleCalculationEngine, create_calculation_engine
from .optimized_simulation import (
    OptimizedSimulationRunner,
    ModuleOptimizer,
    create_optimized_runner,
    integrate_optimization_to_simulation
)
from .module_optimizers import (
    Module3Optimizer,
    Module5Optimizer,
    Module6Optimizer,
    create_module_optimizer,
    create_all_optimizers
)
from .performance_dashboard import (
    PerformanceDashboard,
    RealTimeMonitor,
    get_dashboard,
    track_performance
)
from .incremental_processor import (
    IncrementalProcessor,
    ChangeDetector,
    DeltaCalculator,
    create_incremental_processor
)
from . import table_mapping

__all__ = [
    # 数据库连接
    'DatabaseConnection',
    'ExcelImporter',
    'ModuleDataWriter',
    'DatabaseInitializer',
    'initialize_database',
    
    # 高性能处理
    'OptimizedDataProcessor',
    'ModuleCalculationEngine',
    'create_calculation_engine',
    
    # 优化仿真
    'OptimizedSimulationRunner',
    'ModuleOptimizer',
    'create_optimized_runner',
    'integrate_optimization_to_simulation',
    
    # 模块优化器
    'Module3Optimizer',
    'Module5Optimizer',
    'Module6Optimizer',
    'create_module_optimizer',
    'create_all_optimizers',
    
    # 性能监控
    'PerformanceDashboard',
    'RealTimeMonitor',
    'get_dashboard',
    'track_performance',
    
    # 增量处理
    'IncrementalProcessor',
    'ChangeDetector',
    'DeltaCalculator',
    'create_incremental_processor',
    
    # 配置
    'table_mapping'
]

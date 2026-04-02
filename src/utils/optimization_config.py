# -*- coding: utf-8 -*-
"""
统一优化配置模块

集中管理所有性能优化开关，便于调试和性能测试。
支持通过环境变量或配置文件覆盖默认设置。

性能测试结果 (2026-01-23):
========================
测试环境: BC_S5.xlsx, 5天仿真 (2025-10-06 ~ 2025-10-10)

| 版本         | 运行时间     | 相对性能 |
|--------------|-------------|----------|
| ChainSight_Dev (原始) | 460.91秒 (7分41秒) | 1.00x |
| src 本地版 (重构优化) | 323.04秒 (5分23秒) | 1.43x 🚀 |
| src 数据库版 (仿真)   | 164.17秒 (2分44秒) | 2.81x 🚀 |
| src 数据库版 (含写入) | 225.36秒 (3分45秒) | 2.04x 🚀 |

数据一致性: 三个版本输出100%一致 ✅

DuckDB vs Pandas 测试结论:
- MERGE: Pandas 更快 (DuckDB永不建议)
- GROUPBY: Pandas 通常更快 (仅500K+行DuckDB有优势)
- FILTER: Pandas 更快 (DuckDB永不建议)
- SORT: Pandas 更快 (DuckDB永不建议)

使用方法:
    from src.utils.optimization_config import OptimizationConfig
    
    # 检查是否启用DuckDB
    if OptimizationConfig.USE_DUCKDB:
        # 使用DuckDB处理
        ...
    else:
        # 使用Pandas处理
        ...
    
    # 运行时修改配置
    OptimizationConfig.set_duckdb_enabled(False)
"""

import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class _OptimizationConfig:
    """优化配置类"""
    
    # ==================== DuckDB 优化 ====================
    # 主开关：是否启用DuckDB优化
    # 注意：经过性能测试，在内存中处理数据时Pandas通常更快
    # DuckDB优势场景：超大数据集(500K+)、复杂多表JOIN、SQL便利性
    USE_DUCKDB: bool = False  # 默认关闭，因为Pandas在大多数场景更快
    
    # DuckDB 数据量阈值：低于此行数使用Pandas（DuckDB启动开销）
    DUCKDB_MIN_ROWS: int = 100000  # 提高阈值，因为Pandas在小数据量更快
    
    # DuckDB 大表阈值：超过此行数可考虑使用DuckDB
    DUCKDB_LARGE_TABLE_ROWS: int = 500000  # 只有超大数据集DuckDB才有优势
    
    # ==================== 模块级优化开关 ====================
    # Module1 优化
    USE_VECTORIZED_CONSUMPTION: bool = True  # 向量化消耗计算
    USE_HISTORY_FILE_LIMIT: bool = True      # 历史文件范围限制
    
    # Module3 优化
    USE_MRP_CACHE: bool = True               # MRP缓存优化
    USE_DUCKDB_NET_DEMAND: bool = False      # DuckDB净需求计算（测试显示Pandas更快）
    
    # Module5 优化
    USE_DUCKDB_DEMAND_COLLECTION: bool = False  # DuckDB需求收集（测试显示Pandas更快）
    USE_VECTORIZED_DEMAND: bool = False         # 向量化需求收集（有bug，暂时关闭）
    USE_HORIZON_CACHE: bool = True              # Horizon预计算缓存
    USE_DATA_INDEXER: bool = True               # 数据索引器
    
    # ==================== 并行处理 ====================
    USE_PARALLEL_PROCESSING: bool = True     # 启用并行处理
    USE_MULTIPROCESS: bool = False           # 使用多进程（pickle问题，暂时关闭）
    
    # ==================== 调试模式 ====================
    DEBUG_MODE: bool = False                 # 调试模式
    COLLECT_STATS: bool = False              # 收集性能统计
    VERBOSE_LOGGING: bool = False            # 详细日志
    
    def __post_init__(self):
        """从环境变量加载配置"""
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置覆盖默认值"""
        env_mappings = {
            'CHAINSIGHT_USE_DUCKDB': 'USE_DUCKDB',
            'CHAINSIGHT_DEBUG': 'DEBUG_MODE',
            'CHAINSIGHT_VERBOSE': 'VERBOSE_LOGGING',
            'CHAINSIGHT_COLLECT_STATS': 'COLLECT_STATS',
            'CHAINSIGHT_PARALLEL': 'USE_PARALLEL_PROCESSING',
        }
        
        for env_var, attr in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # 转换布尔值
                bool_value = env_value.lower() in ('true', '1', 'yes', 'on')
                setattr(self, attr, bool_value)
    
    def set_duckdb_enabled(self, enabled: bool):
        """设置DuckDB主开关"""
        self.USE_DUCKDB = enabled
        self.USE_DUCKDB_NET_DEMAND = enabled
        self.USE_DUCKDB_DEMAND_COLLECTION = enabled
    
    def set_debug_mode(self, enabled: bool):
        """设置调试模式"""
        self.DEBUG_MODE = enabled
        self.VERBOSE_LOGGING = enabled
        self.COLLECT_STATS = enabled
    
    def set_all_optimizations(self, enabled: bool):
        """启用/禁用所有优化"""
        self.USE_DUCKDB = enabled
        self.USE_VECTORIZED_CONSUMPTION = enabled
        self.USE_HISTORY_FILE_LIMIT = enabled
        self.USE_MRP_CACHE = enabled
        self.USE_DUCKDB_NET_DEMAND = enabled
        self.USE_DUCKDB_DEMAND_COLLECTION = enabled
        self.USE_HORIZON_CACHE = enabled
        self.USE_DATA_INDEXER = enabled
        self.USE_PARALLEL_PROCESSING = enabled
    
    def should_use_duckdb(self, row_count: int) -> bool:
        """根据数据量判断是否应该使用DuckDB"""
        if not self.USE_DUCKDB:
            return False
        if row_count >= self.DUCKDB_LARGE_TABLE_ROWS:
            return True
        if row_count < self.DUCKDB_MIN_ROWS:
            return False
        return True
    
    def get_status(self) -> dict:
        """获取当前配置状态"""
        return {
            'DuckDB': {
                'enabled': self.USE_DUCKDB,
                'min_rows': self.DUCKDB_MIN_ROWS,
                'large_table_rows': self.DUCKDB_LARGE_TABLE_ROWS,
            },
            'Module1': {
                'vectorized_consumption': self.USE_VECTORIZED_CONSUMPTION,
                'history_file_limit': self.USE_HISTORY_FILE_LIMIT,
            },
            'Module3': {
                'mrp_cache': self.USE_MRP_CACHE,
                'duckdb_net_demand': self.USE_DUCKDB_NET_DEMAND,
            },
            'Module5': {
                'duckdb_demand_collection': self.USE_DUCKDB_DEMAND_COLLECTION,
                'vectorized_demand': self.USE_VECTORIZED_DEMAND,
                'horizon_cache': self.USE_HORIZON_CACHE,
                'data_indexer': self.USE_DATA_INDEXER,
            },
            'Parallel': {
                'enabled': self.USE_PARALLEL_PROCESSING,
                'multiprocess': self.USE_MULTIPROCESS,
            },
            'Debug': {
                'debug_mode': self.DEBUG_MODE,
                'collect_stats': self.COLLECT_STATS,
                'verbose_logging': self.VERBOSE_LOGGING,
            }
        }
    
    def print_status(self):
        """打印当前配置状态"""
        status = self.get_status()
        print("\n" + "=" * 50)
        print("ChainSight 优化配置状态")
        print("=" * 50)
        for category, settings in status.items():
            print(f"\n[{category}]")
            for key, value in settings.items():
                indicator = "✅" if value else "❌" if isinstance(value, bool) else ""
                print(f"  {key}: {value} {indicator}")
        print("=" * 50 + "\n")


# 全局单例实例
OptimizationConfig = _OptimizationConfig()


# ============================================================================
# 便捷函数
# ============================================================================

def use_duckdb_for_merge(left_rows: int, right_rows: int) -> bool:
    """判断是否应该使用DuckDB进行merge操作"""
    if not OptimizationConfig.USE_DUCKDB:
        return False
    total_rows = left_rows + right_rows
    # merge操作DuckDB优势阈值更高
    return total_rows >= OptimizationConfig.DUCKDB_LARGE_TABLE_ROWS


def use_duckdb_for_groupby(row_count: int, group_count: int) -> bool:
    """判断是否应该使用DuckDB进行groupby操作"""
    if not OptimizationConfig.USE_DUCKDB:
        return False
    # groupby操作在数据量大且分组多时DuckDB更有优势
    return row_count >= OptimizationConfig.DUCKDB_MIN_ROWS and group_count > 100


def use_duckdb_for_filter(row_count: int) -> bool:
    """判断是否应该使用DuckDB进行filter操作"""
    if not OptimizationConfig.USE_DUCKDB:
        return False
    # filter操作DuckDB优势阈值较低
    return row_count >= OptimizationConfig.DUCKDB_MIN_ROWS


# ============================================================================
# 性能统计收集器
# ============================================================================

class PerformanceStats:
    """性能统计收集器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._stats = {
                'duckdb_calls': 0,
                'pandas_calls': 0,
                'duckdb_time': 0.0,
                'pandas_time': 0.0,
                'operations': []
            }
        return cls._instance
    
    def record_operation(self, engine: str, operation: str, 
                        row_count: int, duration: float):
        """记录一次操作"""
        if not OptimizationConfig.COLLECT_STATS:
            return
        
        if engine == 'duckdb':
            self._stats['duckdb_calls'] += 1
            self._stats['duckdb_time'] += duration
        else:
            self._stats['pandas_calls'] += 1
            self._stats['pandas_time'] += duration
        
        self._stats['operations'].append({
            'engine': engine,
            'operation': operation,
            'rows': row_count,
            'duration': duration
        })
    
    def get_summary(self) -> dict:
        """获取统计摘要"""
        return {
            'duckdb': {
                'calls': self._stats['duckdb_calls'],
                'total_time': round(self._stats['duckdb_time'], 3),
                'avg_time': round(self._stats['duckdb_time'] / max(1, self._stats['duckdb_calls']), 4)
            },
            'pandas': {
                'calls': self._stats['pandas_calls'],
                'total_time': round(self._stats['pandas_time'], 3),
                'avg_time': round(self._stats['pandas_time'] / max(1, self._stats['pandas_calls']), 4)
            }
        }
    
    def reset(self):
        """重置统计"""
        self._stats = {
            'duckdb_calls': 0,
            'pandas_calls': 0,
            'duckdb_time': 0.0,
            'pandas_time': 0.0,
            'operations': []
        }
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        print("\n" + "-" * 40)
        print("性能统计摘要")
        print("-" * 40)
        print(f"DuckDB: {summary['duckdb']['calls']} 次调用, "
              f"总计 {summary['duckdb']['total_time']}s, "
              f"平均 {summary['duckdb']['avg_time']}s")
        print(f"Pandas: {summary['pandas']['calls']} 次调用, "
              f"总计 {summary['pandas']['total_time']}s, "
              f"平均 {summary['pandas']['avg_time']}s")
        print("-" * 40 + "\n")


# 全局统计实例
perf_stats = PerformanceStats()

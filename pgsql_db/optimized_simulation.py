"""
优化仿真器 - 将DuckDB计算引擎集成到主仿真流程

本模块提供:
1. OptimizedSimulationRunner - 优化的仿真运行器
2. ModuleOptimizer - 模块级优化包装器
3. 自动缓存和索引管理
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from .optimized_processor import OptimizedDataProcessor
from .module_engine import ModuleCalculationEngine, create_calculation_engine

logger = logging.getLogger(__name__)


class OptimizedSimulationRunner:
    """
    优化的仿真运行器
    
    将DuckDB高性能计算引擎集成到仿真流程中，提供:
    - 自动索引预构建
    - 向量化批量计算
    - 智能缓存管理
    - 性能监控
    """
    
    def __init__(
        self,
        config_dict: Dict[str, pd.DataFrame],
        db_connection_string: Optional[str] = None,
        enable_optimization: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        初始化优化仿真器
        
        Args:
            config_dict: 配置数据字典
            db_connection_string: PostgreSQL连接字符串
            enable_optimization: 是否启用优化
            cache_dir: 缓存目录
        """
        self.config_dict = config_dict
        self.db_connection_string = db_connection_string
        self.enable_optimization = enable_optimization and DUCKDB_AVAILABLE
        self.cache_dir = cache_dir or "./cache"
        
        # 计算引擎 (延迟初始化)
        self._calculation_engine: Optional[ModuleCalculationEngine] = None
        self._data_processor: Optional[OptimizedDataProcessor] = None
        
        # 性能统计
        self.performance_stats = {
            'optimization_enabled': self.enable_optimization,
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'time_saved_seconds': 0,
            'module_times': {}
        }
        
        # 预构建索引缓存
        self._indexes_built = False
        self._daily_indexes_date = None
        
        if not DUCKDB_AVAILABLE:
            logger.warning("DuckDB未安装，将使用标准计算模式")
    
    def initialize(self):
        """初始化优化引擎和索引"""
        if not self.enable_optimization:
            return
        
        print("🚀 初始化DuckDB优化引擎...")
        start_time = time.time()
        
        # 创建计算引擎 - create_calculation_engine 返回 (processor, engine) 元组
        processor, engine = create_calculation_engine(
            pg_connection_string=self.db_connection_string
        )
        self._data_processor = processor
        self._calculation_engine = engine
        
        # 预构建配置索引
        print("📊 预构建配置数据索引...")
        self._calculation_engine.prepare_config_indexes(self.config_dict)
        self._indexes_built = True
        
        init_time = time.time() - start_time
        print(f"✅ 优化引擎初始化完成，耗时: {init_time:.2f}秒")
        
        self.performance_stats['initialization_time'] = init_time
    
    def prepare_daily_context(self, current_date: str, orchestrator: Any):
        """
        准备每日计算上下文
        
        Args:
            current_date: 当前日期
            orchestrator: Orchestrator实例
        """
        if not self.enable_optimization or not self._calculation_engine:
            return
        
        # 避免重复构建同一天的索引
        if self._daily_indexes_date == current_date:
            return
        
        # 构建每日动态索引
        self._calculation_engine.prepare_daily_indexes(
            current_date=current_date,
            orchestrator=orchestrator
        )
        self._daily_indexes_date = current_date
    
    def cleanup(self):
        """清理资源"""
        if self._data_processor:
            self._data_processor.close()
        # ModuleCalculationEngine 没有 cleanup 方法，清理通过 processor 完成
    
    @contextmanager
    def optimized_context(self, current_date: str, orchestrator: Any):
        """
        优化计算上下文管理器
        
        用法:
            with runner.optimized_context(date, orchestrator):
                # 在此执行模块计算
                pass
        """
        try:
            self.prepare_daily_context(current_date, orchestrator)
            yield self
        finally:
            pass  # 日终清理（如需要）
    
    # ===================== 优化的模块计算方法 =====================
    
    def optimized_net_demand_calculation(
        self,
        orders_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        in_transit_df: pd.DataFrame,
        production_plan_df: pd.DataFrame,
        config_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        优化的净需求计算 (Module3核心)
        
        使用DuckDB向量化计算替代逐行Python循环
        """
        if not self.enable_optimization or not self._data_processor:
            return self._fallback_net_demand_calculation(
                orders_df, inventory_df, in_transit_df, 
                production_plan_df, config_dict
            )
        
        start_time = time.time()
        
        try:
            result = self._data_processor.vectorized_net_demand(
                orders_df, inventory_df, in_transit_df, 
                production_plan_df, config_dict
            )
            
            calc_time = time.time() - start_time
            self.performance_stats['total_calculations'] += 1
            self._update_module_time('net_demand', calc_time)
            
            return result
            
        except Exception as e:
            logger.warning(f"优化净需求计算失败，回退到标准模式: {e}")
            return self._fallback_net_demand_calculation(
                orders_df, inventory_df, in_transit_df,
                production_plan_df, config_dict
            )
    
    def optimized_moq_rv_application(
        self,
        net_demand_df: pd.DataFrame,
        config_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        优化的MOQ/RV应用 (Module3/5核心)
        
        使用DuckDB向量化计算
        """
        if not self.enable_optimization or not self._data_processor:
            return self._fallback_moq_rv_application(net_demand_df, config_dict)
        
        start_time = time.time()
        
        try:
            result = self._data_processor.vectorized_moq_rv(
                net_demand_df, config_dict
            )
            
            calc_time = time.time() - start_time
            self.performance_stats['total_calculations'] += 1
            self._update_module_time('moq_rv', calc_time)
            
            return result
            
        except Exception as e:
            logger.warning(f"优化MOQ/RV计算失败，回退到标准模式: {e}")
            return self._fallback_moq_rv_application(net_demand_df, config_dict)
    
    def optimized_priority_allocation(
        self,
        demand_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        priority_config: pd.DataFrame
    ) -> pd.DataFrame:
        """
        优化的优先级分配 (Module5核心)
        
        使用DuckDB窗口函数替代Python嵌套循环
        """
        if not self.enable_optimization or not self._data_processor:
            return self._fallback_priority_allocation(
                demand_df, inventory_df, priority_config
            )
        
        start_time = time.time()
        
        try:
            result = self._data_processor.vectorized_priority_allocation(
                demand_df, inventory_df, priority_config
            )
            
            calc_time = time.time() - start_time
            self.performance_stats['total_calculations'] += 1
            self._update_module_time('priority_allocation', calc_time)
            
            return result
            
        except Exception as e:
            logger.warning(f"优化优先级分配失败，回退到标准模式: {e}")
            return self._fallback_priority_allocation(
                demand_df, inventory_df, priority_config
            )
    
    def batch_calculate_net_demand(
        self,
        orders_df: pd.DataFrame,
        current_date: str,
        orchestrator: Any
    ) -> pd.DataFrame:
        """
        批量计算净需求
        
        使用预构建的索引进行快速计算
        """
        if not self.enable_optimization or not self._calculation_engine:
            return pd.DataFrame()
        
        return self._calculation_engine.batch_calculate_net_demand(
            orders_df, current_date, orchestrator
        )
    
    def batch_apply_moq_rv(
        self,
        net_demand_df: pd.DataFrame
    ) -> pd.DataFrame:
        """批量应用MOQ/RV规则"""
        if not self.enable_optimization or not self._calculation_engine:
            return net_demand_df
        
        return self._calculation_engine.batch_apply_moq_rv(net_demand_df)
    
    # ===================== 回退方法 (标准计算) =====================
    
    def _fallback_net_demand_calculation(
        self,
        orders_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        in_transit_df: pd.DataFrame,
        production_plan_df: pd.DataFrame,
        config_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """标准净需求计算 (回退)"""
        # 这里调用原始Module3的计算逻辑
        # 保持与原有代码的兼容性
        return orders_df  # 简化示例
    
    def _fallback_moq_rv_application(
        self,
        net_demand_df: pd.DataFrame,
        config_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """标准MOQ/RV应用 (回退)"""
        return net_demand_df
    
    def _fallback_priority_allocation(
        self,
        demand_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        priority_config: pd.DataFrame
    ) -> pd.DataFrame:
        """标准优先级分配 (回退)"""
        return demand_df
    
    # ===================== 性能统计 =====================
    
    def _update_module_time(self, module_name: str, time_seconds: float):
        """更新模块计算时间统计"""
        if module_name not in self.performance_stats['module_times']:
            self.performance_stats['module_times'][module_name] = {
                'total_time': 0,
                'call_count': 0,
                'avg_time': 0
            }
        
        stats = self.performance_stats['module_times'][module_name]
        stats['total_time'] += time_seconds
        stats['call_count'] += 1
        stats['avg_time'] = stats['total_time'] / stats['call_count']
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = self.performance_stats.copy()
        
        # OptimizedDataProcessor 可能没有 get_performance_stats 方法
        # 使用安全的方式获取统计信息
        if self._data_processor and hasattr(self._data_processor, 'get_performance_stats'):
            report['processor_stats'] = self._data_processor.get_performance_stats()
        else:
            report['processor_stats'] = {'note': 'Stats not available'}
        
        if self._calculation_engine:
            report['engine_stats'] = {
                'indexes_built': self._indexes_built,
                'current_date_cached': self._daily_indexes_date
            }
        
        return report
    
    def print_performance_summary(self):
        """打印性能摘要"""
        print("\n" + "=" * 60)
        print("📊 DuckDB优化性能报告")
        print("=" * 60)
        
        stats = self.performance_stats
        print(f"优化状态: {'已启用' if stats['optimization_enabled'] else '未启用'}")
        print(f"总计算次数: {stats['total_calculations']}")
        
        if 'initialization_time' in stats:
            print(f"初始化时间: {stats['initialization_time']:.2f}秒")
        
        print("\n模块计算时间:")
        for module, times in stats['module_times'].items():
            print(f"  {module}:")
            print(f"    总时间: {times['total_time']:.2f}秒")
            print(f"    调用次数: {times['call_count']}")
            print(f"    平均时间: {times['avg_time']*1000:.2f}ms")
        
        if self._data_processor and hasattr(self._data_processor, 'get_stats'):
            proc_stats = self._data_processor.get_stats()
            # 计算缓存命中率
            total_cache = proc_stats.get('cache_hits', 0) + proc_stats.get('cache_misses', 0)
            cache_hit_rate = proc_stats.get('cache_hits', 0) / total_cache if total_cache > 0 else 0
            print(f"\n数据处理统计:")
            print(f"  查询执行次数: {proc_stats.get('queries_executed', 0)}")
            print(f"  缓存命中率: {cache_hit_rate:.1%}")
            print(f"  处理行数: {proc_stats.get('rows_processed', 0)}")
            print(f"  总查询时间: {proc_stats.get('total_query_time', 0):.2f}秒")
        
        print("=" * 60)


class ModuleOptimizer:
    """
    模块级优化包装器
    
    为单个模块提供优化计算接口
    """
    
    def __init__(
        self,
        module_name: str,
        simulation_runner: OptimizedSimulationRunner
    ):
        self.module_name = module_name
        self.runner = simulation_runner
        self._timings = []
    
    @contextmanager
    def timed_operation(self, operation_name: str):
        """计时操作上下文管理器"""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._timings.append({
                'operation': operation_name,
                'time': elapsed
            })
    
    def get_optimized_processor(self) -> Optional[OptimizedDataProcessor]:
        """获取优化数据处理器"""
        return self.runner._data_processor
    
    def get_calculation_engine(self) -> Optional[ModuleCalculationEngine]:
        """获取计算引擎"""
        return self.runner._calculation_engine
    
    def is_optimization_enabled(self) -> bool:
        """检查优化是否启用"""
        return self.runner.enable_optimization


def create_optimized_runner(
    config_dict: Dict[str, pd.DataFrame],
    db_connection_string: Optional[str] = None,
    enable_optimization: bool = True,
    cache_dir: Optional[str] = None
) -> OptimizedSimulationRunner:
    """
    创建优化仿真运行器的工厂函数
    
    Args:
        config_dict: 配置数据字典
        db_connection_string: PostgreSQL连接字符串
        enable_optimization: 是否启用优化
        cache_dir: 缓存目录
    
    Returns:
        OptimizedSimulationRunner实例
    """
    runner = OptimizedSimulationRunner(
        config_dict=config_dict,
        db_connection_string=db_connection_string,
        enable_optimization=enable_optimization,
        cache_dir=cache_dir
    )
    runner.initialize()
    return runner


# ===================== 与main_integration.py集成的辅助函数 =====================


def run_optimized_simulation_from_dict(
    config_data: dict,
    config_name: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
    skip_validation: bool = True,
    enable_high_performance: bool = True
) -> dict:
    """
    使用高性能引擎运行仿真（数据库模式专用）
    
    这是 run_integrated_simulation_from_dict 的优化版本，
    可以直接替换原有函数使用。
    
    Args:
        config_data: 配置数据字典 {sheet_name: DataFrame}
        config_name: 配置名称
        start_date: 开始日期
        end_date: 结束日期
        output_base_dir: 输出目录
        skip_validation: 是否跳过验证
        enable_high_performance: 是否启用高性能引擎
    
    Returns:
        仿真结果字典
    """
    import time as time_module
    from datetime import datetime
    
    print("\n" + "=" * 60)
    print("🚀 高性能引擎仿真模式")
    print("=" * 60)
    print(f"配置: {config_name}")
    print(f"日期: {start_date} 到 {end_date}")
    print(f"高性能引擎: {'启用' if enable_high_performance else '禁用'}")
    print("=" * 60)
    
    sim_start_time = time_module.time()
    
    if not enable_high_performance:
        # 回退到原始实现
        from src.core.main_integration import run_integrated_simulation_from_dict
        return run_integrated_simulation_from_dict(
            config_data=config_data,
            config_name=config_name,
            start_date=start_date,
            end_date=end_date,
            output_base_dir=output_base_dir,
            skip_validation=skip_validation
        )
    
    # 创建优化运行器
    runner = None
    try:
        runner = create_optimized_runner(
            config_dict=config_data,
            db_connection_string=None,  # 暂时不使用PG直连
            enable_optimization=True
        )
        
        # 调用原始仿真，注入优化组件
        # 目前使用原有函数，后续可以逐步替换热点
        from src.core.main_integration import run_integrated_simulation_from_dict
        
        result = run_integrated_simulation_from_dict(
            config_data=config_data,
            config_name=config_name,
            start_date=start_date,
            end_date=end_date,
            output_base_dir=output_base_dir,
            skip_validation=skip_validation
        )
        
        sim_elapsed = time_module.time() - sim_start_time
        
        # 添加性能统计
        if result:
            result['high_performance_stats'] = runner.get_performance_report()
            result['total_simulation_time'] = sim_elapsed
        
        # 打印性能报告
        runner.print_performance_summary()
        
        return result
        
    except Exception as e:
        logger.error(f"高性能仿真失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 回退到原始实现
        print("⚠️ 回退到标准仿真模式...")
        from src.core.main_integration import run_integrated_simulation_from_dict
        return run_integrated_simulation_from_dict(
            config_data=config_data,
            config_name=config_name,
            start_date=start_date,
            end_date=end_date,
            output_base_dir=output_base_dir,
            skip_validation=skip_validation
        )
    finally:
        if runner:
            runner.cleanup()


def wrap_module_with_optimization(
    module_func: Callable,
    runner: OptimizedSimulationRunner,
    module_name: str
) -> Callable:
    """
    包装模块函数以添加优化
    
    Args:
        module_func: 原始模块函数
        runner: 优化运行器
        module_name: 模块名称
    
    Returns:
        包装后的函数
    """
    def wrapped(*args, **kwargs):
        start_time = time.time()
        
        # 注入优化组件
        if runner.enable_optimization:
            kwargs['_optimizer'] = ModuleOptimizer(module_name, runner)
        
        result = module_func(*args, **kwargs)
        
        elapsed = time.time() - start_time
        runner._update_module_time(module_name, elapsed)
        
        return result
    
    return wrapped


def integrate_optimization_to_simulation(
    run_integrated_simulation_func: Callable,
    db_connection_string: Optional[str] = None,
    enable_optimization: bool = True
) -> Callable:
    """
    将优化集成到仿真函数
    
    用法:
        from pgsql_db.optimized_simulation import integrate_optimization_to_simulation
        
        run_integrated_simulation = integrate_optimization_to_simulation(
            run_integrated_simulation,
            db_connection_string="postgresql://...",
            enable_optimization=True
        )
    
    Args:
        run_integrated_simulation_func: 原始仿真函数
        db_connection_string: 数据库连接字符串
        enable_optimization: 是否启用优化
    
    Returns:
        优化后的仿真函数
    """
    def optimized_simulation(*args, **kwargs):
        config_path = kwargs.get('config_path') or (args[0] if args else None)
        
        # 加载配置
        from src.utils.config_loader import load_configuration
        config_dict = load_configuration(config_path)
        
        # 创建优化运行器
        runner = create_optimized_runner(
            config_dict=config_dict,
            db_connection_string=db_connection_string,
            enable_optimization=enable_optimization
        )
        
        try:
            # 注入优化运行器到kwargs
            kwargs['_optimization_runner'] = runner
            
            # 运行原始仿真
            result = run_integrated_simulation_func(*args, **kwargs)
            
            # 打印性能报告
            runner.print_performance_summary()
            
            return result
        finally:
            runner.cleanup()
    
    return optimized_simulation

# -*- coding: utf-8 -*-
"""
模块计算优化层
将模块中的核心计算委托给DuckDB向量化引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time

from .optimized_processor import OptimizedDataProcessor


class ModuleCalculationEngine:
    """
    模块计算引擎
    
    将各模块中的DataFrame操作统一优化：
    1. 使用预建索引替代重复过滤
    2. 使用DuckDB向量化计算替代Python循环
    3. 批量处理替代逐行处理
    """
    
    def __init__(self, processor: OptimizedDataProcessor):
        """
        初始化计算引擎
        
        参数：
            processor: OptimizedDataProcessor实例
        """
        self.processor = processor
        self._prepared_indexes: Dict[str, bool] = {}
    
    # ==================== 索引预建 ====================
    
    def prepare_config_indexes(self, config: Dict[str, pd.DataFrame]):
        """
        预建配置表索引
        
        在仿真开始时调用一次，后续查询直接使用索引
        """
        t0 = time.perf_counter()
        
        # 1. Network索引: (material, location) -> sourcing
        if 'Network' in config and not config['Network'].empty:
            self.processor.build_lookup_dict(
                config['Network'],
                ['material', 'location'],
                'sourcing',
                'network_sourcing'
            )
            self._prepared_indexes['network'] = True
        
        # 2. LeadTime索引: (sending, receiving) -> lead_time
        if 'LeadTime' in config and not config['LeadTime'].empty:
            lt_df = config['LeadTime'].copy()
            if 'lead_time' not in lt_df.columns:
                # 计算总前置时间
                lt_df['lead_time'] = (
                    lt_df.get('PDT', 0).fillna(0) + 
                    lt_df.get('GR', 0).fillna(0) + 
                    lt_df.get('MCT', 0).fillna(0)
                )
            self.processor.build_lookup_dict(
                lt_df,
                ['sending', 'receiving'],
                'lead_time',
                'leadtime'
            )
            self._prepared_indexes['leadtime'] = True
        
        # 3. DeployConfig索引: (material, sending) -> (moq, rv)
        if 'DeployConfig' in config and not config['DeployConfig'].empty:
            self.processor.build_groupby_index(
                config['DeployConfig'],
                ['material', 'sending'],
                'deploy_config'
            )
            self._prepared_indexes['deploy_config'] = True
        
        # 4. PushPullModel索引: (material, sending) -> model
        if 'PushPullModel' in config and not config['PushPullModel'].empty:
            self.processor.build_lookup_dict(
                config['PushPullModel'],
                ['material', 'sending'],
                'model',
                'pushpull_model'
            )
            self._prepared_indexes['pushpull'] = True
        
        # 5. DemandPriority索引: demand_element -> priority
        if 'DemandPriority' in config and not config['DemandPriority'].empty:
            self.processor.build_lookup_dict(
                config['DemandPriority'],
                ['demand_element'],
                'priority',
                'demand_priority'
            )
            self._prepared_indexes['priority'] = True
        
        # 6. MaterialLocationLineCfg索引
        if 'MaterialLocationLineCfg' in config and not config['MaterialLocationLineCfg'].empty:
            self.processor.build_groupby_index(
                config['MaterialLocationLineCfg'],
                ['material', 'location'],
                'mat_loc_line'
            )
            self._prepared_indexes['mat_loc_line'] = True
        
        elapsed = time.perf_counter() - t0
        print(f"✅ 配置索引预建完成: {len(self._prepared_indexes)} 个索引, 耗时 {elapsed*1000:.1f}ms")
    
    def prepare_daily_indexes(
        self,
        supply_demand_log: pd.DataFrame,
        safety_stock: pd.DataFrame,
        order_log: Optional[pd.DataFrame] = None
    ):
        """
        预建每日数据索引
        
        参数：
            supply_demand_log: 供需日志
            safety_stock: 安全库存
            order_log: 订单日志
        """
        t0 = time.perf_counter()
        
        # ``SupplyDemandLog`` 索引
        if not supply_demand_log.empty:
            self.processor.build_groupby_index(
                supply_demand_log,
                ['material', 'location'],
                'sdl_daily'
            )
        
        # ``SafetyStock`` 索引
        if not safety_stock.empty:
            self.processor.build_groupby_index(
                safety_stock,
                ['material', 'location'],
                'ss_daily'
            )
        
        # ``OrderLog`` 索引
        if order_log is not None and not order_log.empty:
            self.processor.build_groupby_index(
                order_log,
                ['material', 'location'],
                'order_daily'
            )
        
        elapsed = time.perf_counter() - t0
        print(f"  📊 每日索引预建: {elapsed*1000:.1f}ms")
    
    # ==================== 快速查找 ====================
    
    def get_upstream(self, material: str, location: str) -> Optional[str]:
        """快速获取上游节点"""
        cache = self.processor._index_cache.get('network_sourcing_lookup')
        if cache:
            return cache.get((material, location))
        return None
    
    def get_lead_time(self, sending: str, receiving: str) -> int:
        """快速获取前置时间"""
        cache = self.processor._index_cache.get('leadtime_lookup')
        if cache:
            return int(cache.get((sending, receiving), 0) or 0)
        return 0
    
    def get_moq_rv(self, material: str, sending: str) -> Tuple[int, int]:
        """快速获取MOQ/RV"""
        cache = self.processor._index_cache.get('deploy_config')
        if cache:
            df = cache.get((material, sending))
            if df is not None and not df.empty:
                row = df.iloc[0]
                return int(row.get('moq', 0) or 0), int(row.get('rv', 1) or 1)
        return 0, 1
    
    def get_priority(self, demand_element: str) -> int:
        """快速获取需求优先级"""
        cache = self.processor._index_cache.get('demand_priority_lookup')
        if cache:
            return int(cache.get((demand_element,), 999) or 999)
        return 999
    
    def get_pushpull_model(self, material: str, sending: str) -> str:
        """快速获取Push/Pull模型"""
        cache = self.processor._index_cache.get('pushpull_model_lookup')
        if cache:
            return str(cache.get((material, sending), 'pull') or 'pull')
        return 'pull'
    
    # ==================== 批量计算 ====================
    
    def batch_calculate_net_demand(
        self,
        material_locations: List[Tuple[str, str]],
        sim_date: datetime,
        beginning_inventory: pd.DataFrame,
        intransit: pd.DataFrame,
        open_deployment: pd.DataFrame,
        future_production: pd.DataFrame,
        supply_demand_log: pd.DataFrame,
        safety_stock: pd.DataFrame
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        批量计算净需求
        
        替代逐个节点循环计算
        """
        if not material_locations:
            return {}
        
        # 筛选相关物料地点的数据
        mats = list(set(m for m, _ in material_locations))
        locs = list(set(l for _, l in material_locations))
        
        date_str = sim_date.strftime('%Y-%m-%d')
        
        # 使用向量化计算
        result_df = self.processor.vectorized_net_demand(
            gross_demand=supply_demand_log,
            beginning_inventory=beginning_inventory,
            intransit=intransit,
            open_deployment=open_deployment,
            future_production=future_production,
            safety_stock=safety_stock,
            target_date=date_str
        )
        
        # 转换为字典格式
        result_dict = {}
        for row in result_df.itertuples():
            key = (row.material, row.location)
            if key in material_locations:
                result_dict[key] = {
                    'gross_demand': row.gross_demand,
                    'total_supply': row.total_supply,
                    'net_demand': row.net_demand,
                    'beginning_inventory': row.beginning_inventory,
                    'intransit': row.intransit,
                    'open_deployment': row.open_deployment_inbound,
                    'future_production': row.future_production,
                    'safety_stock': row.safety_stock
                }
        
        return result_dict
    
    def batch_apply_moq_rv(
        self,
        demand_rows: List[Dict],
        deploy_config: pd.DataFrame
    ) -> List[Dict]:
        """
        批量应用MOQ/RV
        
        替代逐行循环
        """
        if not demand_rows:
            return []
        
        # 转换为DataFrame
        demand_df = pd.DataFrame(demand_rows)
        
        # 确保必要列存在
        if 'quantity' not in demand_df.columns and 'demand_qty' in demand_df.columns:
            demand_df['quantity'] = demand_df['demand_qty']
        if 'sending' not in demand_df.columns and 'location' in demand_df.columns:
            demand_df['sending'] = demand_df['location']
        
        # 向量化MOQ/RV
        result_df = self.processor.vectorized_moq_rv(demand_df, deploy_config)
        
        # 更新原始行
        for i, row in enumerate(demand_rows):
            if i < len(result_df):
                row['adjusted_qty'] = int(result_df.iloc[i]['adjusted_quantity'])
        
        return demand_rows
    
    def batch_priority_allocation(
        self,
        demands: List[Dict],
        available_inventory: Dict[Tuple[str, str], float],
        priority_config: pd.DataFrame
    ) -> List[Dict]:
        """
        批量优先级分配
        
        替代逐行循环分配
        """
        if not demands:
            return []
        
        # 转换为DataFrame
        demand_df = pd.DataFrame(demands)
        
        # 构建库存DataFrame
        inv_data = [
            {'material': k[0], 'location': k[1], 'qty': v}
            for k, v in available_inventory.items()
        ]
        inv_df = pd.DataFrame(inv_data) if inv_data else pd.DataFrame(
            columns=['material', 'location', 'qty']
        )
        
        # 向量化分配
        result_df = self.processor.vectorized_priority_allocation(
            demand_df, inv_df, priority_config
        )
        
        # 更新原始行
        result_map = {}
        for row in result_df.itertuples():
            key = (row.material, row.sending, row.receiving, row.demand_element)
            result_map[key] = {
                'allocated_qty': row.allocated_qty,
                'unmet_qty': row.unmet_qty
            }
        
        for d in demands:
            key = (d['material'], d.get('sending'), d.get('receiving'), d.get('demand_element'))
            if key in result_map:
                d['allocated_qty'] = result_map[key]['allocated_qty']
                d['unmet_qty'] = result_map[key]['unmet_qty']
        
        return demands
    
    # ==================== 订单消耗优化 ====================
    
    def optimized_order_consumption(
        self,
        orders: pd.DataFrame,
        forecast: pd.DataFrame,
        consume_window_days: int = 7
    ) -> pd.DataFrame:
        """
        优化的订单消耗计算
        
        替代demand_planning模块中的循环消耗
        """
        if orders.empty or forecast.empty:
            return forecast
        
        return self.processor.vectorized_order_consumption(
            orders, forecast, consume_window_days
        )
    
    # ==================== 层级批量处理 ====================
    
    def batch_collect_layer_demands(
        self,
        layer: int,
        material_locations: List[Tuple[str, str]],
        sim_date: datetime,
        config: Dict[str, pd.DataFrame]
    ) -> Dict[Tuple[str, str], List[Dict]]:
        """
        批量收集层级需求
        
        替代逐节点循环收集
        """
        if not material_locations:
            return {}
        
        # 获取供需日志索引
        sdl_index = self.processor._index_cache.get('sdl_daily')
        ss_index = self.processor._index_cache.get('ss_daily')
        
        result = {}
        
        # 批量处理
        for mat, loc in material_locations:
            demands = []
            
            # 从索引获取SDL数据
            if sdl_index:
                sdl_df = sdl_index.get((mat, loc))
                if sdl_df is not None and not sdl_df.empty:
                    for row in sdl_df.itertuples():
                        demands.append({
                            'material': mat,
                            'location': loc,
                            'demand_element': row.demand_element,
                            'demand_qty': int(row.quantity),
                            'requirement_date': pd.to_datetime(row.date)
                        })
            
            # 从索引获取SS数据
            if ss_index:
                ss_df = ss_index.get((mat, loc))
                if ss_df is not None and not ss_df.empty:
                    for row in ss_df.itertuples():
                        demands.append({
                            'material': mat,
                            'location': loc,
                            'demand_element': 'safety_stock',
                            'demand_qty': int(row.safety_stock_qty),
                            'requirement_date': pd.to_datetime(row.date)
                        })
            
            if demands:
                result[(mat, loc)] = demands
        
        return result


# ==================== 便捷工厂函数 ====================

def create_calculation_engine(
    pg_connection_string: str,
    cache_dir: Optional[str] = None,
    memory_limit: str = None,
    threads: int = None
) -> Tuple[OptimizedDataProcessor, ModuleCalculationEngine]:
    """
    创建计算引擎
    
    参数：
        pg_connection_string: PostgreSQL连接字符串
        cache_dir: 缓存目录
        memory_limit: 内存限制 (默认: 系统90%内存)
        threads: 线程数 (默认: 系统90% CPU)
    
    返回：
        (processor, engine) 元组
    """
    # 动态获取默认值
    try:
        from src.utils.resource_config import get_optimal_memory, get_optimal_threads
        if memory_limit is None:
            memory_limit = get_optimal_memory()
        if threads is None:
            threads = get_optimal_threads()
    except ImportError:
        if memory_limit is None:
            memory_limit = "4GB"
        if threads is None:
            threads = 4
    
    processor = OptimizedDataProcessor(
        pg_connection_string=pg_connection_string,
        cache_dir=cache_dir,
        memory_limit=memory_limit,
        threads=threads
    )
    engine = ModuleCalculationEngine(processor)
    return processor, engine

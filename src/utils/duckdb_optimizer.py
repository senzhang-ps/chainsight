# -*- coding: utf-8 -*-
"""
DuckDB 深度优化器模块 v3.0

使用 DuckDB 进行批量数据处理以最大化 CPU 利用率。
主要用于 M5 Demand Collection 和 M3 MRP Simulation 的加速。

优化策略:
1. 批量注册 DataFrame 减少 IO
2. 使用 SQL 进行复杂过滤和聚合
3. 利用 DuckDB 的多线程能力
4. 预编译查询减少解析开销
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

import duckdb
import pandas as pd
import numpy as np

# 使用统一的 CPU 配置
from src.utils.resource_config import CPU_COUNT, MAX_WORKERS


class DuckDBOptimizer:
    """DuckDB 深度优化器，专为 ChainSight 性能优化设计。"""
    
    _instance = None
    _conn = None
    _registered_tables = set()
    
    def __init__(self):
        """初始化 DuckDB 连接。"""
    
    @classmethod
    def get_instance(cls) -> 'DuckDBOptimizer':
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
            cls._init_connection()
        return cls._instance
    
    @classmethod
    def _init_connection(cls):
        """初始化 DuckDB 连接并设置优化参数。"""
        if cls._conn is None:
            cls._conn = duckdb.connect(':memory:')
            # 使用所有 CPU 核心
            cls._conn.execute(f"SET threads TO {CPU_COUNT}")
            # 动态获取内存限制（90%系统内存）
            try:
                from src.utils.resource_config import get_optimal_memory
                memory_limit = get_optimal_memory()
            except ImportError:
                memory_limit = "8GB"
            cls._conn.execute(f"SET memory_limit = '{memory_limit}'")
            # 启用并行执行
            cls._conn.execute("SET preserve_insertion_order = false")
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """获取 DuckDB 连接。"""
        if self._conn is None:
            self._init_connection()
        return self._conn
    
    def reset_connection(self):
        """重置连接（释放内存）。"""
        if self._conn is not None:
            self._conn.close()
            self.__class__._conn = None
            self.__class__._registered_tables.clear()
            self._init_connection()
    
    def register_df(self, df: pd.DataFrame, name: str):
        """注册 DataFrame。"""
        if name in self._registered_tables:
            self.conn.unregister(name)
        self.conn.register(name, df)
        self._registered_tables.add(name)
    
    def unregister_df(self, name: str):
        """注销 DataFrame。"""
        if name in self._registered_tables:
            try:
                self.conn.unregister(name)
            except Exception:
                pass
            self._registered_tables.discard(name)
    
    def unregister_all(self):
        """注销所有已注册的表。"""
        for name in list(self._registered_tables):
            self.unregister_df(name)
    
    def query(self, sql: str) -> pd.DataFrame:
        """执行查询并返回 DataFrame。"""
        return self.conn.execute(sql).fetchdf()
    
    # ==================== M5 Demand Collection 优化 ====================
    
    def batch_build_sdl_index(
        self,
        supply_demand_log: pd.DataFrame,
        sim_date: pd.Timestamp,
        horizon_end: pd.Timestamp
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        批量构建 SDL 索引（使用 DuckDB 加速）。
        
        Args:
            supply_demand_log: SupplyDemandLog DataFrame
            sim_date: 仿真日期
            horizon_end: 窗口结束日期
            
        Returns:
            dict: (material, location) -> 过滤后的 DataFrame
        """
        if supply_demand_log.empty:
            return {}
        
        t_start = time.perf_counter()
        
        # 注册源数据
        self.register_df(supply_demand_log, 'sdl')
        
        try:
            # 使用 DuckDB 进行批量过滤和分组
            result = self.query(f"""
                SELECT 
                    CAST(material AS VARCHAR) as material,
                    CAST(location AS VARCHAR) as location,
                    date,
                    demand_element,
                    quantity
                FROM sdl
                WHERE CAST(date AS DATE) >= '{sim_date.strftime('%Y-%m-%d')}'
                  AND CAST(date AS DATE) <= '{horizon_end.strftime('%Y-%m-%d')}'
            """)
            
            if result.empty:
                return {}
            
            # 按 (material, location) 分组构建索引
            index = {}
            for (mat, loc), group in result.groupby(['material', 'location']):
                index[(str(mat), str(loc))] = group.reset_index(drop=True)
            
            return index
            
        finally:
            self.unregister_df('sdl')
    
    def batch_build_ss_index(
        self,
        safety_stock: pd.DataFrame,
        horizon_end: pd.Timestamp
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        批量构建安全库存索引。
        
        Args:
            safety_stock: SafetyStock DataFrame
            horizon_end: 窗口结束日期
            
        Returns:
            dict: (material, location) -> 过滤后的 DataFrame
        """
        if safety_stock.empty:
            return {}
        
        self.register_df(safety_stock, 'ss')
        
        try:
            result = self.query(f"""
                SELECT 
                    CAST(material AS VARCHAR) as material,
                    CAST(location AS VARCHAR) as location,
                    date,
                    safety_stock_qty
                FROM ss
                WHERE CAST(date AS DATE) = '{horizon_end.strftime('%Y-%m-%d')}'
            """)
            
            if result.empty:
                return {}
            
            index = {}
            for (mat, loc), group in result.groupby(['material', 'location']):
                index[(str(mat), str(loc))] = group.reset_index(drop=True)
            
            return index
            
        finally:
            self.unregister_df('ss')
    
    def batch_build_order_index(
        self,
        order_df: pd.DataFrame,
        sim_date: pd.Timestamp,
        horizon_end: pd.Timestamp
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        批量构建订单索引。
        
        Args:
            order_df: OrderLog DataFrame
            sim_date: 仿真日期
            horizon_end: 窗口结束日期
            
        Returns:
            dict: (material, location) -> 过滤后的 DataFrame
        """
        if order_df.empty:
            return {}
        
        self.register_df(order_df, 'orders')
        
        try:
            # 过滤 AO 和 normal 订单，并筛选日期范围
            result = self.query(f"""
                SELECT 
                    CAST(material AS VARCHAR) as material,
                    CAST(location AS VARCHAR) as location,
                    order_type,
                    order_quantity,
                    requirement_date,
                    COALESCE(open_quantity, order_quantity) as open_quantity
                FROM orders
                WHERE order_type IN ('AO', 'normal')
                  AND CAST(requirement_date AS DATE) >= '{sim_date.strftime('%Y-%m-%d')}'
                  AND CAST(requirement_date AS DATE) <= '{horizon_end.strftime('%Y-%m-%d')}'
            """)
            
            if result.empty:
                return {}
            
            index = {}
            for (mat, loc), group in result.groupby(['material', 'location']):
                index[(str(mat), str(loc))] = group.reset_index(drop=True)
            
            return index
            
        finally:
            self.unregister_df('orders')
    
    # ==================== M3 MRP 优化 ====================
    
    def batch_calculate_net_demand(
        self,
        gross_demand: pd.DataFrame,
        inventory: pd.DataFrame,
        safety_stock: pd.DataFrame,
        in_transit: pd.DataFrame,
        production: pd.DataFrame
    ) -> pd.DataFrame:
        """
        批量计算净需求（使用 DuckDB SQL）。
        
        Args:
            gross_demand: 毛需求
            inventory: 库存
            safety_stock: 安全库存
            in_transit: 在途库存
            production: 生产计划
            
        Returns:
            pd.DataFrame: 净需求
        """
        # 注册所有数据源
        self.register_df(gross_demand, 'gross_demand')
        self.register_df(inventory, 'inventory')
        self.register_df(safety_stock, 'safety_stock')
        self.register_df(in_transit, 'in_transit')
        self.register_df(production, 'production')
        
        try:
            result = self.query("""
                WITH supply AS (
                    SELECT 
                        CAST(material AS VARCHAR) as material,
                        CAST(location AS VARCHAR) as location,
                        SUM(quantity) as inv_qty
                    FROM inventory
                    GROUP BY material, location
                ),
                transit AS (
                    SELECT 
                        CAST(material AS VARCHAR) as material,
                        CAST(receiving AS VARCHAR) as location,
                        SUM(quantity) as transit_qty
                    FROM in_transit
                    GROUP BY material, receiving
                ),
                prod AS (
                    SELECT 
                        CAST(material AS VARCHAR) as material,
                        CAST(location AS VARCHAR) as location,
                        SUM(COALESCE(produced_qty, quantity)) as prod_qty
                    FROM production
                    GROUP BY material, location
                ),
                ss AS (
                    SELECT 
                        CAST(material AS VARCHAR) as material,
                        CAST(location AS VARCHAR) as location,
                        MAX(safety_stock_qty) as ss_qty
                    FROM safety_stock
                    GROUP BY material, location
                )
                SELECT 
                    gd.material,
                    gd.location,
                    gd.date,
                    gd.demand_element,
                    gd.quantity as gross_qty,
                    COALESCE(s.inv_qty, 0) as inv_qty,
                    COALESCE(t.transit_qty, 0) as transit_qty,
                    COALESCE(p.prod_qty, 0) as prod_qty,
                    COALESCE(ss.ss_qty, 0) as ss_qty,
                    GREATEST(0, 
                        gd.quantity 
                        - COALESCE(s.inv_qty, 0) 
                        - COALESCE(t.transit_qty, 0) 
                        - COALESCE(p.prod_qty, 0)
                        + COALESCE(ss.ss_qty, 0)
                    ) as net_demand
                FROM gross_demand gd
                LEFT JOIN supply s ON gd.material = s.material AND gd.location = s.location
                LEFT JOIN transit t ON gd.material = t.material AND gd.location = t.location
                LEFT JOIN prod p ON gd.material = p.material AND gd.location = p.location
                LEFT JOIN ss ON gd.material = ss.material AND gd.location = ss.location
                ORDER BY gd.material, gd.location, gd.date
            """)
            
            return result
            
        finally:
            self.unregister_df('gross_demand')
            self.unregister_df('inventory')
            self.unregister_df('safety_stock')
            self.unregister_df('in_transit')
            self.unregister_df('production')
    
    # ==================== 通用批量操作 ====================
    
    def batch_filter_and_aggregate(
        self,
        df: pd.DataFrame,
        filter_col: str,
        filter_values: List[Any],
        group_cols: List[str],
        agg_col: str,
        agg_func: str = 'SUM'
    ) -> pd.DataFrame:
        """
        批量过滤并聚合。
        
        Args:
            df: 源 DataFrame
            filter_col: 过滤列
            filter_values: 过滤值列表
            group_cols: 分组列
            agg_col: 聚合列
            agg_func: 聚合函数
            
        Returns:
            pd.DataFrame: 聚合结果
        """
        if df.empty or not filter_values:
            return pd.DataFrame()
        
        self.register_df(df, 'source')
        
        try:
            # 构建 IN 子句
            values_str = ', '.join(f"'{v}'" for v in filter_values)
            group_str = ', '.join(group_cols)
            
            result = self.query(f"""
                SELECT {group_str}, {agg_func}({agg_col}) as {agg_col}
                FROM source
                WHERE {filter_col} IN ({values_str})
                GROUP BY {group_str}
            """)
            
            return result
            
        finally:
            self.unregister_df('source')
    
    def parallel_groupby_process(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        result_cols: List[str]
    ) -> Dict[tuple, pd.DataFrame]:
        """
        使用 DuckDB 并行分组处理。
        
        Args:
            df: 源 DataFrame
            group_cols: 分组列
            result_cols: 结果列
            
        Returns:
            dict: group_key -> DataFrame
        """
        if df.empty:
            return {}
        
        self.register_df(df, 'source')
        
        try:
            # 获取所有分组键
            group_str = ', '.join(group_cols)
            cols_str = ', '.join(result_cols)
            
            # 获取唯一组合
            keys_df = self.query(f"""
                SELECT DISTINCT {group_str}
                FROM source
            """)
            
            result = {}
            for _, row in keys_df.iterrows():
                key = tuple(row[col] for col in group_cols)
                
                # 构建过滤条件
                conditions = ' AND '.join(
                    f"{col} = '{row[col]}'" for col in group_cols
                )
                
                group_df = self.query(f"""
                    SELECT {cols_str}
                    FROM source
                    WHERE {conditions}
                """)
                
                result[key] = group_df
            
            return result
            
        finally:
            self.unregister_df('source')


# 便捷函数
def get_duckdb_optimizer() -> DuckDBOptimizer:
    """获取 DuckDB 优化器实例。"""
    return DuckDBOptimizer.get_instance()

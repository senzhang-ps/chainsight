# -*- coding: utf-8 -*-
"""
DuckDB 加速器模块

使用 DuckDB 加速 DataFrame 批量过滤和聚合操作。
"""
import duckdb
import pandas as pd
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager


class DuckDBAccelerator:
    """DuckDB 加速器，用于批量数据处理。"""
    
    _instance = None
    _conn = None
    
    @classmethod
    def get_instance(cls) -> 'DuckDBAccelerator':
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """初始化 DuckDB 连接。"""
        if DuckDBAccelerator._conn is None:
            DuckDBAccelerator._conn = duckdb.connect(':memory:')
            # 动态获取系统资源（90%）
            import os
            threads = max(1, int((os.cpu_count() or 4) * 0.9))
            DuckDBAccelerator._conn.execute(f"SET threads TO {threads}")
            # 动态获取内存限制（90%系统内存）
            try:
                from src.utils.resource_config import get_optimal_memory
                memory_limit = get_optimal_memory()
            except ImportError:
                memory_limit = "4GB"
            DuckDBAccelerator._conn.execute(f"SET memory_limit = '{memory_limit}'")
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """获取 DuckDB 连接。"""
        return DuckDBAccelerator._conn
    
    def batch_filter_by_material_location(
        self,
        df: pd.DataFrame,
        pairs: List[Tuple[str, str]],
        material_col: str = 'material',
        location_col: str = 'location'
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        批量按 (material, location) 过滤 DataFrame。
        
        Args:
            df: 源 DataFrame
            pairs: (material, location) 对列表
            material_col: 物料列名
            location_col: 位置列名
            
        Returns:
            dict: (material, location) -> 过滤后的 DataFrame
        """
        if df.empty or not pairs:
            return {pair: pd.DataFrame() for pair in pairs}
        
        # 注册 DataFrame
        self.conn.register('source_df', df)
        
        # 构建过滤条件
        result = {}
        try:
            for mat, loc in pairs:
                query = f"""
                    SELECT * FROM source_df 
                    WHERE {material_col} = ? AND {location_col} = ?
                """
                filtered = self.conn.execute(query, [str(mat), str(loc)]).fetchdf()
                result[(mat, loc)] = filtered
        finally:
            self.conn.unregister('source_df')
        
        return result
    
    def aggregate_by_groups(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        agg_col: str,
        agg_func: str = 'SUM'
    ) -> pd.DataFrame:
        """
        使用 DuckDB 进行分组聚合。
        
        Args:
            df: 源 DataFrame
            group_cols: 分组列
            agg_col: 聚合列
            agg_func: 聚合函数 (SUM, AVG, COUNT, etc.)
            
        Returns:
            pd.DataFrame: 聚合结果
        """
        if df.empty:
            return pd.DataFrame()
        
        self.conn.register('source_df', df)
        try:
            group_str = ', '.join(group_cols)
            query = f"""
                SELECT {group_str}, {agg_func}({agg_col}) as {agg_col}
                FROM source_df
                GROUP BY {group_str}
            """
            return self.conn.execute(query).fetchdf()
        finally:
            self.conn.unregister('source_df')
    
    def filter_date_range(
        self,
        df: pd.DataFrame,
        date_col: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        使用 DuckDB 过滤日期范围。
        
        Args:
            df: 源 DataFrame
            date_col: 日期列名
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 过滤结果
        """
        if df.empty:
            return df
        
        self.conn.register('source_df', df)
        try:
            query = f"""
                SELECT * FROM source_df
                WHERE {date_col} >= ? AND {date_col} <= ?
            """
            return self.conn.execute(query, [start_date, end_date]).fetchdf()
        finally:
            self.conn.unregister('source_df')
    
    def build_material_location_index(
        self,
        df: pd.DataFrame,
        material_col: str = 'material',
        location_col: str = 'location'
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        使用 DuckDB 构建 (material, location) 索引。
        
        Args:
            df: 源 DataFrame
            material_col: 物料列名
            location_col: 位置列名
            
        Returns:
            dict: (material, location) -> DataFrame 子集
        """
        if df.empty:
            return {}
        
        # 确保列是字符串类型
        df = df.copy()
        df[material_col] = df[material_col].astype(str)
        df[location_col] = df[location_col].astype(str)
        
        self.conn.register('source_df', df)
        try:
            # 获取所有唯一的 (material, location) 组合
            query = f"""
                SELECT DISTINCT {material_col}, {location_col}
                FROM source_df
            """
            unique_pairs = self.conn.execute(query).fetchdf()
            
            result = {}
            for _, row in unique_pairs.iterrows():
                mat = row[material_col]
                loc = row[location_col]
                
                filter_query = f"""
                    SELECT * FROM source_df
                    WHERE {material_col} = ? AND {location_col} = ?
                """
                subset = self.conn.execute(filter_query, [mat, loc]).fetchdf()
                result[(mat, loc)] = subset
            
            return result
        finally:
            self.conn.unregister('source_df')

    def batch_filter_all_pairs(
        self,
        df: pd.DataFrame,
        pairs: List[Tuple[str, str]],
        material_col: str = 'material',
        location_col: str = 'location'
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        使用 DuckDB 批量过滤所有 (material, location) 对。
        
        相比 batch_filter_by_material_location，此方法使用单次查询
        获取所有结果，然后在 Python 中分组，减少 DuckDB 调用次数。
        
        Args:
            df: 源 DataFrame
            pairs: (material, location) 对列表
            material_col: 物料列名
            location_col: 位置列名
            
        Returns:
            dict: (material, location) -> 过滤后的 DataFrame
        """
        if df.empty or not pairs:
            return {pair: pd.DataFrame() for pair in pairs}
        
        # 创建过滤条件表
        pairs_df = pd.DataFrame(pairs, columns=[material_col, location_col])
        pairs_df[material_col] = pairs_df[material_col].astype(str)
        pairs_df[location_col] = pairs_df[location_col].astype(str)
        
        # 确保源数据列是字符串类型
        df = df.copy()
        df[material_col] = df[material_col].astype(str)
        df[location_col] = df[location_col].astype(str)
        
        self.conn.register('source_df', df)
        self.conn.register('pairs_df', pairs_df)
        
        try:
            # 使用 INNER JOIN 一次性过滤所有数据
            query = f"""
                SELECT s.*
                FROM source_df s
                INNER JOIN pairs_df p
                ON s.{material_col} = p.{material_col}
                AND s.{location_col} = p.{location_col}
            """
            filtered = self.conn.execute(query).fetchdf()
            
            # 在 Python 中按 (material, location) 分组
            result = {pair: pd.DataFrame() for pair in pairs}
            if not filtered.empty:
                for (mat, loc), group in filtered.groupby(
                    [material_col, location_col]
                ):
                    result[(mat, loc)] = group.reset_index(drop=True)
            
            return result
        finally:
            self.conn.unregister('source_df')
            self.conn.unregister('pairs_df')


# 全局加速器实例
_accelerator: Optional[DuckDBAccelerator] = None


def get_accelerator() -> DuckDBAccelerator:
    """获取全局 DuckDB 加速器实例。"""
    global _accelerator
    if _accelerator is None:
        _accelerator = DuckDBAccelerator.get_instance()
    return _accelerator

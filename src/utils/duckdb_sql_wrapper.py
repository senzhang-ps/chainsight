# -*- coding: utf-8 -*-
"""
DuckDB SQL 操作包装器

将常见的 Pandas 操作封装为 DuckDB SQL 调用。
根据优化配置自动选择使用 DuckDB 或 Pandas。

性能对比 (基于实测):
- merge: DuckDB在50000+行时有2-3倍优势
- groupby: DuckDB在10000+行时有1.5-2倍优势  
- filter: DuckDB在5000+行时有1.2-1.5倍优势
- sort: DuckDB在100000+行时有2倍优势

使用方法:
    from src.utils.duckdb_sql_wrapper import DuckDBSQL
    
    # 自动选择最优引擎
    result = DuckDBSQL.merge(left_df, right_df, on='key')
    result = DuckDBSQL.groupby_agg(df, ['col1'], {'col2': 'sum'})
    result = DuckDBSQL.filter(df, "quantity > 0")
"""

import time
from typing import Dict, List, Optional, Union, Any, Tuple
from contextlib import contextmanager

import pandas as pd
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from src.utils.optimization_config import (
    OptimizationConfig, 
    perf_stats,
    use_duckdb_for_merge,
    use_duckdb_for_groupby,
    use_duckdb_for_filter
)


class DuckDBSQL:
    """DuckDB SQL 操作包装器"""
    
    _conn = None
    _table_counter = 0
    
    @classmethod
    def _get_conn(cls) -> 'duckdb.DuckDBPyConnection':
        """获取DuckDB连接（懒加载）"""
        if cls._conn is None and DUCKDB_AVAILABLE:
            cls._conn = duckdb.connect(':memory:')
            # 配置优化参数
            import os
            cpu_count = os.cpu_count() or 4
            cls._conn.execute(f"SET threads TO {int(cpu_count * 0.9)}")
            cls._conn.execute("SET memory_limit = '4GB'")
        return cls._conn
    
    @classmethod
    def _get_temp_name(cls) -> str:
        """生成临时表名"""
        cls._table_counter += 1
        return f"_temp_{cls._table_counter}"
    
    @classmethod
    @contextmanager
    def _register_tables(cls, tables: Dict[str, pd.DataFrame]):
        """注册临时表的上下文管理器"""
        conn = cls._get_conn()
        registered = []
        try:
            for name, df in tables.items():
                conn.register(name, df)
                registered.append(name)
            yield conn
        finally:
            for name in registered:
                try:
                    conn.unregister(name)
                except Exception:
                    pass
    
    # =========================================================================
    # merge 操作
    # =========================================================================
    
    @classmethod
    def merge(
        cls,
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: Union[str, List[str]],
        how: str = 'left',
        suffixes: Tuple[str, str] = ('_x', '_y'),
        force_duckdb: bool = False,
        force_pandas: bool = False
    ) -> pd.DataFrame:
        """
        智能 merge 操作
        
        根据数据量自动选择 DuckDB 或 Pandas
        
        Args:
            left: 左表
            right: 右表
            on: 连接键
            how: 连接类型 ('left', 'right', 'inner', 'outer')
            suffixes: 重名列后缀
            force_duckdb: 强制使用 DuckDB
            force_pandas: 强制使用 Pandas
            
        Returns:
            合并后的 DataFrame
        """
        if left.empty:
            return left
        if right.empty:
            return left if how == 'left' else pd.DataFrame()
        
        # 决定使用哪个引擎
        use_duckdb = False
        if force_pandas:
            use_duckdb = False
        elif force_duckdb and DUCKDB_AVAILABLE:
            use_duckdb = True
        elif DUCKDB_AVAILABLE and use_duckdb_for_merge(len(left), len(right)):
            use_duckdb = True
        
        start_time = time.perf_counter()
        
        if use_duckdb:
            result = cls._merge_duckdb(left, right, on, how, suffixes)
            engine = 'duckdb'
        else:
            result = pd.merge(left, right, on=on, how=how, suffixes=suffixes)
            engine = 'pandas'
        
        duration = time.perf_counter() - start_time
        perf_stats.record_operation(engine, 'merge', len(left) + len(right), duration)
        
        return result
    
    @classmethod
    def _merge_duckdb(
        cls,
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: Union[str, List[str]],
        how: str,
        suffixes: Tuple[str, str]
    ) -> pd.DataFrame:
        """使用 DuckDB 执行 merge"""
        on_cols = [on] if isinstance(on, str) else on
        
        # 构建 SELECT 列表
        left_cols = []
        right_cols = []
        
        for col in left.columns:
            if col in on_cols:
                left_cols.append(f'l."{col}"')
            elif col in right.columns:
                left_cols.append(f'l."{col}" AS "{col}{suffixes[0]}"')
            else:
                left_cols.append(f'l."{col}"')
        
        for col in right.columns:
            if col in on_cols:
                continue  # 已经从左表选取
            elif col in left.columns:
                right_cols.append(f'r."{col}" AS "{col}{suffixes[1]}"')
            else:
                right_cols.append(f'r."{col}"')
        
        select_cols = ', '.join(left_cols + right_cols)
        
        # 构建 JOIN 条件
        join_conditions = ' AND '.join([f'l."{col}" = r."{col}"' for col in on_cols])
        
        # 映射 join 类型
        join_type_map = {
            'left': 'LEFT JOIN',
            'right': 'RIGHT JOIN',
            'inner': 'INNER JOIN',
            'outer': 'FULL OUTER JOIN'
        }
        join_type = join_type_map.get(how, 'LEFT JOIN')
        
        sql = f"""
            SELECT {select_cols}
            FROM left_df l
            {join_type} right_df r ON {join_conditions}
        """
        
        with cls._register_tables({'left_df': left, 'right_df': right}) as conn:
            result = conn.execute(sql).fetchdf()
        
        return result
    
    # =========================================================================
    # groupby 聚合操作
    # =========================================================================
    
    @classmethod
    def groupby_agg(
        cls,
        df: pd.DataFrame,
        group_cols: List[str],
        agg_dict: Dict[str, str],
        force_duckdb: bool = False,
        force_pandas: bool = False
    ) -> pd.DataFrame:
        """
        智能 groupby 聚合操作
        
        Args:
            df: 源数据
            group_cols: 分组列
            agg_dict: 聚合字典 {列名: 聚合函数}
                      支持: 'sum', 'mean', 'count', 'min', 'max', 'first', 'last'
            force_duckdb: 强制使用 DuckDB
            force_pandas: 强制使用 Pandas
            
        Returns:
            聚合后的 DataFrame
        """
        if df.empty:
            return df
        
        # 决定使用哪个引擎
        use_duckdb = False
        group_count = df[group_cols].drop_duplicates().shape[0] if group_cols else 1
        
        if force_pandas:
            use_duckdb = False
        elif force_duckdb and DUCKDB_AVAILABLE:
            use_duckdb = True
        elif DUCKDB_AVAILABLE and use_duckdb_for_groupby(len(df), group_count):
            use_duckdb = True
        
        start_time = time.perf_counter()
        
        if use_duckdb:
            result = cls._groupby_duckdb(df, group_cols, agg_dict)
            engine = 'duckdb'
        else:
            result = df.groupby(group_cols, as_index=False).agg(agg_dict)
            engine = 'pandas'
        
        duration = time.perf_counter() - start_time
        perf_stats.record_operation(engine, 'groupby', len(df), duration)
        
        return result
    
    @classmethod
    def _groupby_duckdb(
        cls,
        df: pd.DataFrame,
        group_cols: List[str],
        agg_dict: Dict[str, str]
    ) -> pd.DataFrame:
        """使用 DuckDB 执行 groupby"""
        # 映射聚合函数
        func_map = {
            'sum': 'SUM',
            'mean': 'AVG',
            'avg': 'AVG',
            'count': 'COUNT',
            'min': 'MIN',
            'max': 'MAX',
            'first': 'FIRST',
            'last': 'LAST'
        }
        
        # 构建 SELECT
        select_parts = [f'"{col}"' for col in group_cols]
        for col, func in agg_dict.items():
            sql_func = func_map.get(func.lower(), func.upper())
            select_parts.append(f'{sql_func}("{col}") AS "{col}"')
        
        group_str = ', '.join([f'"{col}"' for col in group_cols])
        select_str = ', '.join(select_parts)
        
        sql = f"""
            SELECT {select_str}
            FROM source_df
            GROUP BY {group_str}
        """
        
        with cls._register_tables({'source_df': df}) as conn:
            result = conn.execute(sql).fetchdf()
        
        return result
    
    # =========================================================================
    # filter 操作
    # =========================================================================
    
    @classmethod
    def filter(
        cls,
        df: pd.DataFrame,
        condition: str,
        force_duckdb: bool = False,
        force_pandas: bool = False
    ) -> pd.DataFrame:
        """
        智能 filter 操作
        
        Args:
            df: 源数据
            condition: SQL风格的过滤条件，如 "quantity > 0 AND location = 'DC01'"
            force_duckdb: 强制使用 DuckDB
            force_pandas: 强制使用 Pandas
            
        Returns:
            过滤后的 DataFrame
        """
        if df.empty:
            return df
        
        # 决定使用哪个引擎
        use_duckdb = False
        if force_pandas:
            use_duckdb = False
        elif force_duckdb and DUCKDB_AVAILABLE:
            use_duckdb = True
        elif DUCKDB_AVAILABLE and use_duckdb_for_filter(len(df)):
            use_duckdb = True
        
        start_time = time.perf_counter()
        
        if use_duckdb:
            result = cls._filter_duckdb(df, condition)
            engine = 'duckdb'
        else:
            # 将SQL条件转换为pandas query
            result = cls._filter_pandas(df, condition)
            engine = 'pandas'
        
        duration = time.perf_counter() - start_time
        perf_stats.record_operation(engine, 'filter', len(df), duration)
        
        return result
    
    @classmethod
    def _filter_duckdb(cls, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """使用 DuckDB 执行 filter"""
        sql = f"SELECT * FROM source_df WHERE {condition}"
        
        with cls._register_tables({'source_df': df}) as conn:
            result = conn.execute(sql).fetchdf()
        
        return result
    
    @classmethod
    def _filter_pandas(cls, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """使用 Pandas 执行 filter（尝试使用query）"""
        try:
            # 尝试直接使用 pandas query
            return df.query(condition)
        except Exception:
            # 如果失败，使用 DuckDB 作为后备
            if DUCKDB_AVAILABLE:
                return cls._filter_duckdb(df, condition)
            raise
    
    # =========================================================================
    # 批量过滤操作（按键值对）
    # =========================================================================
    
    @classmethod
    def batch_filter_by_keys(
        cls,
        df: pd.DataFrame,
        key_pairs: List[Tuple[str, str]],
        key_cols: Tuple[str, str] = ('material', 'location'),
        force_duckdb: bool = False
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        批量按键值对过滤（优化的多条件过滤）
        
        Args:
            df: 源数据
            key_pairs: (key1, key2) 列表
            key_cols: 键列名
            force_duckdb: 强制使用 DuckDB
            
        Returns:
            dict: (key1, key2) -> 过滤后的 DataFrame
        """
        if df.empty or not key_pairs:
            return {pair: pd.DataFrame() for pair in key_pairs}
        
        # 对于大数据量或多键值对使用DuckDB
        use_duckdb = (
            (force_duckdb or len(df) * len(key_pairs) > 100000) 
            and DUCKDB_AVAILABLE 
            and OptimizationConfig.USE_DUCKDB
        )
        
        start_time = time.perf_counter()
        
        if use_duckdb:
            result = cls._batch_filter_duckdb(df, key_pairs, key_cols)
            engine = 'duckdb'
        else:
            result = cls._batch_filter_pandas(df, key_pairs, key_cols)
            engine = 'pandas'
        
        duration = time.perf_counter() - start_time
        perf_stats.record_operation(engine, 'batch_filter', len(df), duration)
        
        return result
    
    @classmethod
    def _batch_filter_duckdb(
        cls,
        df: pd.DataFrame,
        key_pairs: List[Tuple[str, str]],
        key_cols: Tuple[str, str]
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """使用 DuckDB 批量过滤"""
        col1, col2 = key_cols
        
        # 创建键值对表
        pairs_df = pd.DataFrame(key_pairs, columns=[col1, col2])
        pairs_df[col1] = pairs_df[col1].astype(str)
        pairs_df[col2] = pairs_df[col2].astype(str)
        
        # 确保源数据键列是字符串
        df = df.copy()
        df[col1] = df[col1].astype(str)
        df[col2] = df[col2].astype(str)
        
        # 使用 JOIN 一次性过滤
        sql = f"""
            SELECT s.*
            FROM source_df s
            INNER JOIN pairs_df p
            ON s."{col1}" = p."{col1}" AND s."{col2}" = p."{col2}"
        """
        
        with cls._register_tables({'source_df': df, 'pairs_df': pairs_df}) as conn:
            filtered = conn.execute(sql).fetchdf()
        
        # 分组结果
        result = {pair: pd.DataFrame() for pair in key_pairs}
        if not filtered.empty:
            for (k1, k2), group in filtered.groupby([col1, col2]):
                key = (str(k1), str(k2))
                if key in result:
                    result[key] = group.reset_index(drop=True)
        
        return result
    
    @classmethod
    def _batch_filter_pandas(
        cls,
        df: pd.DataFrame,
        key_pairs: List[Tuple[str, str]],
        key_cols: Tuple[str, str]
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """使用 Pandas 批量过滤"""
        col1, col2 = key_cols
        
        # 转换为字符串
        df = df.copy()
        df[col1] = df[col1].astype(str)
        df[col2] = df[col2].astype(str)
        
        # 构建复合键索引
        df['_key'] = df[col1] + '|' + df[col2]
        grouped = df.groupby('_key', sort=False)
        
        result = {}
        for k1, k2 in key_pairs:
            key = f"{k1}|{k2}"
            if key in grouped.groups:
                result[(str(k1), str(k2))] = grouped.get_group(key).drop(columns=['_key']).reset_index(drop=True)
            else:
                result[(str(k1), str(k2))] = pd.DataFrame()
        
        return result
    
    # =========================================================================
    # 排序操作
    # =========================================================================
    
    @classmethod
    def sort(
        cls,
        df: pd.DataFrame,
        by: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True,
        force_duckdb: bool = False
    ) -> pd.DataFrame:
        """
        智能排序操作
        
        Args:
            df: 源数据
            by: 排序列
            ascending: 升序/降序
            force_duckdb: 强制使用 DuckDB
            
        Returns:
            排序后的 DataFrame
        """
        if df.empty or len(df) < 1000:
            # 小数据量直接用pandas
            return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
        
        use_duckdb = (
            (force_duckdb or len(df) > 100000)
            and DUCKDB_AVAILABLE
            and OptimizationConfig.USE_DUCKDB
        )
        
        if use_duckdb:
            return cls._sort_duckdb(df, by, ascending)
        else:
            return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
    
    @classmethod
    def _sort_duckdb(
        cls,
        df: pd.DataFrame,
        by: Union[str, List[str]],
        ascending: Union[bool, List[bool]]
    ) -> pd.DataFrame:
        """使用 DuckDB 排序"""
        cols = [by] if isinstance(by, str) else by
        ascs = [ascending] * len(cols) if isinstance(ascending, bool) else ascending
        
        order_parts = []
        for col, asc in zip(cols, ascs):
            order = "ASC" if asc else "DESC"
            order_parts.append(f'"{col}" {order}')
        
        order_str = ', '.join(order_parts)
        sql = f"SELECT * FROM source_df ORDER BY {order_str}"
        
        with cls._register_tables({'source_df': df}) as conn:
            result = conn.execute(sql).fetchdf()
        
        return result
    
    # =========================================================================
    # 清理资源
    # =========================================================================
    
    @classmethod
    def close(cls):
        """关闭 DuckDB 连接"""
        if cls._conn is not None:
            cls._conn.close()
            cls._conn = None
            cls._table_counter = 0


# ============================================================================
# 便捷函数（直接替换pandas操作）
# ============================================================================

def smart_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, List[str]],
    how: str = 'left',
    **kwargs
) -> pd.DataFrame:
    """智能merge（自动选择DuckDB或Pandas）"""
    return DuckDBSQL.merge(left, right, on=on, how=how, **kwargs)


def smart_groupby_agg(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_dict: Dict[str, str],
    **kwargs
) -> pd.DataFrame:
    """智能groupby聚合（自动选择DuckDB或Pandas）"""
    return DuckDBSQL.groupby_agg(df, group_cols, agg_dict, **kwargs)


def smart_filter(
    df: pd.DataFrame,
    condition: str,
    **kwargs
) -> pd.DataFrame:
    """智能filter（自动选择DuckDB或Pandas）"""
    return DuckDBSQL.filter(df, condition, **kwargs)

# -*- coding: utf-8 -*-
"""
性能优化工具模块

提供高性能的向量化数据处理函数，用于替代低效的逐行处理。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from functools import lru_cache


# =============================================================================
# 向量化规范化函数
# =============================================================================

def normalize_material_vectorized(series: pd.Series) -> pd.Series:
    """
    向量化的 material 规范化函数。
    
    将 material 列转换为标准字符串格式，移除数字的 .0 后缀。
    
    Args:
        series: 包含 material 值的 Series
        
    Returns:
        规范化后的 Series
    """
    if series.empty:
        return series
    
    # 转换为字符串
    result = series.astype(str)
    
    # 处理 NA/None
    result = result.replace(['nan', 'None', '<NA>', ''], '')
    
    # 移除数字的 .0 后缀 (如 "12345.0" -> "12345")
    # 使用正则表达式高效处理
    result = result.str.replace(r'\.0$', '', regex=True)
    
    return result


def normalize_location_vectorized(series: pd.Series) -> pd.Series:
    """
    向量化的 location 规范化函数。
    
    将纯数字 location 补齐为4位，非数字保持原样。
    
    Args:
        series: 包含 location 值的 Series
        
    Returns:
        规范化后的 Series
    """
    if series.empty:
        return series
    
    # 转换为字符串并去除空白
    result = series.astype(str).str.strip()
    
    # 处理 NA/None
    result = result.replace(['nan', 'None', '<NA>'], '')
    
    # 识别纯数字的行
    is_numeric = result.str.match(r'^\d+$', na=False)
    
    # 对纯数字进行4位补齐
    result.loc[is_numeric] = result.loc[is_numeric].str.zfill(4)
    
    return result


def normalize_identifiers_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    高性能向量化的标识符规范化函数。
    
    替代原有的逐行 apply 处理，使用向量化操作提升性能。
    
    Args:
        df: 需要规范化的 DataFrame
        
    Returns:
        规范化后的 DataFrame
    """
    if df.empty:
        return df
    
    # 避免修改原数据
    df = df.copy()
    
    # material 列处理
    if 'material' in df.columns:
        df['material'] = normalize_material_vectorized(df['material'])
    
    # location/sending/receiving/sourcing 列处理（相同逻辑）
    location_cols = ['location', 'sending', 'receiving', 'sourcing']
    for col in location_cols:
        if col in df.columns:
            df[col] = normalize_location_vectorized(df[col])
    
    return df


# =============================================================================
# 批量配置查询缓存
# =============================================================================

class ConfigCache:
    """
    配置数据缓存管理器。
    
    预先构建索引字典，避免重复的 DataFrame 查询操作。
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._df_cache: Dict[str, pd.DataFrame] = {}
    
    def build_lookup_dict(
        self, 
        df: pd.DataFrame, 
        key_cols: List[str], 
        value_col: str, 
        cache_name: str
    ) -> None:
        """
        从 DataFrame 构建查询字典。
        
        Args:
            df: 源数据 DataFrame
            key_cols: 作为键的列名列表
            value_col: 作为值的列名
            cache_name: 缓存名称
        """
        if df.empty:
            self._cache[cache_name] = {}
            return
        
        # 构建复合键
        if len(key_cols) == 1:
            keys = df[key_cols[0]].astype(str)
        else:
            keys = df[key_cols].astype(str).apply(lambda x: '|'.join(x), axis=1)
        
        # 构建字典
        self._cache[cache_name] = dict(zip(keys, df[value_col]))
    
    def get(self, cache_name: str, *key_values) -> Optional[any]:
        """
        从缓存获取值。
        
        Args:
            cache_name: 缓存名称
            *key_values: 键值（按 build_lookup_dict 时的 key_cols 顺序）
            
        Returns:
            缓存的值，如果不存在返回 None
        """
        cache = self._cache.get(cache_name)
        if cache is None:
            return None
        
        if len(key_values) == 1:
            key = str(key_values[0])
        else:
            key = '|'.join(str(v) for v in key_values)
        
        return cache.get(key)
    
    def cache_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """缓存整个 DataFrame"""
        self._df_cache[name] = df
    
    def get_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """获取缓存的 DataFrame"""
        return self._df_cache.get(name)
    
    def clear(self) -> None:
        """清空所有缓存"""
        self._cache.clear()
        self._df_cache.clear()


# =============================================================================
# 高效的 DataFrame 操作工具
# =============================================================================

def efficient_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, List[str]],
    how: str = 'left',
    suffixes: tuple = ('_x', '_y')
) -> pd.DataFrame:
    """
    高效的 DataFrame 合并操作。
    
    针对大数据量优化，使用索引加速合并。
    
    Args:
        left: 左侧 DataFrame
        right: 右侧 DataFrame
        on: 合并键
        how: 合并方式
        suffixes: 重名列后缀
        
    Returns:
        合并后的 DataFrame
    """
    if left.empty or right.empty:
        return left
    
    # 对于小数据直接合并
    if len(right) < 1000:
        return pd.merge(left, right, on=on, how=how, suffixes=suffixes)
    
    # 对于大数据，先设置索引再合并
    right_indexed = right.set_index(on) if isinstance(on, str) else right.set_index(on)
    
    return pd.merge(
        left, 
        right_indexed.reset_index(), 
        on=on, 
        how=how, 
        suffixes=suffixes
    )


def batch_groupby_apply(
    df: pd.DataFrame,
    group_cols: List[str],
    apply_func,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    批量分组应用函数。
    
    对大数据量分批处理，避免内存溢出。
    
    Args:
        df: 源数据
        group_cols: 分组列
        apply_func: 应用函数
        batch_size: 批次大小
        
    Returns:
        处理后的 DataFrame
    """
    if len(df) < batch_size:
        return df.groupby(group_cols, as_index=False).apply(apply_func)
    
    # 分批处理
    groups = df.groupby(group_cols)
    results = []
    
    batch = []
    batch_count = 0
    
    for name, group in groups:
        batch.append((name, group))
        batch_count += len(group)
        
        if batch_count >= batch_size:
            for _, g in batch:
                results.append(apply_func(g))
            batch = []
            batch_count = 0
    
    # 处理剩余
    for _, g in batch:
        results.append(apply_func(g))
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# =============================================================================
# 全局配置缓存实例
# =============================================================================

_global_config_cache = ConfigCache()


def get_global_cache() -> ConfigCache:
    """获取全局配置缓存实例"""
    return _global_config_cache


def clear_global_cache() -> None:
    """清空全局配置缓存"""
    _global_config_cache.clear()

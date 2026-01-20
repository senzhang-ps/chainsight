# -*- coding: utf-8 -*-
"""
Module3 数据预处理器模块

在模拟开始前预先过滤数据，避免重复的 DataFrame 过滤操作。
通过构建索引字典，将 O(n*m) 的过滤操作降为 O(1) 的字典查找。
"""

import pandas as pd
from typing import Dict, Tuple, Optional, Set
from collections import defaultdict


class DataIndexer:
    """
    数据索引器 - 为 DataFrame 构建 (material, location) 索引
    
    优化效果:
    - 原来: 每次 process_node 调用 6 次 df.filter，每次 O(n)
    - 现在: 一次性构建索引 O(n)，后续查找 O(1)
    - 对于 100 个节点: 从 600n 降为 6n + 100*6 ≈ 6n
    """
    
    def __init__(self):
        """初始化索引器。"""
        # 按 (material, location) 的索引
        self._ml_index: Dict[str, Dict[Tuple[str, str], pd.DataFrame]] = {}
        # 按 (material, receiving) 的索引
        self._mr_index: Dict[str, Dict[Tuple[str, str], pd.DataFrame]] = {}
        # 按 (material, sending) 的索引（出库）
        self._ms_index: Dict[str, Dict[Tuple[str, str], pd.DataFrame]] = {}
    
    def build_index(
        self,
        name: str,
        df: pd.DataFrame,
        material_col: str = 'material',
        location_col: str = 'location',
        extra_filter: Optional[Dict[str, any]] = None
    ) -> 'DataIndexer':
        """
        构建 (material, location) 索引。
        
        Args:
            name: 索引名称
            df: 数据 DataFrame
            material_col: 物料列名
            location_col: 位置列名
            extra_filter: 额外的过滤条件 {列名: 值}
            
        Returns:
            self: 支持链式调用
        """
        if df is None or df.empty or material_col not in df.columns:
            self._ml_index[name] = {}
            return self
        
        # 应用额外过滤
        if extra_filter:
            for col, val in extra_filter.items():
                if col in df.columns:
                    if callable(val):
                        df = df[val(df[col])]
                    else:
                        df = df[df[col] == val]
        
        index = {}
        if location_col in df.columns:
            grouped = df.groupby([material_col, location_col], sort=False)
            for (mat, loc), group in grouped:
                index[(str(mat), str(loc))] = group
        
        self._ml_index[name] = index
        return self
    
    def build_receiving_index(
        self,
        name: str,
        df: pd.DataFrame,
        material_col: str = 'material',
        receiving_col: str = 'receiving'
    ) -> 'DataIndexer':
        """
        构建 (material, receiving) 索引。
        
        Args:
            name: 索引名称
            df: 数据 DataFrame
            material_col: 物料列名
            receiving_col: 接收位置列名
            
        Returns:
            self: 支持链式调用
        """
        if df is None or df.empty or material_col not in df.columns:
            self._mr_index[name] = {}
            return self
        
        index = {}
        if receiving_col in df.columns:
            grouped = df.groupby([material_col, receiving_col], sort=False)
            for (mat, recv), group in grouped:
                index[(str(mat), str(recv))] = group
        
        self._mr_index[name] = index
        return self
    
    def build_sending_index(
        self,
        name: str,
        df: pd.DataFrame,
        material_col: str = 'material',
        sending_col: str = 'sending',
        receiving_col: str = 'receiving'
    ) -> 'DataIndexer':
        """
        构建 (material, sending) 索引（用于调拨出库）。
        排除 sending == receiving 的记录。
        
        Args:
            name: 索引名称
            df: 数据 DataFrame
            material_col: 物料列名
            sending_col: 发送位置列名
            receiving_col: 接收位置列名
            
        Returns:
            self: 支持链式调用
        """
        if df is None or df.empty or material_col not in df.columns:
            self._ms_index[name] = {}
            return self
        
        index = {}
        if sending_col in df.columns and receiving_col in df.columns:
            # 过滤掉 sending == receiving 的记录
            filtered = df[df[sending_col] != df[receiving_col]]
            grouped = filtered.groupby([material_col, sending_col], sort=False)
            for (mat, send), group in grouped:
                index[(str(mat), str(send))] = group
        
        self._ms_index[name] = index
        return self
    
    def get_ml(
        self,
        name: str,
        material: str,
        location: str
    ) -> pd.DataFrame:
        """
        获取 (material, location) 过滤后的数据。
        
        Args:
            name: 索引名称
            material: 物料编码
            location: 位置编码
            
        Returns:
            pd.DataFrame: 过滤后的数据，如果不存在返回空 DataFrame
        """
        index = self._ml_index.get(name, {})
        return index.get((str(material), str(location)), pd.DataFrame())
    
    def get_mr(
        self,
        name: str,
        material: str,
        receiving: str
    ) -> pd.DataFrame:
        """
        获取 (material, receiving) 过滤后的数据。
        
        Args:
            name: 索引名称
            material: 物料编码
            receiving: 接收位置编码
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        index = self._mr_index.get(name, {})
        return index.get((str(material), str(receiving)), pd.DataFrame())
    
    def get_ms(
        self,
        name: str,
        material: str,
        sending: str
    ) -> pd.DataFrame:
        """
        获取 (material, sending) 过滤后的数据（调拨出库）。
        
        Args:
            name: 索引名称
            material: 物料编码
            sending: 发送位置编码
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        index = self._ms_index.get(name, {})
        return index.get((str(material), str(sending)), pd.DataFrame())
    
    def get_all_keys(self, name: str, index_type: str = 'ml') -> Set[Tuple[str, str]]:
        """
        获取索引中的所有键。
        
        Args:
            name: 索引名称
            index_type: 索引类型 ('ml', 'mr', 'ms')
            
        Returns:
            Set[Tuple[str, str]]: 所有键的集合
        """
        if index_type == 'ml':
            return set(self._ml_index.get(name, {}).keys())
        elif index_type == 'mr':
            return set(self._mr_index.get(name, {}).keys())
        else:
            return set(self._ms_index.get(name, {}).keys())


def create_simulation_indexer(
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    future_production_df: pd.DataFrame,
    today_shipment_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    order_df: Optional[pd.DataFrame] = None,
    delivery_shipment_df: Optional[pd.DataFrame] = None,
) -> DataIndexer:
    """
    为 MRP 模拟创建数据索引器。
    
    Args:
        beginning_inventory_df: 期初库存数据
        in_transit_df: 在途数据
        delivery_gr_df: 收货数据
        future_production_df: 生产数据
        today_shipment_df: 今日发货数据
        open_deployment_df: 开放调拨数据
        supply_demand_df: 供需数据
        safety_stock_df: 安全库存数据
        order_df: 订单数据
        delivery_shipment_df: 发运记录
        
    Returns:
        DataIndexer: 配置好的数据索引器
    """
    indexer = DataIndexer()
    
    # 按 (material, location) 索引的数据
    (indexer
        .build_index('bi', beginning_inventory_df)
        .build_index('fp', future_production_df)
        .build_index('ts', today_shipment_df)
        .build_index('sd', supply_demand_df)
        .build_index('ss', safety_stock_df)
    )
    
    # 按 (material, receiving) 索引的数据
    (indexer
        .build_receiving_index('it', in_transit_df)
        .build_receiving_index('dgr', delivery_gr_df)
    )
    
    # 调拨出库索引
    indexer.build_sending_index('od_out', open_deployment_df)
    
    # 可选数据
    if order_df is not None:
        indexer.build_index('order', order_df)
    
    if delivery_shipment_df is not None:
        sending_col = 'sending' if 'sending' in delivery_shipment_df.columns else 'location'
        indexer.build_index('ds', delivery_shipment_df, location_col=sending_col)
    
    return indexer

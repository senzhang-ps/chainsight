"""
增量更新处理器 - 智能识别和处理变化的数据

本模块提供:
1. IncrementalProcessor - 增量数据处理
2. ChangeDetector - 变化检测
3. DeltaCalculator - 增量计算
"""

import hashlib
import time
import logging
from typing import Dict, Any, Optional, Set, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ChangeSet:
    """变化集合"""
    added: pd.DataFrame = field(default_factory=pd.DataFrame)
    modified: pd.DataFrame = field(default_factory=pd.DataFrame)
    deleted: pd.DataFrame = field(default_factory=pd.DataFrame)
    unchanged_count: int = 0
    
    @property
    def has_changes(self) -> bool:
        """判断本次比较是否存在新增、修改或删除记录。"""
        return len(self.added) > 0 or len(self.modified) > 0 or len(self.deleted) > 0
    
    @property
    def total_changes(self) -> int:
        """返回本次比较中发生变化的记录总数。"""
        return len(self.added) + len(self.modified) + len(self.deleted)
    
    def summary(self) -> str:
        """返回变化集合的摘要字符串。"""
        return f"Added: {len(self.added)}, Modified: {len(self.modified)}, Deleted: {len(self.deleted)}, Unchanged: {self.unchanged_count}"


class ChangeDetector:
    """
    变化检测器
    
    通过哈希比较快速识别数据变化
    """
    
    def __init__(self, key_columns: List[str]):
        """
        参数：
            key_columns: 用于唯一标识记录的列
        """
        self.key_columns = key_columns
        self._previous_hashes: Dict[str, str] = {}
        self._previous_data: Optional[pd.DataFrame] = None
    
    def compute_row_hash(self, row: pd.Series) -> str:
        """计算行的哈希值"""
        # 将行数据转换为字符串并计算MD5
        row_str = '|'.join(str(v) for v in row.values)
        return hashlib.md5(row_str.encode()).hexdigest()
    
    def compute_key(self, row: pd.Series) -> str:
        """计算行的主键"""
        return '|'.join(str(row[col]) for col in self.key_columns)
    
    def detect_changes(self, current_df: pd.DataFrame) -> ChangeSet:
        """
        检测数据变化
        
        参数：
            current_df: 当前数据
        
        返回：
            ChangeSet: 变化集合
        """
        if current_df.empty:
            if self._previous_data is not None and not self._previous_data.empty:
                return ChangeSet(deleted=self._previous_data)
            return ChangeSet()
        
        # 计算当前数据的哈希
        current_hashes = {}
        for idx, row in current_df.iterrows():
            key = self.compute_key(row)
            hash_val = self.compute_row_hash(row)
            current_hashes[key] = (hash_val, idx)
        
        # 初次运行，所有数据都是新增
        if not self._previous_hashes:
            self._previous_hashes = {k: v[0] for k, v in current_hashes.items()}
            self._previous_data = current_df.copy()
            return ChangeSet(added=current_df, unchanged_count=0)
        
        # 检测变化
        previous_keys = set(self._previous_hashes.keys())
        current_keys = set(current_hashes.keys())
        
        # 新增的键
        added_keys = current_keys - previous_keys
        # 删除的键
        deleted_keys = previous_keys - current_keys
        # 可能修改的键
        common_keys = current_keys & previous_keys
        
        # 找出修改的记录
        modified_keys = set()
        for key in common_keys:
            if current_hashes[key][0] != self._previous_hashes[key]:
                modified_keys.add(key)
        
        # 构建结果
        added_indices = [current_hashes[k][1] for k in added_keys]
        modified_indices = [current_hashes[k][1] for k in modified_keys]
        
        changeset = ChangeSet(
            added=current_df.loc[added_indices] if added_indices else pd.DataFrame(),
            modified=current_df.loc[modified_indices] if modified_indices else pd.DataFrame(),
            deleted=self._previous_data[
                self._previous_data.apply(lambda r: self.compute_key(r) in deleted_keys, axis=1)
            ] if self._previous_data is not None and deleted_keys else pd.DataFrame(),
            unchanged_count=len(common_keys) - len(modified_keys)
        )
        
        # 更新缓存
        self._previous_hashes = {k: v[0] for k, v in current_hashes.items()}
        self._previous_data = current_df.copy()
        
        return changeset
    
    def reset(self):
        """重置状态"""
        self._previous_hashes.clear()
        self._previous_data = None


class IncrementalProcessor:
    """
    增量数据处理器
    
    只处理变化的数据，大幅减少计算量
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/incremental")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 各数据集的变化检测器
        self._detectors: Dict[str, ChangeDetector] = {}
        
        # DuckDB 连接
        self.duckdb_conn = None
        if DUCKDB_AVAILABLE:
            self.duckdb_conn = duckdb.connect(':memory:')
        
        # 统计
        self.stats = {
            'total_records_processed': 0,
            'incremental_records_processed': 0,
            'full_recalculations': 0,
            'incremental_calculations': 0,
            'time_saved_estimate_s': 0
        }
    
    def register_dataset(self, name: str, key_columns: List[str]):
        """
        注册数据集用于增量检测
        
        参数：
            name: 数据集名称
            key_columns: 主键列
        """
        self._detectors[name] = ChangeDetector(key_columns)
    
    def get_changes(self, name: str, current_df: pd.DataFrame) -> ChangeSet:
        """
        获取数据集的变化
        
        参数：
            name: 数据集名称
            current_df: 当前数据
        
        返回：
            ChangeSet: 变化集合
        """
        if name not in self._detectors:
            raise ValueError(f"数据集 '{name}' 未注册")
        
        return self._detectors[name].detect_changes(current_df)
    
    def incremental_net_demand(
        self,
        orders_changeset: ChangeSet,
        inventory_changeset: ChangeSet,
        full_inventory_df: pd.DataFrame,
        cached_results: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        增量净需求计算
        
        只重新计算受影响的物料-位置组合
        """
        if not self.duckdb_conn:
            return pd.DataFrame()
        
        # 确定受影响的物料-位置组合
        affected_keys = set()
        
        # 订单变化影响的键
        for df in [orders_changeset.added, orders_changeset.modified, orders_changeset.deleted]:
            if not df.empty and 'material' in df.columns and 'location' in df.columns:
                for _, row in df.iterrows():
                    affected_keys.add((row['material'], row['location']))
        
        # 库存变化影响的键
        for df in [inventory_changeset.added, inventory_changeset.modified, inventory_changeset.deleted]:
            if not df.empty and 'material' in df.columns and 'location' in df.columns:
                for _, row in df.iterrows():
                    affected_keys.add((row['material'], row['location']))
        
        if not affected_keys:
            self.stats['incremental_calculations'] += 1
            return cached_results if cached_results is not None else pd.DataFrame()
        
        # 只重新计算受影响的组合
        if cached_results is not None and not cached_results.empty:
            # 保留未受影响的缓存结果
            unchanged_mask = ~cached_results.apply(
                lambda r: (r['material'], r['location']) in affected_keys, axis=1
            )
            unchanged_results = cached_results[unchanged_mask]
        else:
            unchanged_results = pd.DataFrame()
        
        # 计算受影响的组合
        affected_materials = [k[0] for k in affected_keys]
        affected_locations = [k[1] for k in affected_keys]
        
        # 创建受影响的订单和库存
        # (这里简化处理，实际应从完整数据中筛选)
        
        self.stats['incremental_records_processed'] += len(affected_keys)
        self.stats['incremental_calculations'] += 1
        
        logger.info(f"增量计算: {len(affected_keys)} 个物料-位置组合受影响")
        
        # 返回合并结果
        # (实际实现需要执行DuckDB查询)
        return unchanged_results
    
    def incremental_priority_allocation(
        self,
        demand_changeset: ChangeSet,
        inventory_changeset: ChangeSet,
        priority_config: pd.DataFrame,
        cached_allocation: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        增量优先级分配
        
        只重新分配受影响的需求
        """
        if not demand_changeset.has_changes and not inventory_changeset.has_changes:
            return cached_allocation if cached_allocation is not None else pd.DataFrame()
        
        # 确定需要重新分配的物料-发送位置组合
        affected_sending = set()
        
        for df in [demand_changeset.added, demand_changeset.modified]:
            if not df.empty and 'material' in df.columns and 'sending' in df.columns:
                for _, row in df.iterrows():
                    affected_sending.add((row['material'], row['sending']))
        
        for df in [inventory_changeset.added, inventory_changeset.modified]:
            if not df.empty and 'material' in df.columns and 'location' in df.columns:
                for _, row in df.iterrows():
                    affected_sending.add((row['material'], row['location']))
        
        if not affected_sending:
            return cached_allocation if cached_allocation is not None else pd.DataFrame()
        
        self.stats['incremental_records_processed'] += len(affected_sending)
        
        logger.info(f"增量分配: {len(affected_sending)} 个物料-位置组合需要重新分配")
        
        # (实际实现需要执行重新分配逻辑)
        return cached_allocation if cached_allocation is not None else pd.DataFrame()
    
    def save_checkpoint(self, name: str, data: pd.DataFrame, date: str):
        """
        保存检查点
        
        参数：
            name: 检查点名称
            data: 数据
            date: 日期
        """
        checkpoint_file = self.cache_dir / f"{name}_{date}.parquet"
        data.to_parquet(checkpoint_file, index=False)
        logger.debug(f"保存检查点: {checkpoint_file}")
    
    def load_checkpoint(self, name: str, date: str) -> Optional[pd.DataFrame]:
        """
        加载检查点
        
        参数：
            name: 检查点名称
            date: 日期
        
        返回：
            数据DataFrame或None
        """
        checkpoint_file = self.cache_dir / f"{name}_{date}.parquet"
        if checkpoint_file.exists():
            logger.debug(f"加载检查点: {checkpoint_file}")
            return pd.read_parquet(checkpoint_file)
        return None
    
    def cleanup(self, keep_days: int = 7):
        """
        清理旧检查点
        
        参数：
            keep_days: 保留天数
        """
        import os
        from datetime import timedelta
        
        cutoff_time = time.time() - keep_days * 86400
        
        for file in self.cache_dir.glob("*.parquet"):
            if os.path.getmtime(file) < cutoff_time:
                file.unlink()
                logger.debug(f"删除旧检查点: {file}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats['total_records_processed']
        incremental = self.stats['incremental_records_processed']
        
        if total > 0:
            efficiency = (1 - incremental / total) * 100
        else:
            efficiency = 0
        
        return {
            **self.stats,
            'efficiency': f"{efficiency:.1f}%",
            'records_saved': total - incremental
        }
    
    def close(self):
        """关闭连接"""
        if self.duckdb_conn:
            self.duckdb_conn.close()


class DeltaCalculator:
    """
    增量计算器
    
    提供各种增量计算方法
    """
    
    @staticmethod
    def delta_inventory(
        previous_inventory: pd.DataFrame,
        transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        增量库存计算
        
        new_inventory = previous_inventory + inbound - outbound
        """
        if previous_inventory.empty:
            return transactions
        
        # 按物料-位置聚合变化
        if not transactions.empty:
            delta = transactions.groupby(['material', 'location'])['quantity'].sum().reset_index()
            delta.columns = ['material', 'location', 'delta_qty']
        else:
            delta = pd.DataFrame(columns=['material', 'location', 'delta_qty'])
        
        # 合并计算
        result = previous_inventory.merge(
            delta,
            on=['material', 'location'],
            how='left'
        )
        result['delta_qty'] = result['delta_qty'].fillna(0)
        result['quantity'] = result['quantity'] + result['delta_qty']
        
        return result[['material', 'location', 'quantity']]
    
    @staticmethod
    def delta_demand(
        base_demand: pd.DataFrame,
        new_orders: pd.DataFrame,
        fulfilled_orders: pd.DataFrame
    ) -> pd.DataFrame:
        """
        增量需求计算
        
        new_demand = base_demand + new_orders - fulfilled_orders
        """
        # 聚合新订单
        if not new_orders.empty:
            new_agg = new_orders.groupby(['material', 'location', 'date'])['quantity'].sum().reset_index()
        else:
            new_agg = pd.DataFrame(columns=['material', 'location', 'date', 'quantity'])
        
        # 聚合已完成订单
        if not fulfilled_orders.empty:
            fulfilled_agg = fulfilled_orders.groupby(['material', 'location', 'date'])['quantity'].sum().reset_index()
            fulfilled_agg['quantity'] = -fulfilled_agg['quantity']
        else:
            fulfilled_agg = pd.DataFrame(columns=['material', 'location', 'date', 'quantity'])
        
        # 合并所有变化
        all_changes = pd.concat([base_demand, new_agg, fulfilled_agg], ignore_index=True)
        
        # 聚合计算
        result = all_changes.groupby(['material', 'location', 'date'])['quantity'].sum().reset_index()
        result = result[result['quantity'] > 0]  # 过滤掉负数或零
        
        return result
    
    @staticmethod
    def identify_ripple_effects(
        changed_items: Set[Tuple[str, str]],
        bom_df: pd.DataFrame,
        network_df: pd.DataFrame
    ) -> Set[Tuple[str, str]]:
        """
        识别连锁影响
        
        当某个物料-位置变化时，识别所有受影响的下游
        
        参数：
            changed_items: 变化的物料-位置集合
            bom_df: BOM数据
            network_df: 网络配置数据
        
        返回：
            所有受影响的物料-位置集合
        """
        affected = set(changed_items)
        to_process = list(changed_items)
        
        # 构建BOM父子映射
        bom_children = {}
        if not bom_df.empty and 'parent_material' in bom_df.columns:
            for _, row in bom_df.iterrows():
                parent = row['parent_material']
                if parent not in bom_children:
                    bom_children[parent] = []
                bom_children[parent].append(row['child_material'])
        
        # 构建网络发送-接收映射
        network_downstream = {}
        if not network_df.empty and 'sending' in network_df.columns:
            for _, row in network_df.iterrows():
                sending = row['sending']
                if sending not in network_downstream:
                    network_downstream[sending] = []
                network_downstream[sending].append(row['receiving'])
        
        # 使用 BFS 遍历受影响节点
        while to_process:
            material, location = to_process.pop(0)
            
            # BOM 下游节点
            if material in bom_children:
                for child in bom_children[material]:
                    key = (child, location)
                    if key not in affected:
                        affected.add(key)
                        to_process.append(key)
            
            # 网络下游
            if location in network_downstream:
                for downstream_loc in network_downstream[location]:
                    key = (material, downstream_loc)
                    if key not in affected:
                        affected.add(key)
                        to_process.append(key)
        
        return affected


# ===================== 工厂函数 =====================

def create_incremental_processor(
    cache_dir: Optional[str] = None,
    datasets: Optional[Dict[str, List[str]]] = None
) -> IncrementalProcessor:
    """
    创建增量处理器
    
    参数：
        cache_dir: 缓存目录
        datasets: 数据集配置 {name: key_columns}
    
    返回：
        IncrementalProcessor实例
    """
    processor = IncrementalProcessor(cache_dir)
    
    # 注册默认数据集
    default_datasets = {
        'orders': ['order_id'],
        'inventory': ['material', 'location'],
        'demand': ['material', 'location', 'date'],
        'deployment': ['material', 'sending', 'receiving', 'date'],
        'production': ['material', 'location', 'production_date']
    }
    
    if datasets:
        default_datasets.update(datasets)
    
    for name, key_cols in default_datasets.items():
        processor.register_dataset(name, key_cols)
    
    return processor

# -*- coding: utf-8 -*-
"""
仿真预计算缓存模块

缓存跨天不变的计算结果，避免重复计算。

核心优化:
1. Network层级分配只计算一次
2. PTF/LSK缓存只构建一次
3. LeadTime缓存只构建一次
4. Network索引只构建一次
5. DeployConfig索引只构建一次

预期性能提升: 20-30%
"""
import time
from typing import Dict, Optional, Set, Tuple
import pandas as pd


class SimulationCache:
    """
    仿真预计算缓存。
    
    缓存跨天不变的配置数据和计算结果，避免每天重复计算。
    """
    
    def __init__(self, config_dict: dict, sim_start_date: str, sim_end_date: str):
        """
        初始化缓存并预计算所有可缓存的数据。
        
        Args:
            config_dict: 配置字典
            sim_start_date: 仿真开始日期
            sim_end_date: 仿真结束日期
        """
        self.config_dict = config_dict
        self.sim_start = pd.to_datetime(sim_start_date)
        self.sim_end = pd.to_datetime(sim_end_date)
        
        # 缓存的计算结果
        self._layer_assignment: Optional[pd.DataFrame] = None
        self._ptf_lsk_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._lead_time_cache: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
        self._network_index: Dict[Tuple[str, str], str] = {}  # (material, location) -> sourcing
        self._deploy_config_index: Dict[Tuple[str, str], Tuple[float, float]] = {}  # (material, sending) -> (moq, rv)
        self._safety_stock_index: Dict[Tuple[str, str, str], float] = {}  # (material, location, date) -> qty
        
        # 预过滤的数据
        self._active_network: Optional[pd.DataFrame] = None
        self._filtered_safety_stock: Optional[pd.DataFrame] = None
        
        # 统计信息
        self._build_time_ms = 0.0
        self._hit_count = 0
        self._miss_count = 0
        
        # 构建缓存
        self._build_all_caches()
    
    def _build_all_caches(self):
        """构建所有缓存。"""
        t_start = time.perf_counter()
        
        print("📊 预构建仿真缓存...")
        
        # 1. 构建Network缓存和层级分配
        self._build_network_cache()
        
        # 2. 构建PTF/LSK缓存
        self._build_ptf_lsk_cache()
        
        # 3. 构建LeadTime缓存
        self._build_lead_time_cache()
        
        # 4. 构建DeployConfig缓存
        self._build_deploy_config_cache()
        
        # 5. 构建SafetyStock索引
        self._build_safety_stock_index()
        
        self._build_time_ms = (time.perf_counter() - t_start) * 1000
        
        print(f"✅ 缓存构建完成: {self._build_time_ms:.1f}ms")
        print(f"   - Network索引: {len(self._network_index)} 条")
        print(f"   - PTF/LSK缓存: {len(self._ptf_lsk_cache)} 条")
        print(f"   - LeadTime缓存: {len(self._lead_time_cache)} 条")
        print(f"   - DeployConfig缓存: {len(self._deploy_config_index)} 条")
        print(f"   - SafetyStock索引: {len(self._safety_stock_index)} 条")
    
    def _build_network_cache(self):
        """构建Network缓存和层级分配。"""
        network_df = self.config_dict.get('Global_Network', pd.DataFrame())
        if network_df.empty:
            return
        
        # 标准化
        network_df = network_df.copy()
        for col in ['material', 'location', 'sourcing']:
            if col in network_df.columns:
                network_df[col] = network_df[col].astype(str).str.strip()
        
        # 过滤有效期内的数据（使用整个仿真期间）
        if 'eff_from' in network_df.columns and 'eff_to' in network_df.columns:
            network_df['eff_from'] = pd.to_datetime(network_df['eff_from'])
            network_df['eff_to'] = pd.to_datetime(network_df['eff_to'])
            self._active_network = network_df[
                (network_df['eff_from'] <= self.sim_end) &
                (network_df['eff_to'] >= self.sim_start)
            ].copy()
        else:
            self._active_network = network_df.copy()
        
        # 构建 (material, location) -> sourcing 索引
        for _, row in self._active_network.iterrows():
            key = (str(row['material']), str(row['location']))
            self._network_index[key] = str(row.get('sourcing', ''))
        
        # 构建层级分配
        self._build_layer_assignment()
    
    def _build_layer_assignment(self):
        """构建层级分配（只计算一次）。"""
        if self._active_network is None or self._active_network.empty:
            return
        
        from ..modules.deployment_planning.cache_utils import assign_location_layers
        self._layer_assignment = assign_location_layers(self._active_network)
    
    def _build_ptf_lsk_cache(self):
        """构建PTF/LSK缓存。"""
        m4_mlcfg = self.config_dict.get('M4_MaterialLocationLineCfg', pd.DataFrame())
        if m4_mlcfg.empty:
            return
        
        m4_mlcfg = m4_mlcfg.copy()
        for col in ['material', 'location']:
            if col in m4_mlcfg.columns:
                m4_mlcfg[col] = m4_mlcfg[col].astype(str).str.strip()
        
        ptf_col = 'ptf' if 'ptf' in m4_mlcfg.columns else 'PTF'
        lsk_col = 'lsk' if 'lsk' in m4_mlcfg.columns else 'LSK'
        
        for _, row in m4_mlcfg.iterrows():
            key = (str(row['material']), str(row['location']))
            ptf = str(row.get(ptf_col, ''))
            lsk = str(row.get(lsk_col, ''))
            self._ptf_lsk_cache[key] = (ptf, lsk)
    
    def _build_lead_time_cache(self):
        """构建LeadTime缓存。"""
        leadtime_df = self.config_dict.get('Global_LeadTime', pd.DataFrame())
        if leadtime_df.empty:
            return
        
        leadtime_df = leadtime_df.copy()
        for col in ['sending', 'receiving']:
            if col in leadtime_df.columns:
                leadtime_df[col] = leadtime_df[col].astype(str).str.strip()
        
        for _, row in leadtime_df.iterrows():
            key = (str(row['sending']), str(row['receiving']))
            pdt = int(row.get('pdt', row.get('PDT', 0)) or 0)
            gr = int(row.get('gr', row.get('GR', 0)) or 0)
            mct = int(row.get('mct', row.get('MCT', 0)) or 0)
            self._lead_time_cache[key] = (pdt, gr, mct)
    
    def _build_deploy_config_cache(self):
        """构建DeployConfig缓存。"""
        deploy_config = self.config_dict.get('M5_DeployConfig', pd.DataFrame())
        if deploy_config.empty:
            return
        
        deploy_config = deploy_config.copy()
        for col in ['material', 'sending']:
            if col in deploy_config.columns:
                deploy_config[col] = deploy_config[col].astype(str).str.strip()
        
        moq_col = 'moq' if 'moq' in deploy_config.columns else 'MOQ'
        rv_col = 'rv' if 'rv' in deploy_config.columns else 'RV'
        
        for _, row in deploy_config.iterrows():
            key = (str(row['material']), str(row['sending']))
            moq = float(row.get(moq_col, 0) or 0)
            rv = float(row.get(rv_col, 0) or 0)
            self._deploy_config_index[key] = (moq, rv)
    
    def _build_safety_stock_index(self):
        """构建SafetyStock索引。"""
        ss_df = self.config_dict.get('M3_SafetyStock', pd.DataFrame())
        if ss_df.empty:
            return
        
        ss_df = ss_df.copy()
        for col in ['material', 'location']:
            if col in ss_df.columns:
                ss_df[col] = ss_df[col].astype(str).str.strip()
        
        if 'date' in ss_df.columns:
            ss_df['date'] = pd.to_datetime(ss_df['date']).dt.strftime('%Y-%m-%d')
        
        qty_col = 'safety_stock' if 'safety_stock' in ss_df.columns else 'SafetyStock'
        if qty_col not in ss_df.columns:
            qty_col = 'quantity' if 'quantity' in ss_df.columns else None
        
        if qty_col:
            for _, row in ss_df.iterrows():
                key = (str(row['material']), str(row['location']), str(row.get('date', '')))
                self._safety_stock_index[key] = float(row.get(qty_col, 0) or 0)
    
    # ============== 缓存访问接口 ==============
    
    def get_layer_assignment(self) -> pd.DataFrame:
        """获取预计算的层级分配。"""
        self._hit_count += 1
        return self._layer_assignment if self._layer_assignment is not None else pd.DataFrame()
    
    def get_active_network(self, sim_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        获取有效的Network数据。
        
        如果sim_date在缓存的仿真期间内，返回缓存的数据。
        """
        self._hit_count += 1
        return self._active_network if self._active_network is not None else pd.DataFrame()
    
    def get_ptf_lsk(self, material: str, location: str) -> Tuple[str, str]:
        """
        获取PTF/LSK值。
        
        Returns:
            (ptf, lsk) 元组，未找到返回 ('', '')
        """
        key = (str(material), str(location))
        result = self._ptf_lsk_cache.get(key, ('', ''))
        if result != ('', ''):
            self._hit_count += 1
        else:
            self._miss_count += 1
        return result
    
    def get_lead_time(self, sending: str, receiving: str) -> Tuple[int, int, int]:
        """
        获取LeadTime值。
        
        Returns:
            (pdt, gr, mct) 元组，未找到返回 (0, 0, 0)
        """
        key = (str(sending), str(receiving))
        result = self._lead_time_cache.get(key, (0, 0, 0))
        if result != (0, 0, 0):
            self._hit_count += 1
        else:
            self._miss_count += 1
        return result
    
    def get_upstream(self, material: str, location: str) -> str:
        """
        获取上游节点(sourcing)。
        
        Returns:
            sourcing字符串，未找到返回 ''
        """
        key = (str(material), str(location))
        result = self._network_index.get(key, '')
        if result:
            self._hit_count += 1
        else:
            self._miss_count += 1
        return result
    
    def get_moq_rv(self, material: str, sending: str) -> Tuple[float, float]:
        """
        获取MOQ/RV值。
        
        Returns:
            (moq, rv) 元组，未找到返回 (0, 0)
        """
        key = (str(material), str(sending))
        result = self._deploy_config_index.get(key, (0.0, 0.0))
        if result != (0.0, 0.0):
            self._hit_count += 1
        else:
            self._miss_count += 1
        return result
    
    def get_safety_stock(self, material: str, location: str, date: str) -> float:
        """
        获取安全库存值。
        
        Returns:
            安全库存数量，未找到返回 0.0
        """
        key = (str(material), str(location), str(date))
        result = self._safety_stock_index.get(key, 0.0)
        if result > 0:
            self._hit_count += 1
        else:
            self._miss_count += 1
        return result
    
    def get_all_material_location_pairs(self) -> Set[Tuple[str, str]]:
        """获取所有 (material, location) 对。"""
        return set(self._network_index.keys())
    
    # ============== 统计接口 ==============
    
    def get_stats(self) -> dict:
        """获取缓存统计信息。"""
        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100) if total > 0 else 0
        
        return {
            'build_time_ms': self._build_time_ms,
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'hit_rate': f"{hit_rate:.1f}%",
            'network_entries': len(self._network_index),
            'ptf_lsk_entries': len(self._ptf_lsk_cache),
            'lead_time_entries': len(self._lead_time_cache),
            'deploy_config_entries': len(self._deploy_config_index),
            'safety_stock_entries': len(self._safety_stock_index),
        }
    
    def print_stats(self):
        """打印缓存统计信息。"""
        stats = self.get_stats()
        print("\n📊 仿真缓存统计:")
        print(f"   构建时间: {stats['build_time_ms']:.1f}ms")
        print(f"   命中次数: {stats['hit_count']}")
        print(f"   未命中次数: {stats['miss_count']}")
        print(f"   命中率: {stats['hit_rate']}")


# 全局缓存实例
_global_cache: Optional[SimulationCache] = None


def get_simulation_cache() -> Optional[SimulationCache]:
    """获取全局仿真缓存实例。"""
    return _global_cache


def initialize_simulation_cache(config_dict: dict, sim_start: str, sim_end: str) -> SimulationCache:
    """
    初始化全局仿真缓存。
    
    Args:
        config_dict: 配置字典
        sim_start: 仿真开始日期
        sim_end: 仿真结束日期
        
    Returns:
        SimulationCache实例
    """
    global _global_cache
    _global_cache = SimulationCache(config_dict, sim_start, sim_end)
    return _global_cache


def clear_simulation_cache():
    """清除全局仿真缓存。"""
    global _global_cache
    _global_cache = None

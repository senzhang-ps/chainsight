# -*- coding: utf-8 -*-
"""
M5 多进程优化器

使用多进程突破 GIL 限制，实现层内节点的真正并行处理。

优化历史:
- v1.0: 基础多进程实现
- v1.1: 动态CPU配置，使用90%CPU资源
"""

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd

# 使用统一的CPU配置
from ...utils.resource_config import MAX_WORKERS, get_optimal_workers


def _serialize_config(config: dict) -> bytes:
    """序列化配置（移除不可序列化的对象）。"""
    serializable = {}
    for key, value in config.items():
        if isinstance(value, pd.DataFrame):
            serializable[key] = value.to_dict('records')
        elif isinstance(value, dict):
            serializable[key] = value
        elif isinstance(value, (str, int, float, bool, list)):
            serializable[key] = value
    return pickle.dumps(serializable)


def _deserialize_config(data: bytes) -> dict:
    """反序列化配置。"""
    config = pickle.loads(data)
    result = {}
    for key, value in config.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            result[key] = pd.DataFrame(value)
        else:
            result[key] = value
    return result


def _process_node_worker(
    args: Tuple[str, str, pd.Timestamp, bytes, dict, dict, dict, dict, dict, dict, dict, dict]
) -> Tuple[str, str, List[dict]]:
    """
    工作进程中处理单个节点。
    
    由于跨进程传递，config 需要序列化。
    """
    from .demand_collector import collect_node_demands
    
    (mat, loc, sim_date, config_data, up_gap_buffer,
     ptf_lsk_cache, lead_time_cache, active_network_cache,
     sdl_index, ss_index, order_index, deploy_config_index) = args
    
    # 反序列化配置
    config = _deserialize_config(config_data)
    
    # 收集需求
    demands = collect_node_demands(
        mat, loc, sim_date, config, up_gap_buffer,
        ptf_lsk_cache, lead_time_cache, active_network_cache,
        sdl_index, ss_index, order_index, deploy_config_index
    )
    
    return (mat, loc, demands)


def process_layer_multiprocess(
    all_pairs: Set[Tuple[str, str]],
    sim_date: pd.Timestamp,
    config: dict,
    up_gap_buffer: dict,
    ptf_lsk_cache: dict,
    lead_time_cache: dict,
    active_network_cache: dict,
    sdl_index: Optional[dict] = None,
    ss_index: Optional[dict] = None,
    order_index: Optional[dict] = None,
    deploy_config_index: Optional[dict] = None,
    max_workers: Optional[int] = None
) -> Dict[Tuple[str, str], List[dict]]:
    """
    使用多进程并行处理层内所有节点。
    
    参数：
        all_pairs: (material, location) 对集合
        sim_date: 仿真日期
        config: 配置字典
        up_gap_buffer: 上游缺口缓冲区
        ptf_lsk_cache: PTF/LSK缓存
        lead_time_cache: LeadTime缓存
        active_network_cache: Network缓存
        sdl_index: SDL预建索引
        ss_index: SafetyStock预建索引
        order_index: OrderLog预建索引
        deploy_config_index: DeployConfig预建索引
        max_workers: 最大工作进程数
        
    返回：
        dict: (material, location) -> 需求行列表
    """
    if not all_pairs:
        return {}
    
    # 排序确保遍历顺序一致
    sorted_pairs = sorted(all_pairs)
    
    n_workers = max_workers or MAX_WORKERS
    n_workers = min(n_workers, len(sorted_pairs))
    
    # 序列化配置
    config_data = _serialize_config(config)
    
    # 准备任务参数
    tasks = [
        (mat, loc, sim_date, config_data, up_gap_buffer,
         ptf_lsk_cache, lead_time_cache, active_network_cache,
         sdl_index, ss_index, order_index, deploy_config_index)
        for (mat, loc) in sorted_pairs
    ]
    
    result = {}
    
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_node_worker, task): task[:2] 
                      for task in tasks}
            
            for future in as_completed(futures):
                key = futures[future]
                try:
                    mat, loc, demands = future.result()
                    result[(mat, loc)] = demands
                except Exception as e:
                    result[key] = []
                    
    except Exception as e:
        # 回退到串行处理
        from .demand_collector import collect_node_demands
        for mat, loc in sorted_pairs:
            result[(mat, loc)] = collect_node_demands(
                mat, loc, sim_date, config, up_gap_buffer,
                ptf_lsk_cache, lead_time_cache, active_network_cache,
                sdl_index, ss_index, order_index, deploy_config_index
            )
    
    return result

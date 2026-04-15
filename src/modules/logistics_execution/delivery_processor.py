# -*- coding: utf-8 -*-
"""
发货处理模块

提供 Module6 的发货处理功能，包括：
- 延迟采样
- MDQ 旁路规则判断
- 交货时间计算

典型用法示例:
    delay = sample_delivery_delay('WH1', 'WH2', delay_dist_df)
    bypass, rule_id = should_bypass_mdq(context, rules, evaluator)
    lead_time = calculate_lead_time(lead_time_df, 'WH1', 'WH2')
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .expression_evaluator import SafeExpressionEvaluator


def sample_delivery_delay(
    sending: str,
    receiving: str,
    dist_df: pd.DataFrame
) -> int:
    """
    采样配送延迟天数。
    
    优先级：
    1. 精确路线匹配
    2. 全局兜底规则（sending=ALL & receiving=ALL）
    3. 默认 0 天
    
    参数：
        sending: 发送地点
        receiving: 接收地点
        dist_df: 延迟分布 DataFrame
        
    返回：
        采样的延迟天数
    """
    if not _is_valid_delay_distribution(dist_df):
        return 0
    
    delays, probs = _get_delay_distribution(dist_df, sending, receiving)
    
    if delays is None:
        return 0
    
    return _sample_from_distribution(delays, probs)


def _is_valid_delay_distribution(dist_df: pd.DataFrame) -> bool:
    """
    检查延迟分布数据是否有效。
    
    参数：
        dist_df: 延迟分布 DataFrame
        
    返回：
        是否有效
    """
    if dist_df is None or dist_df.empty:
        return False
    
    required_cols = {'delay_days', 'probability', 'sending', 'receiving'}
    return required_cols.issubset(set(dist_df.columns))


def _get_delay_distribution(
    dist_df: pd.DataFrame,
    sending: str,
    receiving: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    获取延迟分布数据。
    
    参数：
        dist_df: 延迟分布 DataFrame
        sending: 发送地点
        receiving: 接收地点
        
    返回：
        (延迟天数数组, 概率数组) 元组，无数据时返回 (None, None)
    """
    # 尝试精确路线匹配
    exact = dist_df[
        (dist_df['sending'] == sending) &
        (dist_df['receiving'] == receiving)
    ]
    
    if not exact.empty:
        return exact['delay_days'].to_numpy(), exact['probability'].to_numpy()
    
    # 尝试全局兜底规则
    global_mask = (
        (dist_df['sending'].astype(str).str.upper() == 'ALL') &
        (dist_df['receiving'].astype(str).str.upper() == 'ALL')
    )
    global_rows = dist_df[global_mask]
    
    if not global_rows.empty:
        return global_rows['delay_days'].to_numpy(), global_rows['probability'].to_numpy()
    
    return None, None


def _sample_from_distribution(
    delays: np.ndarray,
    probs: np.ndarray
) -> int:
    """
    从分布中采样延迟天数。
    
    参数：
        delays: 延迟天数数组
        probs: 概率数组
        
    返回：
        采样的延迟天数
    """
    probs = np.array(probs, dtype=float)
    
    if probs.sum() <= 0:
        return 0
    
    probs = probs / probs.sum()
    return int(np.random.choice(delays, p=probs))


def should_bypass_mdq(
    context: Dict[str, Any],
    rules: pd.DataFrame,
    evaluator: SafeExpressionEvaluator
) -> Tuple[bool, Optional[str]]:
    """
    判断是否应该绕过 MDQ 规则。
    
    遍历规则表，检查是否有匹配的旁路规则。
    
    参数：
        context: 上下文变量字典
        rules: 旁路规则 DataFrame
        evaluator: 表达式解析器
        
    返回：
        (是否绕过, 命中的规则ID) 元组
    """
    for rule in rules.itertuples(index=False):
        if not _rule_matches_context(rule, context):
            continue
        
        if _evaluate_rule_condition(rule, context, evaluator):
            return True, getattr(rule, 'rule_id', None)
    
    return False, None


def _rule_matches_context(rule, context: Dict[str, Any]) -> bool:
    """
    检查规则是否匹配上下文。
    
    参数：
        rule: 规则行 (namedtuple 或 Series)
        context: 上下文字典
        
    返回：
        是否匹配
    """
    match_cols = ['sending', 'receiving', 'truck_type', 'demand_element']
    
    for col in match_cols:
        # 支持 namedtuple 和 Series
        if hasattr(rule, '_fields'):
            rule_val = str(getattr(rule, col, 'ALL'))
        else:
            rule_val = str(rule.get(col, 'ALL'))
        if rule_val != 'ALL' and str(context.get(col)) != rule_val:
            return False
    
    return True


def _evaluate_rule_condition(
    rule,
    context: Dict[str, Any],
    evaluator: SafeExpressionEvaluator
) -> bool:
    """
    评估规则条件表达式。
    
    参数：
        rule: 规则行 (namedtuple 或 Series)
        context: 上下文字典
        evaluator: 表达式解析器
        
    返回：
        条件是否满足
    """
    try:
        # 支持 namedtuple 和 Series
        if hasattr(rule, '_fields'):
            expr = rule.condition_logic
            rule_id = getattr(rule, 'rule_id', None)
        else:
            expr = rule['condition_logic']
            rule_id = rule.get('rule_id')
        return evaluator.eval(expr, context)
    except Exception as e:
        return False


def calculate_lead_time(
    lead_time_df: pd.DataFrame,
    sending: str,
    receiving: str
) -> Dict[str, int]:
    """
    计算交货时间参数。
    
    参数：
        lead_time_df: 交货时间配置 DataFrame
        sending: 发送地点
        receiving: 接收地点
        
    返回：
        包含 PDT, OTD, GR 的字典
        
    异常：
        ValueError: 当缺少路线配置时
    """
    lt_rows = lead_time_df[
        (lead_time_df['sending'] == sending) &
        (lead_time_df['receiving'] == receiving)
    ]
    
    if lt_rows.empty:
        raise ValueError(f"缺少路线 {sending}->{receiving} 的 LeadTime 行")
    
    return {
        'PDT': _get_time_value(lt_rows, 'PDT'),
        'OTD': _get_time_value(lt_rows, 'OTD'),
        'GR': _get_time_value(lt_rows, 'GR'),
    }


def _get_time_value(
    lt_rows: pd.DataFrame,
    col_name: str
) -> int:
    """
    从 LeadTime 行获取时间值。
    
    参数：
        lt_rows: LeadTime 行
        col_name: 列名
        
    返回：
        时间值（天数）
    """
    # 尝试多种列名变体
    for col in [col_name, col_name.lower(), col_name.upper()]:
        if col in lt_rows.columns:
            value = pd.to_numeric(lt_rows[col].iloc[0], errors='coerce')
            return max(0, int(value) if pd.notna(value) else 0)
    
    return 0


def calculate_actual_delivery_date(
    ship_date: pd.Timestamp,
    otd: int,
    gr: int,
    delay: int
) -> pd.Timestamp:
    """
    计算实际交货日期。
    
    actual_delivery_date = actual_ship_date + OTD + GR + delay
    
    参数：
        ship_date: 发运日期
        otd: 在途时间（天）
        gr: 收货处理时间（天）
        delay: 采样的延迟天数
        
    返回：
        实际交货日期
    """
    return ship_date + pd.Timedelta(days=otd + gr + delay)


def create_delivery_record(
    vehicle_uid: str,
    uid: str,
    demand_row: pd.Series,
    load_qty: int,
    ship_date: pd.Timestamp,
    delivery_date: pd.Timestamp,
    truck_type: str,
    wfr: float,
    vfr: float
) -> Dict[str, Any]:
    """
    创建发货记录。
    
    参数：
        vehicle_uid: 车辆唯一标识
        uid: 部署计划唯一标识
        demand_row: 需求行数据
        load_qty: 装载数量
        ship_date: 发运日期
        delivery_date: 交货日期
        truck_type: 车型
        wfr: 重量填充率
        vfr: 体积填充率
        
    返回：
        发货记录字典
    """
    return {
        'vehicle_uid': vehicle_uid,
        'ori_deployment_uid': uid,
        'material': demand_row['material'],
        'sending': demand_row['sending'],
        'receiving': demand_row['receiving'],
        'planned_deployment_date': demand_row['planned_deployment_date'],
        'actual_ship_date': ship_date,
        'actual_delivery_date': delivery_date,
        'delivery_qty': load_qty,
        'truck_type': truck_type,
        'truck_load_pct': min(max(wfr, vfr), 1.0),
        'WFR': min(wfr, 1.0),
        'VFR': min(vfr, 1.0)
    }


def create_bypass_record(
    uid: str,
    rule_id: str,
    sim_date: pd.Timestamp,
    context: Dict[str, Any],
    vehicle_uid: str
) -> Dict[str, Any]:
    """
    创建旁路规则命中记录。
    
    参数：
        uid: 部署计划唯一标识
        rule_id: 规则ID
        sim_date: 仿真日期
        context: 上下文快照
        vehicle_uid: 车辆唯一标识
        
    返回：
        旁路记录字典
    """
    return {
        'ori_deployment_uid': uid,
        'rule_id': rule_id,
        'simulation_date': sim_date,
        'context_snapshot': str(context),
        'vehicle_uid': vehicle_uid
    }


def create_unsatisfied_record(
    uid: str,
    row: pd.Series,
    sending: str,
    receiving: str,
    sim_date: pd.Timestamp,
    waiting_days: int,
    accumulated_qty: float,
    min_mdq: float,
    reason: str = 'waited_too_long'
) -> Dict[str, Any]:
    """
    创建未满足需求记录。
    
    参数：
        uid: 部署计划唯一标识
        row: 需求行数据
        sending: 发送地点
        receiving: 接收地点
        sim_date: 仿真日期
        waiting_days: 等待天数
        accumulated_qty: 累计数量
        min_mdq: 最小发货量
        reason: 原因
        
    返回：
        未满足记录字典
    """
    return {
        'ori_deployment_uid': uid,
        'material': row['material'],
        'sending': sending,
        'receiving': receiving,
        'demand_element': row['demand_element'],
        'planned_deployment_date': row['planned_deployment_date'],
        'simulation_date': sim_date,
        'waiting_days': waiting_days,
        'accumulated_qty': accumulated_qty,
        'min_MDQ': min_mdq,
        'reason': reason
    }

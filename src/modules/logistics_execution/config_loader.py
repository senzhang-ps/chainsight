# -*- coding: utf-8 -*-
"""
配置加载器模块

提供 Module6 物流模块的配置数据加载功能，支持两种模式：
- 独立模式：从 Excel 文件读取配置
- 集成模式：从 Orchestrator 和配置字典读取配置

典型用法示例:
    # 独立模式
    config = load_standalone_config('input.xlsx')
    
    # 集成模式
    config = load_integrated_config(config_dict, orchestrator, current_date)
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============= 常量定义 =============
M6_CONFIG_MAPPING: Dict[str, str] = {
    'TruckReleaseCon': 'M6_TruckReleaseCon',
    'TruckCapacityPlan': 'M6_TruckCapacityPlan',
    'TruckTypeSpecs': 'M6_TruckTypeSpecs',
    'MaterialMD': 'M6_MaterialMD',
    'DeliveryDelayDistribution': 'M6_DeliveryDelayDistribution',
    'MDQBypassRules': 'M6_MDQBypassRules'
}

GLOBAL_CONFIG_MAPPING: Dict[str, str] = {
    'DemandPriority': 'Global_DemandPriority',
    'LeadTime': 'Global_LeadTime'
}

DEFAULT_DEPLOYMENT_COLUMNS: List[str] = [
    'material', 'sending', 'receiving', 'planned_deployment_date',
    'deployed_qty', 'demand_element', 'ori_deployment_uid'
]


def load_standalone_config(input_excel: str) -> Dict[str, pd.DataFrame]:
    """
    加载独立模式的配置数据（从 Excel 文件）。
    
    参数：
        input_excel: 输入 Excel 文件路径
        
    返回：
        配置数据字典，键为配置名称，值为 DataFrame
        
    异常：
        Exception: 读取 Excel 文件失败时抛出
    """
    try:
        config = _read_excel_sheets(input_excel)
        _apply_random_seed_from_file(input_excel)
        return config
    except Exception as e:
        print(f"❌ 读取输入失败: {e}")
        raise


def _read_excel_sheets(input_excel: str) -> Dict[str, pd.DataFrame]:
    """
    读取 Excel 文件中的各配置表。
    
    参数：
        input_excel: Excel 文件路径
        
    返回：
        配置数据字典
    """
    sheet_names = [
        'DeploymentPlan', 'TruckReleaseCon', 'TruckCapacityPlan',
        'TruckTypeSpecs', 'MaterialMD', 'DemandPriority',
        'LeadTime', 'DeliveryDelayDistribution', 'MDQBypassRules'
    ]
    
    return {
        name: pd.read_excel(input_excel, sheet_name=name)
        for name in sheet_names
    }


def _apply_random_seed_from_file(input_excel: str) -> None:
    """
    从 Excel 文件中读取并应用随机种子。
    
    参数：
        input_excel: Excel 文件路径
    """
    xl = pd.ExcelFile(input_excel)
    if 'RandomSeed' not in xl.sheet_names:
        return
    
    rs = pd.read_excel(input_excel, sheet_name='RandomSeed')
    if 'random_seed' not in rs.columns or rs.empty:
        return
    
    seed_value = rs.iloc[0]['random_seed']
    if pd.notna(seed_value):
        np.random.seed(int(seed_value))


def load_integrated_config(
    config_dict: Dict[str, Any],
    orchestrator: object,
    current_date: pd.Timestamp
) -> Dict[str, Any]:
    """
    加载集成配置数据，替代原来的 Excel 文件输入。
    
    参数：
        config_dict: 配置数据字典
        orchestrator: Orchestrator 实例
        current_date: 当前日期
        
    返回：
        集成配置数据字典
    """
    config: Dict[str, Any] = {}
    validation_log: List[Dict] = []
    
    try:
        config['DeploymentPlan'] = _load_deployment_plan(
            orchestrator, current_date
        )
        _load_m6_configs(config_dict, config, validation_log)
        _load_global_configs(config_dict, config, validation_log)
        _process_date_fields(config)
        config['ValidationLog'] = validation_log
        
    except Exception as e:
        print(f"❌ Error loading integrated config: {str(e)}")
        validation_log.append({
            'sheet': 'General',
            'row': '',
            'issue': f'Config loading error: {str(e)}'
        })
        config['ValidationLog'] = validation_log
        _ensure_default_config(config)
    
    return config


def _load_deployment_plan(
    orchestrator: object,
    current_date: pd.Timestamp
) -> pd.DataFrame:
    """
    从 Orchestrator 加载 OpenDeployment。
    
    参数：
        orchestrator: Orchestrator 实例
        current_date: 当前日期
        
    返回：
        部署计划 DataFrame
    """
    open_deployment = orchestrator.get_open_deployment(current_date)
    
    if open_deployment is None or open_deployment.empty:
        print(f"[WARN] No open deployment for {current_date.strftime('%Y-%m-%d')}")
        return pd.DataFrame(columns=DEFAULT_DEPLOYMENT_COLUMNS)
    
    _log_route_statistics(open_deployment)
    _ensure_date_format(open_deployment)
    planned_dates = pd.to_datetime(open_deployment['planned_deployment_date'], errors='coerce')
    planned_leq = int((planned_dates <= current_date).sum())
    print(
        f"[M6] OpenDeployment rows={len(open_deployment)} "
        f"planned<=sim_date={planned_leq} "
        f"min={planned_dates.min()} max={planned_dates.max()}"
    )
    
    return open_deployment


def _log_route_statistics(deployment_df: pd.DataFrame) -> None:
    """
    记录路线类型统计信息。
    
    参数：
        deployment_df: 部署计划 DataFrame
    """
    deployment_df['route_type'] = np.where(
        deployment_df['sending'] == deployment_df['receiving'],
        'self_loop',
        'cross_node'
    )
    
    cross_node = deployment_df[deployment_df['route_type'] == 'cross_node']
    if len(cross_node) == 0:
        print("  ⚠️  无跨节点路线数据")


def _ensure_date_format(deployment_df: pd.DataFrame) -> None:
    """
    确保日期字段格式正确。
    
    参数：
        deployment_df: 部署计划 DataFrame
    """
    if 'planned_deployment_date' in deployment_df.columns:
        deployment_df['planned_deployment_date'] = pd.to_datetime(
            deployment_df['planned_deployment_date']
        )


def _load_m6_configs(
    config_dict: Dict[str, Any],
    config: Dict[str, Any],
    validation_log: List[Dict]
) -> None:
    """
    加载 M6_ 开头的配置数据。
    
    参数：
        config_dict: 源配置字典
        config: 目标配置字典
        validation_log: 验证日志列表
    """
    for config_key, sheet_name in M6_CONFIG_MAPPING.items():
        if sheet_name in config_dict:
            config[config_key] = config_dict[sheet_name].copy()
            _normalize_config_columns(config, config_key, validation_log)
        else:
            _log_missing_config(sheet_name, validation_log)
            config[config_key] = pd.DataFrame()


def _normalize_config_columns(
    config: Dict[str, Any],
    config_key: str,
    validation_log: List[Dict]
) -> None:
    """
    标准化配置表的列名。
    
    参数：
        config: 配置字典
        config_key: 配置键名
        validation_log: 验证日志列表
    """
    route_configs = ['TruckReleaseCon', 'TruckCapacityPlan', 'DeliveryDelayDistribution']
    if config_key not in route_configs:
        return
    
    required_cols = ['sending', 'receiving']
    if config_key == 'DeliveryDelayDistribution':
        required_cols += ['delay_days', 'probability']
    
    df = config[config_key]
    for col in required_cols:
        if col not in df.columns:
            _find_and_map_column(df, col, config_key, validation_log)


def _find_and_map_column(
    df: pd.DataFrame,
    col: str,
    sheet_name: str,
    validation_log: List[Dict]
) -> None:
    """
    查找并映射列名变体。
    
    参数：
        df: DataFrame
        col: 目标列名
        sheet_name: 表名（用于日志）
        validation_log: 验证日志列表
    """
    variants = _get_column_variants(col)
    
    for variant in variants:
        if variant in df.columns:
            df[col] = df[variant]
            return
    
    validation_log.append({
        'sheet': sheet_name,
        'row': '',
        'issue': f'Missing required column: {col}. '
                 f'Available columns: {list(df.columns)}'
    })


def _get_column_variants(col: str) -> List[str]:
    """
    获取列名的可能变体。
    
    参数：
        col: 列名
        
    返回：
        可能的列名变体列表
    """
    variants = [
        col.upper(),
        col.capitalize(),
        col.replace('_', ' '),
        col.replace('_', ' ').title(),
        col.replace('_', ' ').capitalize()
    ]
    
    special_variants = {
        'sending': ['Sending', 'from', 'From'],
        'receiving': ['Receiving', 'to', 'To']
    }
    
    if col in special_variants:
        variants.extend(special_variants[col])
    
    return variants


def _load_global_configs(
    config_dict: Dict[str, Any],
    config: Dict[str, Any],
    validation_log: List[Dict]
) -> None:
    """
    加载 Global_ 开头的共享配置数据。
    
    参数：
        config_dict: 源配置字典
        config: 目标配置字典
        validation_log: 验证日志列表
    """
    for config_key, sheet_name in GLOBAL_CONFIG_MAPPING.items():
        if sheet_name in config_dict:
            config[config_key] = config_dict[sheet_name].copy()
            if config_key == 'LeadTime':
                _normalize_leadtime_columns(config[config_key], validation_log)
        else:
            _log_missing_config(sheet_name, validation_log, is_global=True)
            config[config_key] = pd.DataFrame()


def _normalize_leadtime_columns(
    df: pd.DataFrame,
    validation_log: List[Dict]
) -> None:
    """
    标准化 LeadTime 表的列名。
    
    参数：
        df: LeadTime DataFrame
        validation_log: 验证日志列表
    """
    required_cols = ['sending', 'receiving', 'PDT', 'GR']
    
    for col in required_cols:
        if col not in df.columns:
            _find_and_map_column(df, col, 'Global_LeadTime', validation_log)


def _log_missing_config(
    sheet_name: str,
    validation_log: List[Dict],
    is_global: bool = False
) -> None:
    """
    记录缺失的配置表。
    
    参数：
        sheet_name: 表名
        validation_log: 验证日志列表
        is_global: 是否为全局配置
    """
    config_type = 'global configuration' if is_global else 'configuration'
    validation_log.append({
        'sheet': sheet_name,
        'row': '',
        'issue': f'Missing required {config_type} sheet: {sheet_name}'
    })


def _process_date_fields(config: Dict[str, Any]) -> None:
    """
    处理配置中的日期字段。
    
    参数：
        config: 配置字典
    """
    date_fields = {
        'DeploymentPlan': ['planned_deployment_date'],
        'TruckCapacityPlan': ['date', 'eff_from', 'eff_to']
    }
    
    # 检查 DeliveryDelayDistribution 是否有 date 列
    delay_dist = config.get('DeliveryDelayDistribution', pd.DataFrame())
    if not delay_dist.empty and 'date' in delay_dist.columns:
        date_fields['DeliveryDelayDistribution'] = ['date']
    
    for sheet, fields in date_fields.items():
        if sheet not in config or config[sheet].empty:
            continue
        
        for field in fields:
            _convert_date_column(config[sheet], field)


def _convert_date_column(df: pd.DataFrame, field: str) -> None:
    """
    转换 DataFrame 中的日期列。
    
    参数：
        df: DataFrame
        field: 字段名
    """
    if field not in df.columns:
        return
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df[field] = pd.to_datetime(df[field], errors='coerce')


def _ensure_default_config(config: Dict[str, Any]) -> None:
    """
    确保配置中包含所有必需的空 DataFrame。
    
    参数：
        config: 配置字典
    """
    required_keys = [
        'DeploymentPlan', 'TruckReleaseCon', 'TruckCapacityPlan',
        'TruckTypeSpecs', 'MaterialMD', 'DeliveryDelayDistribution',
        'MDQBypassRules', 'DemandPriority', 'LeadTime'
    ]
    
    for key in required_keys:
        if key not in config:
            config[key] = pd.DataFrame()

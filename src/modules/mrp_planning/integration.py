"""
Module3 集成模式模块。

提供与Orchestrator集成运行的入口函数。
支持DuckDB内存模式，可跳过磁盘IO直接写入内存表。
"""

import time
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.date_safe import parse_mixed_datetime

from .config_loader import load_module1_daily_outputs
from .mrp_simulation import run_mrp_layered_simulation_daily
from .utils import normalize_identifiers

# 延迟导入内存存储模块（避免循环导入）
_memory_store_imported = False
_is_memory_mode_enabled = None
_write_module3_output = None


def _ensure_memory_store_imported():
    """延迟导入内存存储模块"""
    global _memory_store_imported, _is_memory_mode_enabled, _write_module3_output
    if not _memory_store_imported:
        try:
            from src.utils.memory_data_store import (
                is_memory_mode_enabled,
                write_module3_output,
            )
            _is_memory_mode_enabled = is_memory_mode_enabled
            _write_module3_output = write_module3_output
        except ImportError:
            _is_memory_mode_enabled = lambda: False
            _write_module3_output = lambda *args, **kwargs: False
        _memory_store_imported = True


def run_integrated_mode(
    module1_output_dir: str,
    orchestrator: object,
    config_dict: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    output_dir: str,
    skip_file_output: bool = False,
    module1_result: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, Any]:
    """
    Module3集成模式运行函数。

    参数：
        module1_output_dir: Module1输出目录
        orchestrator: Orchestrator实例
        config_dict: 配置数据字典
        start_date: 仿真开始日期
        end_date: 仿真结束日期
        output_dir: 输出目录
        skip_file_output: 是否跳过写入Excel文件
        module1_result: Module1内存数据

    返回：
        dict: 包含输出结果的字典
    """
    t_total = time.perf_counter()

    # 加载配置（来自main_integration的数据已被规范化，跳过重复规范化）
    configs = _load_static_configs(config_dict, skip_normalize=True)

    # 生成日期范围
    date_range = pd.date_range(start_date, end_date, freq='D')
    all_net_demand = []

    for current_date in date_range:
        daily_result = _process_single_day(
            current_date=current_date,
            module1_output_dir=module1_output_dir,
            module1_result=module1_result,
            orchestrator=orchestrator,
            configs=configs,
            output_dir=output_dir,
            skip_file_output=skip_file_output,
        )
        all_net_demand.extend(daily_result)

    # 汇总结果
    result = _build_result(all_net_demand, date_range)

    elapsed = time.perf_counter() - t_total

    return result


def _load_static_configs(config_dict: Dict[str, pd.DataFrame], skip_normalize: bool = False) -> dict:
    """加载并预处理静态配置。
    
    参数：
        config_dict: 配置字典
        skip_normalize: 是否跳过规范化（当config_dict来自main_integration时已被规范化）
    """
    safety_stock_df = config_dict.get('M3_SafetyStock', pd.DataFrame())
    network_df = config_dict.get('Global_Network', pd.DataFrame())
    lead_time_df = config_dict.get('Global_LeadTime', pd.DataFrame())
    m4_mlcfg_df = config_dict.get('M4_MaterialLocationLineCfg', pd.DataFrame())
    deploy_config_df = config_dict.get('M5_DeployConfig', pd.DataFrame())

    # 日期转换
    safety_stock_df = _convert_dates(safety_stock_df, 'M3_SafetyStock', 'date')
    network_df = _convert_dates(network_df, 'Global_Network', 'eff_from', 'eff_to')

    # 标识符规范化（来自main_integration的数据已被规范化，可跳过）
    if not skip_normalize:
        safety_stock_df = normalize_identifiers(safety_stock_df)
        network_df = normalize_identifiers(network_df)
        m4_mlcfg_df = normalize_identifiers(m4_mlcfg_df)
        deploy_config_df = normalize_identifiers(deploy_config_df)

    return {
        'safety_stock_df': safety_stock_df,
        'network_df': network_df,
        'lead_time_df': lead_time_df,
        'm4_mlcfg_df': m4_mlcfg_df,
        'deploy_config_df': deploy_config_df,
    }


def _convert_dates(df: pd.DataFrame, context_prefix: str, *cols) -> pd.DataFrame:
    """转换日期列。"""
    if df.empty:
        return df
    for col in cols:
        if col in df.columns:
            df[col] = parse_mixed_datetime(df[col], f"{context_prefix}.{col}")
    return df


def _process_single_day(
    current_date: pd.Timestamp,
    module1_output_dir: str,
    module1_result: Optional[Dict[str, pd.DataFrame]],
    orchestrator: object,
    configs: dict,
    output_dir: str,
    skip_file_output: bool,
) -> list:
    """处理单日数据。"""
    # 加载Module1数据
    m1_data = _load_module1_data(
        current_date, module1_output_dir, module1_result
    )

    # 加载Orchestrator数据
    orch_data = _load_orchestrator_data(orchestrator, current_date)

    # 计算净需求
    net_demand_df = _calculate_net_demand(
        current_date, m1_data, orch_data, configs
    )

    # 写入DuckDB内存表（如果内存模式已启用）
    _ensure_memory_store_imported()
    if _is_memory_mode_enabled and _is_memory_mode_enabled():
        date_str = current_date.strftime('%Y%m%d')
        _write_module3_output(date_str=date_str, net_demand=net_demand_df)

    # 保存Excel文件输出
    if not skip_file_output:
        _save_daily_output(net_demand_df, output_dir, current_date)

    return net_demand_df.to_dict('records') if not net_demand_df.empty else []


def _load_module1_data(
    current_date: pd.Timestamp,
    module1_output_dir: str,
    module1_result: Optional[Dict[str, pd.DataFrame]]
) -> dict:
    """加载Module1数据。"""
    try:
        if module1_result is not None:
            # 🔧 修复：使用累积订单(all_orders_for_next_day)而非仅当日订单(orders_df)
            # AO 缺口计算需要包含历史订单（与 Dev 基线版本一致）
            # Dev 基线版本的 `OrderLog` 包含所有历史生成但未来到期的订单
            order_df = module1_result.get('all_orders_for_next_day')
            if order_df is None or (hasattr(order_df, 'empty') and order_df.empty):
                # 回退到orders_df（兼容旧接口）
                order_df = module1_result.get('orders_df', pd.DataFrame())
            
            data = {
                'supply_demand_df': module1_result.get(
                    'supply_demand_df', pd.DataFrame()
                ),
                'shipment_df': module1_result.get(
                    'shipment_df', pd.DataFrame()
                ),
                'order_df': order_df,
            }
        else:
            data = load_module1_daily_outputs(module1_output_dir, current_date)
        
        # 规范化标识符，确保与其他数据源类型一致
        for key in data:
            data[key] = normalize_identifiers(data[key])
        
        return data
    except Exception as e:
        return {
            'supply_demand_df': pd.DataFrame(),
            'shipment_df': pd.DataFrame(),
            'order_df': pd.DataFrame(),
        }


def _load_orchestrator_data(
    orchestrator: object,
    current_date: pd.Timestamp
) -> dict:
    """加载Orchestrator数据。"""
    date_str = current_date.strftime('%Y-%m-%d')
    try:
        data = {
            'beginning_inventory_df': orchestrator.get_beginning_inventory_view(date_str),
            'in_transit_df': orchestrator.get_planning_intransit_view(date_str),
            'delivery_gr_df': orchestrator.get_delivery_gr_view(date_str),
            'all_production_df': orchestrator.get_all_production_view(date_str),
            'open_deployment_df': orchestrator.get_open_deployment_view(date_str),
            'delivery_shipment_df': orchestrator.get_delivery_shipment_log_view(date_str),
        }

        # 规范化标识符
        for key in data:
            data[key] = normalize_identifiers(data[key])

        return data

    except Exception as e:
        return {
            'beginning_inventory_df': pd.DataFrame(),
            'in_transit_df': pd.DataFrame(),
            'delivery_gr_df': pd.DataFrame(),
            'all_production_df': pd.DataFrame(),
            'open_deployment_df': pd.DataFrame(),
            'delivery_shipment_df': pd.DataFrame(),
        }


def _calculate_net_demand(
    current_date: pd.Timestamp,
    m1_data: dict,
    orch_data: dict,
    configs: dict
) -> pd.DataFrame:
    """计算净需求。"""
    try:
        return run_mrp_layered_simulation_daily(
            sim_date=current_date,
            daily_supply_demand_df=m1_data['supply_demand_df'],
            daily_order_df=m1_data['order_df'],
            daily_shipment_df=m1_data['shipment_df'],
            safety_stock_df=configs['safety_stock_df'],
            beginning_inventory_df=orch_data['beginning_inventory_df'],
            in_transit_df=orch_data['in_transit_df'],
            delivery_gr_df=orch_data['delivery_gr_df'],
            all_production_df=orch_data['all_production_df'],
            open_deployment_df=orch_data['open_deployment_df'],
            network_df=configs['network_df'],
            lead_time_df=configs['lead_time_df'],
            m4_mlcfg_df=configs['m4_mlcfg_df'],
            delivery_shipment_df=orch_data['delivery_shipment_df'],
            deploy_config_df=configs['deploy_config_df'],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def _save_daily_output(
    net_demand_df: pd.DataFrame,
    output_dir: str,
    current_date: pd.Timestamp
) -> None:
    """保存每日输出。"""
    expected_cols = [
        'material', 'location', 'requirement_date', 'quantity',
        'demand_element', 'layer', 'simulation_date', 'horizon_days'
    ]

    try:
        output_df = net_demand_df.copy() if not net_demand_df.empty else pd.DataFrame()

        if not output_df.empty:
            for col in expected_cols:
                if col not in output_df.columns:
                    output_df[col] = pd.Series(dtype='object')
            output_df = output_df[expected_cols]
        else:
            output_df = pd.DataFrame(columns=expected_cols)

        filepath = f"{output_dir}/Module3Output_{current_date.strftime('%Y%m%d')}.xlsx"
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='NetDemand')

    except Exception as e:
        pass


def _build_result(all_net_demand: list, date_range) -> dict:
    """构建返回结果。"""
    net_demand_df = pd.DataFrame(all_net_demand) if all_net_demand else pd.DataFrame()

    return {
        'net_demand_df': net_demand_df,
        'net_demand_count': len(all_net_demand),
        'processed_dates': len(date_range),
        'output_files': [
            f"Module3Output_{d.strftime('%Y%m%d')}.xlsx"
            for d in date_range
        ],
    }

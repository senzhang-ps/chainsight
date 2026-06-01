"""数据库配置装配辅助工具。

职责：
- 从数据库中的 `cfg_*` 表恢复标准配置字典。
- 维护数据库列名到原始 Excel 列名的映射关系。
- 在数据库模式下构造临时配置文件并复用标准仿真流程。
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from ..main_integration import run_integrated_simulation


def _load_config_from_database(db, config_name: str) -> dict:
    """从数据库加载配置表。

    P1 计划书 §6 优化：把 ``read_table(table_name)`` + pandas 过滤改为
    ``read_table(table_name, filters={'config_name': ...})``——DB 侧过滤命中
    ``cfg_*(config_name)`` 索引，避免整表读入内存，对大配置表显著降低 IO 与内存压力。

    `config_name` 兼容 ``BC/BC_S5`` 这种带路径前缀的输入：先用原值过滤，未命中再
    用 basename 重新查一次（两次查询都走索引）。
    """
    all_tables = db.get_all_tables()
    config_data = {}

    # 计算旧格式的配置前缀（如 bc_s5_）
    old_prefix = (
        config_name.lower().replace("-", "_").replace(" ", "_") + "_"
    )

    config_basename = (
        config_name.split('/')[-1] if '/' in config_name else config_name
    )

    # 优先尝试新格式（cfg_开头，通过 config_name 字段区分）
    for table_name in all_tables:
        if not table_name.startswith('cfg_'):
            continue

        # 通过 information_schema 缓存判断目标表是否带 config_name 列；
        # 没有则跳过——cfg_import_manifest 等管理表会落在这里。
        col_types = db._get_column_types(table_name)
        if 'config_name' not in col_types:
            continue

        try:
            # DB 侧过滤：精确匹配 config_name
            filtered = db.read_table(table_name, filters={'config_name': config_name})
        except Exception:
            continue

        # 兼容 ``BC/BC_S5``：精确无数据时回退到 basename
        if filtered.empty and config_basename != config_name:
            try:
                filtered = db.read_table(
                    table_name, filters={'config_name': config_basename}
                )
            except Exception:
                filtered = pd.DataFrame()

        # 清理 DB 元数据列和非标准列（unnamed_*、全 NULL 列等）
        drop_cols = [
            c for c in filtered.columns
            if c in ('config_name', 'config_type', 'db_write_time')
            or c.startswith('unnamed') or c.startswith('Unnamed')
        ]
        if drop_cols:
            filtered = filtered.drop(columns=drop_cols, errors='ignore')
        # 删除全为 NULL 的列（来自其他配置的表结构残留），但保留原有列结构
        cols_before = set(filtered.columns)
        filtered = filtered.dropna(axis=1, how='all')
        for col in cols_before - set(filtered.columns):
            filtered[col] = pd.NA

        # 去掉 cfg_ 前缀，作为配置数据的 key
        clean_table_name = table_name[4:]

        # 即使过滤后为空，也保留表结构（对于某些模块配置表是必要的）
        config_data[clean_table_name] = filtered

    # 如果新格式没有数据，回退到旧格式（兼容旧数据；旧格式没有 config_name 列）
    if not config_data:
        for table_name in all_tables:
            if not table_name.startswith(old_prefix):
                continue

            try:
                df = db.read_table(table_name)
            except Exception:
                continue

            if df.empty:
                continue

            # 去掉旧前缀，作为配置数据的 key
            clean_table_name = table_name[len(old_prefix):]
            config_data[clean_table_name] = df

    return config_data if config_data else None


def _get_column_mapping() -> dict:
    """获取列名映射（数据库小写 -> 原始大小写）"""
    # 定义所有实际使用的列名及其原始大小写形式
    original_columns = [
        # 通用列（小写）
        'material', 'location', 'sourcing', 'location_type',
        'quantity', 'date', 'week', 'day', 'seed',
        # 带下划线的列
        'eff_from', 'eff_to', 'demand_element', 'priority',
        'order_type', 'error_std_percent', 'order_day_flag',
        'advance_days', 'ao_percent', 'dps_location', 'dps_percent',
        'safety_stock_qty', 'key',
        # Global_LeadTime
        'sending', 'receiving', 'PDT', 'GR', 'MCT', 'OTD',
        # M4相关
        'delegate_line', 'prd_rate', 'min_batch', 'rv', 'ptf', 'lsk',
        'line', 'capacity', 'from_material', 'to_material',
        'changeover_id', 'from line', 'to line',
        'time', 'cost', 'mu_loss', 'pr',
        # M5相关
        'model', 'moq',
        # M6相关
        'truck_type', 'optimal_type', 'WFR', 'VFR', 'MDQ',
        'weight', 'volume', 'demand_unit_to_weight',
        'demand_unit_to_volume',
        'delay_days', 'probability', 'condition_logic', 'rule_id',
        'max_weight', 'max_volume', 'capacity_qty_in_weight',
        'capacity_qty_in_volume',
    ]
    
    # 创建小写到原始的映射
    mapping = {}
    for col in original_columns:
        mapping[col.lower().replace(' ', '_')] = col
    
    return mapping


def _run_simulation_with_db_config(
    config_data: dict,
    config_name: str,
    start_date: str,
    end_date: str,
    log_dir: Path,
    logger
) -> dict:
    """使用数据库配置运行仿真"""
    
    # 创建临时Excel文件
    temp_dir = Path(tempfile.mkdtemp())
    temp_config = temp_dir / f"{config_name}.xlsx"
    
    try:
        # 将配置数据写入临时Excel
        with pd.ExcelWriter(temp_config, engine='openpyxl') as writer:
            # 映射表名到原始sheet名
            sheet_mapping = {
                'sit_design': 'SIT Design',
                'global_seed': 'Global_seed',
                'config_guide': 'Config Guide',
                'global_network': 'Global_Network',
                'global_spacecapacity': 'Global_SpaceCapacity',
                'global_leadtime': 'Global_LeadTime',
                'global_demandpriority': 'Global_DemandPriority',
                'm1_initialinventory': 'M1_InitialInventory',
                'm1_initialinventory_30d': 'M1_InitialInventory_30D',
                'sheet1': 'Sheet1',
                'm1_demandforecast': 'M1_DemandForecast',
                'm1_forecasterror': 'M1_ForecastError',
                'm1_ordercalendar': 'M1_OrderCalendar',
                'm1_aoconfig': 'M1_AOConfig',
                'm1_dpsconfig': 'M1_DPSConfig',
                'm1_supplychoiceconfig': 'M1_SupplyChoiceConfig',
                'm3_safetystock': 'M3_SafetyStock',
                'covalidation': 'COValidation',
                'm4_materiallocationlinecfg': 'M4_MaterialLocationLineCfg',
                'm4_linecapacity': 'M4_LineCapacity',
                'm4_changeovermatrix': 'M4_ChangeoverMatrix',
                'm4_changeoverdefinition': 'M4_ChangeoverDefinition',
                'm4_productionreliability': 'M4_ProductionReliability',
                'm5_pushpullmodel': 'M5_PushPullModel',
                'm5_deployconfig': 'M5_DeployConfig',
                'm6_truckreleasecon': 'M6_TruckReleaseCon',
                'm6_materialmd': 'M6_MaterialMD',
                'm6_deliverydelaydistribution': 'M6_DeliveryDelayDistribution',
                'm6_mdqbypassrules': 'M6_MDQBypassRules',
                'm6_trucktypespecs': 'M6_TruckTypeSpecs',
                'm6_truckcapacityplan': 'M6_TruckCapacityPlan',
            }
            
            # 列名映射（数据库小写 -> 原始大小写）
            column_mapping = _get_column_mapping()
            
            for db_name, df in config_data.items():
                # 尝试映射到原始sheet名
                sheet_name = sheet_mapping.get(db_name.lower(), db_name)
                
                # 恢复列名大小写
                df_copy = df.copy()
                df_copy.columns = [
                    column_mapping.get(col.lower(), col)
                    for col in df_copy.columns
                ]
                
                df_copy.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"[LOG] 临时配置文件: {temp_config}")
        
        # 创建临时输出目录
        temp_output = temp_dir / "output"
        temp_output.mkdir(exist_ok=True)
        
        # 运行仿真
        result = run_integrated_simulation(
            config_path=str(temp_config),
            start_date=start_date,
            end_date=end_date,
            output_base_dir=str(temp_output),
            force_restart=True
        )
        
        if result and result.get('simulation_completed'):
            return {
                'success': True,
                'output_dir': str(temp_output)
            }
        else:
            return {'success': False}
            
    except Exception as e:
        logger.error(f"仿真执行出错: {e}")
        return {'success': False, 'error': str(e)}

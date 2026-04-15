"""
config_loader.py

配置加载与标准化模块。
"""

import os
from pathlib import Path

import pandas as pd

from ...utils.normalization import normalize_identifiers


_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


def discover_csv_override_files(excel_path: str) -> tuple[dict, list[str]]:
    """解析某个 Excel 配置文件允许使用的 CSV 覆盖文件。

    为避免一个目录下的通用命名 CSV（例如 ``M3_SafetyStock.csv``）误覆盖多个
    Excel 配置文件，重构版仅支持两种安全模式：

    1. 显式专属目录：
       - ``<excel_stem>_csv/``
       - ``<excel_stem>.csv_overrides/``
    2. 兼容旧行为，但仅在目录中只有这一个 Excel 文件时才启用同目录 ``*.csv``。

    Returns:
        tuple[dict, list[str]]:
            - ``{sheet_name: csv_path}``
            - 解析过程中的说明/提示信息
    """
    messages: list[str] = []
    csv_files: dict[str, Path] = {}

    config_file = Path(excel_path).resolve()
    config_dir = config_file.parent
    if not config_dir.is_dir():
        return csv_files, messages

    explicit_dirs = [
        config_dir / f"{config_file.stem}_csv",
        config_dir / f"{config_file.stem}.csv_overrides",
    ]
    explicit_found = False

    for override_dir in explicit_dirs:
        if not override_dir.is_dir():
            continue
        explicit_found = True
        for csv_path in sorted(override_dir.glob("*.csv")):
            sheet_name = csv_path.stem
            if sheet_name in csv_files:
                messages.append(
                    f"检测到重复 CSV 覆盖文件名 {sheet_name}.csv；已优先使用 {csv_files[sheet_name].parent.name}"
                )
                continue
            csv_files[sheet_name] = csv_path

    if explicit_found or csv_files:
        return csv_files, messages

    same_dir_csv_files = sorted(config_dir.glob("*.csv"))
    if not same_dir_csv_files:
        return csv_files, messages

    excel_siblings = sorted(
        p for p in config_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _EXCEL_SUFFIXES
    )

    if len(excel_siblings) == 1 and excel_siblings[0].resolve() == config_file:
        for csv_path in same_dir_csv_files:
            csv_files[csv_path.stem] = csv_path
        return csv_files, messages

    messages.append(
        "检测到当前目录存在多个 Excel 配置文件，已跳过目录级 CSV 覆盖。"
        f" 若需为 {config_file.name} 启用 CSV 覆盖，请将 CSV 放入专属目录 "
        f"{config_file.stem}_csv\\"
    )
    return csv_files, messages


def load_csv_overrides(excel_path: str, messages: list[str] | None = None) -> dict:
    """读取 Excel 配置文件对应的 CSV 覆盖数据。"""
    csv_overrides = {}
    try:
        csv_files, discovered_messages = discover_csv_override_files(excel_path)
        if messages is not None:
            messages.extend(discovered_messages)

        for sheet_name, csv_path in csv_files.items():
            try:
                csv_overrides[sheet_name] = pd.read_csv(csv_path)
            except Exception as e:
                error_message = f"CSV 文件读取失败: {csv_path.name} - {e}"
                if messages is not None:
                    messages.append(error_message)
                else:
                    pass
    except Exception as e:
        error_message = f"CSV 覆盖扫描失败: {e}"
        if messages is not None:
            messages.append(error_message)
        else:
            pass
    return csv_overrides


def load_configuration_from_dict(config_data: dict, config_name: str = "DB_Config") -> dict:
    """从DataFrame字典加载与标准化配置数据（用于数据库模式）

    目的：
    - 直接接收DataFrame字典，无需创建临时Excel文件
    - 执行与load_configuration基本一致的标准化、去重与键映射流程

    Args:
        config_data: 配置数据字典 {sheet_name: DataFrame}
        config_name: 配置名称（用于日志）

    Returns:
        dict: 标准化后的配置数据字典
    """
    
    # Sheet 名称映射（数据库小写表名 -> 集成流程沿用的工作表名/历史别名）
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
    column_mapping = {
        'material': 'material', 'location': 'location', 'sourcing': 'sourcing',
        'location_type': 'location_type', 'quantity': 'quantity', 'date': 'date',
        'week': 'week', 'day': 'day', 'seed': 'seed', 'eff_from': 'eff_from',
        'eff_to': 'eff_to', 'demand_element': 'demand_element', 'priority': 'priority',
        'order_type': 'order_type', 'error_std_percent': 'error_std_percent',
        'order_day_flag': 'order_day_flag', 'advance_days': 'advance_days',
        'ao_percent': 'ao_percent', 'dps_location': 'dps_location',
        'dps_percent': 'dps_percent', 'safety_stock_qty': 'safety_stock_qty',
        'key': 'key', 'sending': 'sending', 'receiving': 'receiving',
        'pdt': 'PDT', 'gr': 'GR', 'mct': 'MCT', 'otd': 'OTD',
        'delegate_line': 'delegate_line', 'prd_rate': 'prd_rate',
        'min_batch': 'min_batch', 'rv': 'rv', 'ptf': 'ptf', 'lsk': 'lsk',
        'line': 'line', 'capacity': 'capacity', 'from_material': 'from_material',
        'to_material': 'to_material', 'changeover_id': 'changeover_id',
        'from_line': 'from line', 'to_line': 'to line', 'time': 'time',
        'cost': 'cost', 'mu_loss': 'mu_loss', 'pr': 'pr', 'model': 'model',
        'moq': 'moq', 'truck_type': 'truck_type', 'optimal_type': 'optimal_type',
        'wfr': 'WFR', 'vfr': 'VFR', 'mdq': 'MDQ', 'weight': 'weight',
        'volume': 'volume', 'demand_unit_to_weight': 'demand_unit_to_weight',
        'demand_unit_to_volume': 'demand_unit_to_volume', 'delay_days': 'delay_days',
        'probability': 'probability', 'condition_logic': 'condition_logic',
        'rule_id': 'rule_id', 'max_weight': 'max_weight', 'max_volume': 'max_volume',
        'capacity_qty_in_weight': 'capacity_qty_in_weight',
        'capacity_qty_in_volume': 'capacity_qty_in_volume',
    }
    
    config_dict = {}
    
    # 转换配置数据
    for db_name, df in config_data.items():
        if not isinstance(df, pd.DataFrame):
            continue
        
        # 映射sheet名称
        sheet_name = sheet_mapping.get(db_name.lower(), db_name)
        
        # 恢复列名大小写
        df_copy = df.copy()
        df_copy.columns = [column_mapping.get(col.lower(), col) for col in df_copy.columns]
        
        # 清理残留的 DB 元数据列和非标准列（防止影响模块计算）
        drop_cols = [c for c in df_copy.columns 
                     if c in ('config_name', 'config_type', 'db_write_time')
                     or c.lower().startswith('unnamed')
                     or (not c.isascii() and c not in column_mapping.values())]
        if drop_cols:
            df_copy = df_copy.drop(columns=drop_cols, errors='ignore')
        # 删除全为 NULL 的列（来自其他配置的表结构残留），但保留原有列结构
        cols_before = set(df_copy.columns)
        df_copy = df_copy.dropna(axis=1, how='all')
        for col in cols_before - set(df_copy.columns):
            df_copy[col] = pd.NA

        config_dict[sheet_name] = df_copy
    
    # 确保必要的配置表存在
    required_sheets = [
        'M1_InitialInventory',
        'Global_SpaceCapacity',
        'Global_Network',
        'Global_LeadTime',
        'Global_DemandPriority'
    ]
    
    missing_sheets = [sheet for sheet in required_sheets if sheet not in config_dict]
    if missing_sheets:
        for sheet in missing_sheets:
            config_dict[sheet] = pd.DataFrame()
    
    # 统一标准化所有配置表的标识符字段
    standardized_count = 0
    for sheet_name, df in config_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 
                             'dps_location', 'from_material', 'to_material', 'line', 
                             'delegate_line', 'changeover_id']
            has_identifiers = any(col in df.columns for col in identifier_cols)
            
            if has_identifiers:
                original_dtypes = {col: str(df[col].dtype) for col in identifier_cols if col in df.columns}
                config_dict[sheet_name] = normalize_identifiers(df)
                new_dtypes = {col: str(config_dict[sheet_name][col].dtype) for col in identifier_cols if col in config_dict[sheet_name].columns}
                
                normalized_fields = []
                for col in identifier_cols:
                    if col in df.columns and original_dtypes[col] != new_dtypes[col]:
                        normalized_fields.append(f"{col}({original_dtypes[col]}→{new_dtypes[col]})")
                
                if normalized_fields:
                    standardized_count += 1
    
    if standardized_count > 0:
    
        pass
    # Changeover 配置校验和去重
    if 'M4_ChangeoverMatrix' in config_dict and not config_dict['M4_ChangeoverMatrix'].empty:
        co_matrix = config_dict['M4_ChangeoverMatrix']
        duplicates = co_matrix[co_matrix.duplicated(subset=['from_material', 'to_material'], keep=False)]
        if not duplicates.empty:
            original_count = len(co_matrix)
            config_dict['M4_ChangeoverMatrix'] = co_matrix.drop_duplicates(
                subset=['from_material', 'to_material'], keep='first'
            )
        else:
    
            pass
    # ChangeoverDefinition 配置校验和去重
    if 'M4_ChangeoverDefinition' in config_dict and not config_dict['M4_ChangeoverDefinition'].empty:
        co_def = config_dict['M4_ChangeoverDefinition']
        duplicates = co_def[co_def.duplicated(subset=['changeover_id', 'line'], keep=False)]
        if not duplicates.empty:
            original_count = len(co_def)
            config_dict['M4_ChangeoverDefinition'] = co_def.drop_duplicates(
                subset=['changeover_id', 'line'], keep='first'
            )
        else:
    
            pass
    # Module4 配置表映射
    module4_mappings = {
        'M4_MaterialLocationLineCfg': 'MaterialLocationLineCfg',
        'M4_LineCapacity': 'LineCapacity',
        'M4_ChangeoverMatrix': 'ChangeoverMatrix',
        'M4_ChangeoverDefinition': 'ChangeoverDefinition',
        'M4_ProductionReliability': 'ProductionReliability'
    }
    
    mapped_count = 0
    for original_key, mapped_key in module4_mappings.items():
        if original_key in config_dict and not config_dict[original_key].empty:
            config_dict[mapped_key] = config_dict[original_key]
            mapped_count += 1
    
    if mapped_count > 0:
    
        pass
    return config_dict


def load_configuration(config_path: str) -> dict:
    """加载与标准化配置数据

    目的：
    - 从 Excel 读取所有工作表，补齐缺失的必要表，统一标准化标识符字段，并对 M4 换产配置执行重复性检查与去重映射。

    Args:
        config_path: 配置文件路径（Excel）。

    Returns:
        dict: 标准化后的配置数据字典。

    输入数据：
        - Excel 工作簿；可能存在缺失表或非标准类型的标识符列。

    输出/副作用：
        - 打印加载与标准化日志；对 M4 的配置进行去重与键映射以向后兼容。

    逻辑：
        - 加载→补齐必要表→标准化标识符→检验并去重 Changeover 配置→映射关键表→返回字典。
    """

    try:
        xl = pd.ExcelFile(config_path)
        config_dict = {}

        # 加载所有配置表
        for sheet_name in xl.sheet_names:
            config_dict[sheet_name] = xl.parse(sheet_name)

        # 汇总（CSV 覆盖机制已禁用，与 Dev 版本保持一致：仅从 Excel 加载）

        # 确保必要的配置表存在
        required_sheets = [
            'M1_InitialInventory',
            'Global_SpaceCapacity',
            'Global_Network',
            'Global_LeadTime',
            'Global_DemandPriority'
        ]
        
        missing_sheets = [sheet for sheet in required_sheets if sheet not in config_dict]
        if missing_sheets:
            # 创建空的配置表
            for sheet in missing_sheets:
                config_dict[sheet] = pd.DataFrame()
        
        # 统一标准化所有配置表的标识符字段
        standardized_count = 0
        for sheet_name, df in config_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 检查是否包含标识符字段
                identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location', 'from_material', 'to_material', 'line', 'delegate_line', 'changeover_id']
                has_identifiers = any(col in df.columns for col in identifier_cols)
                
                if has_identifiers:
                    original_dtypes = {col: str(df[col].dtype) for col in identifier_cols if col in df.columns}
                    config_dict[sheet_name] = normalize_identifiers(df)
                    new_dtypes = {col: str(config_dict[sheet_name][col].dtype) for col in identifier_cols if col in config_dict[sheet_name].columns}
                    
                    # 记录标准化的字段
                    normalized_fields = []
                    for col in identifier_cols:
                        if col in df.columns and original_dtypes[col] != new_dtypes[col]:
                            normalized_fields.append(f"{col}({original_dtypes[col]}→{new_dtypes[col]})")
                    
                    if normalized_fields:
                        standardized_count += 1
        
        if standardized_count > 0:
            pass
        else:
        
            pass
        # 🔧 Changeover 配置校验和去重
        if 'M4_ChangeoverMatrix' in config_dict and not config_dict['M4_ChangeoverMatrix'].empty:
            co_matrix = config_dict['M4_ChangeoverMatrix']
            
            # 检查重复定义
            duplicates = co_matrix[co_matrix.duplicated(subset=['from_material', 'to_material'], keep=False)]
            if not duplicates.empty:
                
                # 详细检查每组重复
                for (from_mat, to_mat), group in duplicates.groupby(['from_material', 'to_material']):
                    unique_coids = group['changeover_id'].unique()
                    if len(unique_coids) > 1:
                        pass
                    else:
                        pass
                
                # 去重（保留第一条）
                original_count = len(co_matrix)
                config_dict['M4_ChangeoverMatrix'] = co_matrix.drop_duplicates(
                    subset=['from_material', 'to_material'], keep='first'
                )
                removed_count = original_count - len(config_dict['M4_ChangeoverMatrix'])
            else:
        
                pass
        # 🔧 ChangeoverDefinition 配置校验和去重
        if 'M4_ChangeoverDefinition' in config_dict and not config_dict['M4_ChangeoverDefinition'].empty:
            co_def = config_dict['M4_ChangeoverDefinition']
            
            # 检查重复定义
            duplicates = co_def[co_def.duplicated(subset=['changeover_id', 'line'], keep=False)]
            if not duplicates.empty:
                
                # 详细检查每组重复
                for (coid, line), group in duplicates.groupby(['changeover_id', 'line']):
                    unique_times = group['time'].unique()
                    if len(unique_times) > 1:
                        pass
                    else:
                        pass
                
                # 去重（保留第一条）
                original_count = len(co_def)
                config_dict['M4_ChangeoverDefinition'] = co_def.drop_duplicates(
                    subset=['changeover_id', 'line'], keep='first'
                )
                removed_count = original_count - len(config_dict['M4_ChangeoverDefinition'])
            else:
        
                pass
        # Module4 配置表映射（为了向后兼容）
        module4_mappings = {
            'M4_MaterialLocationLineCfg': 'MaterialLocationLineCfg',
            'M4_LineCapacity': 'LineCapacity',
            'M4_ChangeoverMatrix': 'ChangeoverMatrix',
            'M4_ChangeoverDefinition': 'ChangeoverDefinition',
            'M4_ProductionReliability': 'ProductionReliability'
        }

        mapped_count = 0
        for original_key, mapped_key in module4_mappings.items():
            if original_key in config_dict and not config_dict[original_key].empty:
                config_dict[mapped_key] = config_dict[original_key]
                mapped_count += 1

        if mapped_count > 0:
            pass
        else:
        
            pass
        return config_dict
        
    except Exception as e:
        raise

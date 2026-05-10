"""
config_loader.py

配置加载与标准化模块。
"""

import logging
import os
from pathlib import Path

import pandas as pd

from ...utils.normalization import normalize_identifiers

# 复用 src/utils/logger_config.py::DualLogger 创建的同名 logger，
# 这样消息既能进控制台又能进 simulation_log_*.txt。
logger = logging.getLogger("SupplyChainSimulation")


_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


def discover_csv_override_files(excel_path: str) -> tuple[dict, list[str]]:
    """解析某个 Excel 配置文件允许使用的 CSV 覆盖文件。

    支持两种来源（合并；显式子目录优先）：

    1. 显式专属目录：
       - ``<excel_stem>_csv/``
       - ``<excel_stem>.csv_overrides/``
    2. 同目录 ``*.csv``（不再要求目录中只有一个 Excel 文件）。

    覆盖语义由调用方决定（当前 ``load_configuration`` 仅在 Excel 内对应 sheet
    为空时才使用 CSV，所以多 Excel 共享同名 CSV 不会引发误覆盖）。

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

    for override_dir in explicit_dirs:
        if not override_dir.is_dir():
            continue
        for csv_path in sorted(override_dir.glob("*.csv")):
            sheet_name = csv_path.stem
            if sheet_name in csv_files:
                messages.append(
                    f"检测到重复 CSV 覆盖文件名 {sheet_name}.csv；已优先使用 {csv_files[sheet_name].parent.name}"
                )
                continue
            csv_files[sheet_name] = csv_path

    # 同目录 *.csv：作为补充来源，不覆盖显式子目录中的同名 CSV
    for csv_path in sorted(config_dir.glob("*.csv")):
        sheet_name = csv_path.stem
        if sheet_name not in csv_files:
            csv_files[sheet_name] = csv_path

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
                    logger.warning(f"  ⚠️ {error_message}")
    except Exception as e:
        error_message = f"CSV 覆盖扫描失败: {e}"
        if messages is not None:
            messages.append(error_message)
        else:
            logger.warning(f"  ⚠️ {error_message}")
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
    logger.info(f"📋 处理配置数据: {config_name} (共 {len(config_data)} 个表)")

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
        logger.info(f"  ✅ 加载配置表: {sheet_name} ({len(df_copy)} 行)")
    
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
        logger.warning(f"⚠️  缺少必要配置表: {missing_sheets}")
        for sheet in missing_sheets:
            config_dict[sheet] = pd.DataFrame()
    
    # 统一标准化所有配置表的标识符字段
    logger.info("🔧 正在标准化标识符字段...")
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
                    logger.info(f"  🔧 {sheet_name}: {', '.join(normalized_fields)}")
                    standardized_count += 1
    
    if standardized_count > 0:
        logger.info(f"✅ 已标准化 {standardized_count} 个配置表的标识符字段")

    # Changeover 配置校验和去重
    if 'M4_ChangeoverMatrix' in config_dict and not config_dict['M4_ChangeoverMatrix'].empty:
        logger.info("\n🔧 校验 Changeover Matrix 配置...")
        co_matrix = config_dict['M4_ChangeoverMatrix']
        duplicates = co_matrix[co_matrix.duplicated(subset=['from_material', 'to_material'], keep=False)]
        if not duplicates.empty:
            original_count = len(co_matrix)
            config_dict['M4_ChangeoverMatrix'] = co_matrix.drop_duplicates(
                subset=['from_material', 'to_material'], keep='first'
            )
            logger.info(f"  🔧 已去除 {original_count - len(config_dict['M4_ChangeoverMatrix'])} 条重复记录")
        else:
            logger.info("  ✅ Changeover Matrix 无重复定义")

    # ChangeoverDefinition 配置校验和去重
    if 'M4_ChangeoverDefinition' in config_dict and not config_dict['M4_ChangeoverDefinition'].empty:
        logger.info("\n🔧 校验 Changeover Definition 配置...")
        co_def = config_dict['M4_ChangeoverDefinition']
        duplicates = co_def[co_def.duplicated(subset=['changeover_id', 'line'], keep=False)]
        if not duplicates.empty:
            original_count = len(co_def)
            config_dict['M4_ChangeoverDefinition'] = co_def.drop_duplicates(
                subset=['changeover_id', 'line'], keep='first'
            )
            logger.info(f"  🔧 已去除 {original_count - len(config_dict['M4_ChangeoverDefinition'])} 条重复记录")
        else:
            logger.info("  ✅ Changeover Definition 无重复定义")

    # Module4 配置表映射
    logger.info("\n🔧 正在映射 Module4 配置表...")
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
            logger.info(f"  🔧 映射 {original_key} → {mapped_key}")
            mapped_count += 1
    
    if mapped_count > 0:
        logger.info(f"✅ 已映射 {mapped_count} 个 Module4 配置表")

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
    logger.info(f"📋 加载配置文件: {config_path}")

    try:
        xl = pd.ExcelFile(config_path)
        config_dict = {}

        # 加载所有配置表
        for sheet_name in xl.sheet_names:
            config_dict[sheet_name] = xl.parse(sheet_name)
            logger.info(f"  ✅ [Excel] {sheet_name} ({len(config_dict[sheet_name])} 行)")

        # CSV 覆盖：仅当 Excel 中对应 sheet 为空时，才使用同名 CSV 数据
        csv_messages: list[str] = []
        csv_overrides = load_csv_overrides(config_path, csv_messages)
        for msg in csv_messages:
            logger.info(f"  ℹ️ {msg}")
        applied_csv_count = 0
        for sheet_name, csv_df in csv_overrides.items():
            existing = config_dict.get(sheet_name)
            if existing is None or (isinstance(existing, pd.DataFrame) and existing.empty):
                config_dict[sheet_name] = csv_df
                applied_csv_count += 1
                logger.info(f"  ✅ [CSV] {sheet_name} ({len(csv_df)} 行) — Excel 中该 sheet 为空，使用 CSV 数据")

        logger.info(
            f"📊 配置加载汇总: 共 {len(config_dict)} 个配置表 "
            f"(Excel: {len(xl.sheet_names)}, CSV 覆盖空 sheet: {applied_csv_count})"
        )

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
            logger.warning(f"⚠️  缺少必要配置表: {missing_sheets}")
            # 创建空的配置表
            for sheet in missing_sheets:
                config_dict[sheet] = pd.DataFrame()
        
        # 统一标准化所有配置表的标识符字段
        logger.info("🔧 正在标准化标识符字段...")
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
                        logger.info(f"  🔧 {sheet_name}: {', '.join(normalized_fields)}")
                        standardized_count += 1
        
        if standardized_count > 0:
            logger.info(f"✅ 已标准化 {standardized_count} 个配置表的标识符字段")
        else:
            logger.info("✅ 所有配置表的标识符字段已是标准格式")

        # 🔧 Changeover 配置校验和去重
        if 'M4_ChangeoverMatrix' in config_dict and not config_dict['M4_ChangeoverMatrix'].empty:
            logger.info("\n🔧 校验 Changeover Matrix 配置...")
            co_matrix = config_dict['M4_ChangeoverMatrix']
            
            # 检查重复定义
            duplicates = co_matrix[co_matrix.duplicated(subset=['from_material', 'to_material'], keep=False)]
            if not duplicates.empty:
                logger.warning(f"  ⚠️  发现 {len(duplicates)} 条重复的 changeover matrix 定义")

                # 详细检查每组重复
                for (from_mat, to_mat), group in duplicates.groupby(['from_material', 'to_material']):
                    unique_coids = group['changeover_id'].unique()
                    if len(unique_coids) > 1:
                        # 不同的 changeover_id - 严重错误
                        logger.error(f"    ❌ ERROR: {from_mat} → {to_mat} 有 {len(unique_coids)} 个不同的 changeover_id: {list(unique_coids)}")
                    else:
                        # 相同的 changeover_id - 只是重复
                        logger.warning(f"    ⚠️  {from_mat} → {to_mat} 有 {len(group)} 条重复记录 (changeover_id={unique_coids[0]})")

                # 去重（保留第一条）
                original_count = len(co_matrix)
                config_dict['M4_ChangeoverMatrix'] = co_matrix.drop_duplicates(
                    subset=['from_material', 'to_material'], keep='first'
                )
                removed_count = original_count - len(config_dict['M4_ChangeoverMatrix'])
                logger.info(f"  🔧 已去除 {removed_count} 条重复记录")
            else:
                logger.info("  ✅ Changeover Matrix 无重复定义")

        # 🔧 ChangeoverDefinition 配置校验和去重
        if 'M4_ChangeoverDefinition' in config_dict and not config_dict['M4_ChangeoverDefinition'].empty:
            logger.info("\n🔧 校验 Changeover Definition 配置...")
            co_def = config_dict['M4_ChangeoverDefinition']
            
            # 检查重复定义
            duplicates = co_def[co_def.duplicated(subset=['changeover_id', 'line'], keep=False)]
            if not duplicates.empty:
                logger.warning(f"  ⚠️  发现 {len(duplicates)} 条重复的 changeover definition 定义")

                # 详细检查每组重复
                for (coid, line), group in duplicates.groupby(['changeover_id', 'line']):
                    unique_times = group['time'].unique()
                    if len(unique_times) > 1:
                        # 不同的 time - 严重错误
                        logger.error(f"    ❌ ERROR: changeover_id={coid}, line={line} 有 {len(unique_times)} 个不同的 time 值: {list(unique_times)}")
                    else:
                        # 相同的参数 - 只是重复
                        logger.warning(f"    ⚠️  changeover_id={coid}, line={line} 有 {len(group)} 条重复记录 (time={unique_times[0]})")

                # 去重（保留第一条）
                original_count = len(co_def)
                config_dict['M4_ChangeoverDefinition'] = co_def.drop_duplicates(
                    subset=['changeover_id', 'line'], keep='first'
                )
                removed_count = original_count - len(config_dict['M4_ChangeoverDefinition'])
                logger.info(f"  🔧 已去除 {removed_count} 条重复记录")
            else:
                logger.info("  ✅ Changeover Definition 无重复定义")

        # Module4 配置表映射（为了向后兼容）
        logger.info("\n🔧 正在映射 Module4 配置表...")
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
                logger.info(f"  🔧 映射 {original_key} → {mapped_key}")
                mapped_count += 1

        if mapped_count > 0:
            logger.info(f"✅ 已映射 {mapped_count} 个 Module4 配置表")
        else:
            logger.info("✅ 无需映射 Module4 配置表")

        return config_dict
        
    except Exception as e:
        logger.error(f"❌ 配置文件加载失败: {e}")
        raise

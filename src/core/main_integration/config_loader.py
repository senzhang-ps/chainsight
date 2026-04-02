"""
config_loader.py

配置加载与标准化模块。
"""

import os

import pandas as pd

from .normalize import _normalize_identifiers


def load_csv_overrides(excel_path: str) -> dict:
    """扫描 Excel 配置文件所在目录中的 CSV 文件，作为配置表覆盖。

    当某些配置表行数超过 Excel 行数上限（~100万行）时，允许用户将该表导出为 CSV
    放在 Excel 同目录下。CSV 文件名（不含扩展名）即为对应的 Excel 工作表名称。

    Args:
        excel_path: Excel 配置文件的完整路径。

    Returns:
        dict: {sheet_name: DataFrame}，所有找到的 CSV 覆盖数据。
              如果目录中无 CSV 文件或路径无效，返回空字典。
    """
    csv_overrides = {}
    try:
        config_dir = os.path.dirname(os.path.abspath(excel_path))
        if not os.path.isdir(config_dir):
            return csv_overrides

        for filename in sorted(os.listdir(config_dir)):
            if not filename.lower().endswith('.csv'):
                continue
            sheet_name = os.path.splitext(filename)[0]
            csv_path = os.path.join(config_dir, filename)
            try:
                df = pd.read_csv(csv_path)
                csv_overrides[sheet_name] = df
            except Exception as e:
                print(f"  ⚠️ CSV 文件读取失败: {filename} - {e}")
    except Exception as e:
        print(f"  ⚠️ CSV 覆盖扫描失败: {e}")
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
    print(f"📋 处理配置数据: {config_name} (共 {len(config_data)} 个表)")
    
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
        print(f"  ✅ 加载配置表: {sheet_name} ({len(df_copy)} 行)")
    
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
        print(f"⚠️  缺少必要配置表: {missing_sheets}")
        for sheet in missing_sheets:
            config_dict[sheet] = pd.DataFrame()
    
    # 统一标准化所有配置表的标识符字段
    print(f"🔧 正在标准化标识符字段...")
    standardized_count = 0
    for sheet_name, df in config_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 
                             'dps_location', 'from_material', 'to_material', 'line', 
                             'delegate_line', 'changeover_id']
            has_identifiers = any(col in df.columns for col in identifier_cols)
            
            if has_identifiers:
                original_dtypes = {col: str(df[col].dtype) for col in identifier_cols if col in df.columns}
                config_dict[sheet_name] = _normalize_identifiers(df)
                new_dtypes = {col: str(config_dict[sheet_name][col].dtype) for col in identifier_cols if col in config_dict[sheet_name].columns}
                
                normalized_fields = []
                for col in identifier_cols:
                    if col in df.columns and original_dtypes[col] != new_dtypes[col]:
                        normalized_fields.append(f"{col}({original_dtypes[col]}→{new_dtypes[col]})")
                
                if normalized_fields:
                    print(f"  🔧 {sheet_name}: {', '.join(normalized_fields)}")
                    standardized_count += 1
    
    if standardized_count > 0:
        print(f"✅ 已标准化 {standardized_count} 个配置表的标识符字段")
    
    # Changeover 配置校验和去重
    if 'M4_ChangeoverMatrix' in config_dict and not config_dict['M4_ChangeoverMatrix'].empty:
        print(f"\n🔧 校验 Changeover Matrix 配置...")
        co_matrix = config_dict['M4_ChangeoverMatrix']
        duplicates = co_matrix[co_matrix.duplicated(subset=['from_material', 'to_material'], keep=False)]
        if not duplicates.empty:
            original_count = len(co_matrix)
            config_dict['M4_ChangeoverMatrix'] = co_matrix.drop_duplicates(
                subset=['from_material', 'to_material'], keep='first'
            )
            print(f"  🔧 已去除 {original_count - len(config_dict['M4_ChangeoverMatrix'])} 条重复记录")
        else:
            print(f"  ✅ Changeover Matrix 无重复定义")
    
    # ChangeoverDefinition 配置校验和去重
    if 'M4_ChangeoverDefinition' in config_dict and not config_dict['M4_ChangeoverDefinition'].empty:
        print(f"\n🔧 校验 Changeover Definition 配置...")
        co_def = config_dict['M4_ChangeoverDefinition']
        duplicates = co_def[co_def.duplicated(subset=['changeover_id', 'line'], keep=False)]
        if not duplicates.empty:
            original_count = len(co_def)
            config_dict['M4_ChangeoverDefinition'] = co_def.drop_duplicates(
                subset=['changeover_id', 'line'], keep='first'
            )
            print(f"  🔧 已去除 {original_count - len(config_dict['M4_ChangeoverDefinition'])} 条重复记录")
        else:
            print(f"  ✅ Changeover Definition 无重复定义")
    
    # Module4 配置表映射
    print(f"\n🔧 正在映射 Module4 配置表...")
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
            print(f"  🔧 映射 {original_key} → {mapped_key}")
            mapped_count += 1
    
    if mapped_count > 0:
        print(f"✅ 已映射 {mapped_count} 个 Module4 配置表")
    
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
    print(f"📋 加载配置文件: {config_path}")

    try:
        xl = pd.ExcelFile(config_path)
        config_dict = {}

        # 加载所有配置表
        for sheet_name in xl.sheet_names:
            config_dict[sheet_name] = xl.parse(sheet_name)
            print(f"  ✅ [Excel] {sheet_name} ({len(config_dict[sheet_name])} 行)")

        # 扫描并应用 CSV 覆盖（同目录下的 CSV 文件优先于 Excel 工作表）
        csv_overrides = load_csv_overrides(config_path)
        csv_override_count = 0
        csv_new_count = 0
        if csv_overrides:
            print(f"\n  {'─' * 50}")
            print(f"  📄 发现 {len(csv_overrides)} 个 CSV 覆盖文件:")
            for sheet_name, df in csv_overrides.items():
                if sheet_name in config_dict:
                    csv_override_count += 1
                    config_dict[sheet_name] = df
                    print(f"  🔄 [CSV 覆盖] {sheet_name} ({len(df)} 行) ← 替代Excel版本")
                else:
                    csv_new_count += 1
                    config_dict[sheet_name] = df
                    print(f"  ➕ [CSV 新增] {sheet_name} ({len(df)} 行)")
            print(f"  {'─' * 50}")

        # 汇总
        excel_count = len(xl.sheet_names) - csv_override_count
        total_count = excel_count + csv_override_count + csv_new_count
        summary_parts = [f"Excel: {excel_count}"]
        if csv_override_count:
            summary_parts.append(f"CSV覆盖: {csv_override_count}")
        if csv_new_count:
            summary_parts.append(f"CSV新增: {csv_new_count}")
        print(f"📊 配置加载汇总: 共 {total_count} 个配置表 ({', '.join(summary_parts)})")

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
            print(f"⚠️  缺少必要配置表: {missing_sheets}")
            # 创建空的配置表
            for sheet in missing_sheets:
                config_dict[sheet] = pd.DataFrame()
        
        # 统一标准化所有配置表的标识符字段
        print(f"🔧 正在标准化标识符字段...")
        standardized_count = 0
        for sheet_name, df in config_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 检查是否包含标识符字段
                identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location', 'from_material', 'to_material', 'line', 'delegate_line', 'changeover_id']
                has_identifiers = any(col in df.columns for col in identifier_cols)
                
                if has_identifiers:
                    original_dtypes = {col: str(df[col].dtype) for col in identifier_cols if col in df.columns}
                    config_dict[sheet_name] = _normalize_identifiers(df)
                    new_dtypes = {col: str(config_dict[sheet_name][col].dtype) for col in identifier_cols if col in config_dict[sheet_name].columns}
                    
                    # 记录标准化的字段
                    normalized_fields = []
                    for col in identifier_cols:
                        if col in df.columns and original_dtypes[col] != new_dtypes[col]:
                            normalized_fields.append(f"{col}({original_dtypes[col]}→{new_dtypes[col]})")
                    
                    if normalized_fields:
                        print(f"  🔧 {sheet_name}: {', '.join(normalized_fields)}")
                        standardized_count += 1
        
        if standardized_count > 0:
            print(f"✅ 已标准化 {standardized_count} 个配置表的标识符字段")
        else:
            print(f"✅ 所有配置表的标识符字段已是标准格式")
        
        # 🔧 Changeover 配置校验和去重
        if 'M4_ChangeoverMatrix' in config_dict and not config_dict['M4_ChangeoverMatrix'].empty:
            print(f"\n🔧 校验 Changeover Matrix 配置...")
            co_matrix = config_dict['M4_ChangeoverMatrix']
            
            # 检查重复定义
            duplicates = co_matrix[co_matrix.duplicated(subset=['from_material', 'to_material'], keep=False)]
            if not duplicates.empty:
                print(f"  ⚠️  发现 {len(duplicates)} 条重复的 changeover matrix 定义")
                
                # 详细检查每组重复
                for (from_mat, to_mat), group in duplicates.groupby(['from_material', 'to_material']):
                    unique_coids = group['changeover_id'].unique()
                    if len(unique_coids) > 1:
                        # 不同的 changeover_id - 严重错误
                        print(f"    ❌ ERROR: {from_mat} → {to_mat} 有 {len(unique_coids)} 个不同的 changeover_id: {list(unique_coids)}")
                    else:
                        # 相同的 changeover_id - 只是重复
                        print(f"    ⚠️  {from_mat} → {to_mat} 有 {len(group)} 条重复记录 (changeover_id={unique_coids[0]})")
                
                # 去重（保留第一条）
                original_count = len(co_matrix)
                config_dict['M4_ChangeoverMatrix'] = co_matrix.drop_duplicates(
                    subset=['from_material', 'to_material'], keep='first'
                )
                removed_count = original_count - len(config_dict['M4_ChangeoverMatrix'])
                print(f"  🔧 已去除 {removed_count} 条重复记录")
            else:
                print(f"  ✅ Changeover Matrix 无重复定义")
        
        # 🔧 ChangeoverDefinition 配置校验和去重
        if 'M4_ChangeoverDefinition' in config_dict and not config_dict['M4_ChangeoverDefinition'].empty:
            print(f"\n🔧 校验 Changeover Definition 配置...")
            co_def = config_dict['M4_ChangeoverDefinition']
            
            # 检查重复定义
            duplicates = co_def[co_def.duplicated(subset=['changeover_id', 'line'], keep=False)]
            if not duplicates.empty:
                print(f"  ⚠️  发现 {len(duplicates)} 条重复的 changeover definition 定义")
                
                # 详细检查每组重复
                for (coid, line), group in duplicates.groupby(['changeover_id', 'line']):
                    unique_times = group['time'].unique()
                    if len(unique_times) > 1:
                        # 不同的 time - 严重错误
                        print(f"    ❌ ERROR: changeover_id={coid}, line={line} 有 {len(unique_times)} 个不同的 time 值: {list(unique_times)}")
                    else:
                        # 相同的参数 - 只是重复
                        print(f"    ⚠️  changeover_id={coid}, line={line} 有 {len(group)} 条重复记录 (time={unique_times[0]})")
                
                # 去重（保留第一条）
                original_count = len(co_def)
                config_dict['M4_ChangeoverDefinition'] = co_def.drop_duplicates(
                    subset=['changeover_id', 'line'], keep='first'
                )
                removed_count = original_count - len(config_dict['M4_ChangeoverDefinition'])
                print(f"  🔧 已去除 {removed_count} 条重复记录")
            else:
                print(f"  ✅ Changeover Definition 无重复定义")
        
        # Module4 配置表映射（为了向后兼容）
        print(f"\n🔧 正在映射 Module4 配置表...")
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
                print(f"  🔧 映射 {original_key} → {mapped_key}")
                mapped_count += 1

        if mapped_count > 0:
            print(f"✅ 已映射 {mapped_count} 个 Module4 配置表")
        else:
            print(f"✅ 无需映射 Module4 配置表")
        
        return config_dict
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        raise

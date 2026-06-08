"""
config_loader.py

配置加载与标准化模块。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...utils.normalization import normalize_identifiers

if TYPE_CHECKING:
    # 仅类型检查使用，运行期不导入，避免 main_integration ↔ run 循环依赖。
    from ..run.config_dir import ConfigDir

# 复用 src/utils/logger_config.py::DualLogger 创建的同名 logger，
# 这样消息既能进控制台又能进 simulation_log_*.txt。
logger = logging.getLogger("SupplyChainSimulation")


_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


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

    _validate_config_dict(config_dict)
    return config_dict


def _align_csv_dtypes_to_excel(
    csv_df: pd.DataFrame,
    xl: pd.ExcelFile,
    sheet_name: str,
) -> list[str]:
    """以 Excel sheet 的列 dtype 为权威，把 CSV 同名列向其对齐（in-place）。

    背景：纯文本 CSV 读出 ``date``/时间戳列默认是 ``str``；同源 Excel 因 cell
    格式落成 ``datetime64``。两条路径若不对齐，下游 ``df['date'] == Timestamp``
    会因 dtype 不匹配静默返回空集合（曾在 Module5 batch_optimizer 安全库存过滤
    复现）。此函数仅做"以 Excel 为基准"的向上兼容性转换，覆盖 datetime / 数值
    两大类；不动 Excel 中本来就是 object/str 的列。

    返回：实际被对齐的列名列表（用于日志，便于排查）。
    失败兜底：任一列 cast 异常仅 ``logger.warning``，不抛——避免单列脏数据
    拖垮整个加载流程。
    """
    aligned: list[str] = []
    try:
        # 只读 1 行作为 dtype 探针，开销可忽略
        sample = xl.parse(sheet_name, nrows=1)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"[CSV-DtypeAlign] 读取 Excel sheet '{sheet_name}' 探针失败，"
            f"跳过 dtype 对齐：{e}"
        )
        return aligned

    for col in sample.columns:
        if col not in csv_df.columns:
            continue
        target_dtype = sample[col].dtype
        # 已经一致就跳过
        if csv_df[col].dtype == target_dtype:
            continue
        try:
            if pd.api.types.is_datetime64_any_dtype(target_dtype):
                csv_df[col] = pd.to_datetime(csv_df[col], errors="coerce")
                aligned.append(f"{col}→datetime64")
            elif pd.api.types.is_integer_dtype(target_dtype):
                # CSV 整数列可能因含 NaN 被 pandas 读成 float；按 Excel 期望整数对齐
                csv_df[col] = pd.to_numeric(csv_df[col], errors="coerce").astype(target_dtype)
                aligned.append(f"{col}→{target_dtype}")
            elif pd.api.types.is_float_dtype(target_dtype):
                csv_df[col] = pd.to_numeric(csv_df[col], errors="coerce").astype(target_dtype)
                aligned.append(f"{col}→{target_dtype}")
            # 其它 dtype（object/bool/string 等）保持 CSV 原状，避免误判
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"[CSV-DtypeAlign] sheet '{sheet_name}' 列 '{col}' "
                f"对齐到 {target_dtype} 失败（保留 CSV 原 dtype）：{e}"
            )
    return aligned


def load_configuration(
    config,
    input_quality_checker=None,
    input_quality_context: dict | None = None,
) -> dict:
    """加载与标准化配置数据。

    目的：
    - 从 ``config/`` 目录读取唯一 Excel 与同级 CSV，补齐缺失的必要表，统一标准化
      标识符字段，并对 M4 换产配置执行重复性检查与去重映射。

    Args:
        config: ``ConfigDir`` 实例（首选）；或 Excel 路径字符串 / ``Path``（向后兼容）。
                传字符串/Path 时内部走 ``ConfigDir.from_excel_path``（CSV 唯一性仍校验）。
        input_quality_checker: 可选 input DQ checker；不传时按 defaults.yaml 构造。
        input_quality_context: 可选上下文，支持 ``report_dir``。

    Returns:
        dict: 标准化后的配置数据字典。

    核心原则：
        - **sheet 名权威**：以 ``xl.sheet_names`` 为唯一权威；循环前 ``tuple(...)``
          锁定快照，再用该快照逐项调用 ``cfg_dir.csv_for_sheet`` 做大小写不敏感匹配。
          不引入额外的规范 sheet 清单。
        - **CSV 无条件优先**：只要 ``config/`` 目录下存在同名 CSV（大小写不敏感），
          直接用 CSV、跳过 Excel sheet，与 sheet 是否为空无关。
        - **路径独立性**：CSV 来源仅限传入的 ``config/`` 目录一层，不递归、不回退到
          其他目录、不调用旧的 ``discover_csv_override_files``。

    逻辑：
        归一化为 ConfigDir → 加载（CSV 优先）→ 补齐必要表 → 标准化标识符
        → 检验并去重 Changeover 配置 → 映射关键表 → 完整性校验 → 返回字典。
    """
    # ---- 归一化为 ConfigDir（鸭子类型，避免顶层 import 形成循环依赖） ----
    if hasattr(config, "excel_path") and hasattr(config, "csv_map"):
        cfg_dir = config  # 已经是 ConfigDir
    else:
        from ..run.config_dir import ConfigDir  # 惰性 import
        cfg_dir = ConfigDir.from_excel_path(config)

    excel_path = str(cfg_dir.excel_path)
    logger.info(
        f"📋 加载配置目录: {cfg_dir.dir_path}（Excel: {cfg_dir.excel_path.name}, "
        f"CSV: {len(cfg_dir.csv_map)} 个）"
    )

    xl: pd.ExcelFile | None = None
    try:
        xl = pd.ExcelFile(excel_path)

        # ---- 先把 Excel 的 sheet 名锁定成元组快照，作为匹配唯一权威 ----
        sheet_names: tuple[str, ...] = tuple(xl.sheet_names)
        logger.info(f"  📑 Excel sheet 列表（{len(sheet_names)} 个）: {sheet_names}")

        config_dict: dict = {}
        applied_csv_count = 0

        # ---- 逐 sheet：CSV 无条件优先；CSV 后用 Excel 同 sheet 的 dtype 对齐 ----
        # 之所以要对齐：CSV 是纯文本，pandas 读出来 date 列会落成 ``str``，
        # 而 Excel 同列因 cell 格式会落成 ``datetime64``。下游若直接做
        # ``df['date'] == Timestamp`` 比较，CSV 路径会因 dtype 不匹配而静默
        # 返回空集合（曾导致 Module5 batch_optimizer 安全库存过滤失效）。
        # 修复策略：以 Excel sheet 头部 dtypes 为权威，CSV 同名列若需要则 cast。
        # ``float_precision="round_trip"``：用慢但完整精度的 float 解析器，避免
        # 默认 C 解析器在末位舍入导致 CSV 与 Excel 浮点列 1e-13 量级漂移。
        for sheet_name in sheet_names:
            csv_path = cfg_dir.csv_for_sheet(sheet_name)  # 内部按 .lower() 查 csv_map
            if csv_path is not None:
                csv_df = pd.read_csv(csv_path, float_precision="round_trip")
                aligned_cols = _align_csv_dtypes_to_excel(csv_df, xl, sheet_name)
                config_dict[sheet_name] = csv_df
                applied_csv_count += 1
                align_note = f"，对齐列: {aligned_cols}" if aligned_cols else ""
                logger.info(
                    f"  ✅ [CSV优先] {sheet_name} <- {csv_path.name} "
                    f"({len(csv_df)} 行){align_note}"
                )
            else:
                config_dict[sheet_name] = xl.parse(sheet_name)
                logger.info(f"  ✅ [Excel] {sheet_name} ({len(config_dict[sheet_name])} 行)")

        # ---- 仅有 CSV、Excel 无对应 sheet 的扩展数据源 ----
        loaded_lower = {s.lower() for s in sheet_names}
        for stem_lower, csv_path in cfg_dir.csv_map.items():
            if stem_lower not in loaded_lower:
                # 用 CSV 文件名（保留原大小写）作为 sheet 名
                config_dict[csv_path.stem] = pd.read_csv(
                    csv_path, float_precision="round_trip"
                )
                logger.info(
                    f"  ✅ [CSV扩展] {csv_path.stem} <- {csv_path.name} "
                    f"({len(config_dict[csv_path.stem])} 行)"
                )

        logger.info(
            f"📊 配置加载汇总: 共 {len(config_dict)} 个配置表 "
            f"(Excel sheet: {len(sheet_names)}, CSV 优先覆盖: {applied_csv_count}, "
            f"CSV 总数: {len(cfg_dir.csv_map)})"
        )

        # ---- input DQ：CSV dtype 对齐之后、identifier normalize 之前 ----
        try:
            checker = input_quality_checker
            if checker is None:
                from ...utils.data_quality import ConfigInputDataQualityChecker

                checker = ConfigInputDataQualityChecker.from_defaults()
            dq_context = input_quality_context or {}
            dq_result = checker.validate(
                config_dict,
                config_name=cfg_dir.excel_path.stem,
                sub_node="input_pre.config_loader",
                write_reports=bool(dq_context.get("report_dir")),
                output_dir=dq_context.get("report_dir"),
            )
            config_dict = dq_result["cleaned_tables"]
            logger.info(
                "✅ input DQ 完成: issues=%s, ignored_sheets=%s, ignored_columns=%s, blocked=%s",
                dq_result["summary"].get("issues"),
                dq_result["summary"].get("ignored_sheets"),
                dq_result["summary"].get("ignored_columns"),
                dq_result["blocked"],
            )
            if dq_result["blocked"]:
                from ...utils.data_quality import DataQualityError

                raise DataQualityError("Input DQ blocked config loading")
        except Exception:
            logger.exception("❌ input DQ 执行失败")
            raise

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

        # ---- 完整性校验（出口处统一）----
        _validate_config_dict(config_dict)

        return config_dict
    except Exception as e:
        logger.error(f"❌ 配置加载失败: {e}")
        raise
    finally:
        # 显式关闭，避免 Windows 上 ExcelFile 句柄滞留导致同目录文件无法删除/移动。
        if xl is not None:
            try:
                xl.close()
            except Exception:  # noqa: BLE001 — 关闭失败不该掩盖主异常
                pass


# ---------- 完整性校验 ----------
# 业务必需 sheet 名称（大小写不敏感匹配）。与 load_configuration 内 required_sheets 一致。
_REQUIRED_SHEETS: set[str] = {
    "M1_InitialInventory",
    "Global_SpaceCapacity",
    "Global_Network",
    "Global_LeadTime",
    "Global_DemandPriority",
}


def _sample_quantity_warning_values(
    series: pd.Series,
    mask: pd.Series,
    limit: int = 5,
) -> list[str]:
    samples: list[str] = []
    for idx, value in series[mask].head(limit).items():
        samples.append(f"{idx}={value!r}")
    return samples


def _warn_bad_quantity_columns(config_dict: dict) -> None:
    """Print warnings for empty or abnormal values in config columns named quantity."""
    for sheet_name, df in config_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        quantity_cols = [
            col for col in df.columns
            if str(col).strip().lower() == "quantity"
        ]
        for col in quantity_cols:
            series = df[col]
            null_mask = series.isna()
            blank_mask = series.map(
                lambda value: isinstance(value, str) and value.strip() == ""
            )
            empty_mask = null_mask | blank_mask

            try:
                numeric = pd.to_numeric(series, errors="coerce")
                numeric_values = numeric.to_numpy(dtype="float64", na_value=np.nan)
                finite_mask = pd.Series(
                    np.isfinite(numeric_values),
                    index=series.index,
                )
                numeric_na_mask = numeric.isna()
            except Exception:
                finite_mask = pd.Series(False, index=series.index)
                numeric_na_mask = pd.Series(True, index=series.index)

            non_numeric_mask = (~empty_mask) & numeric_na_mask
            infinite_mask = (~numeric_na_mask) & (~finite_mask)

            empty_count = int(empty_mask.sum())
            non_numeric_count = int(non_numeric_mask.sum())
            infinite_count = int(infinite_mask.sum())
            if empty_count == 0 and non_numeric_count == 0 and infinite_count == 0:
                continue

            bad_mask = empty_mask | non_numeric_mask | infinite_mask
            logger.warning(
                "[ConfigValidation][quantity] sheet='%s' column='%s' has "
                "empty=%d, non_numeric=%d, infinite=%d. samples=%s",
                sheet_name,
                col,
                empty_count,
                non_numeric_count,
                infinite_count,
                _sample_quantity_warning_values(series, bad_mask),
            )


def _validate_config_dict(
    config_dict: dict,
    required: set[str] | None = None,
) -> None:
    """校验 ``config_dict`` 中必需 sheet 是否全部加载、各 sheet 是否非空。

    当前实现：必需 sheet 缺失走 warning（与现有 ``load_configuration`` "补空表 + warning"
    语义一致，避免破坏现有可跑配置）；空 DataFrame 走 info 提示。若团队后续决定改成
    硬失败（缺失即 raise），把 ``logger.warning`` 替换为 ``raise ValueError`` 即可。
    """
    _warn_bad_quantity_columns(config_dict)
    if required is None:
        required = _REQUIRED_SHEETS
    loaded_lower = {k.lower() for k in config_dict}
    missing = sorted(s for s in required if s.lower() not in loaded_lower)
    if missing:
        logger.warning(
            f"[ConfigValidation] config_dict 缺少必需 sheet：{missing}（已由上游补空表）"
        )
    for name, df in config_dict.items():
        if isinstance(df, pd.DataFrame) and df.empty:
            logger.info(
                f"[ConfigValidation] sheet '{name}' 加载后为空 DataFrame，请确认数据源是否有效"
            )

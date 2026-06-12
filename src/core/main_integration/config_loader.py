"""Configuration loading, input quality validation, and preparation helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ...utils.normalization import normalize_identifiers

if TYPE_CHECKING:
    from ..run.config_dir import ConfigDir


logger = logging.getLogger("SupplyChainSimulation")

_EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}

_DB_SHEET_MAPPING = {
    "sit_design": "SIT Design",
    "global_seed": "Global_seed",
    "config_guide": "Config Guide",
    "global_network": "Global_Network",
    "global_spacecapacity": "Global_SpaceCapacity",
    "global_leadtime": "Global_LeadTime",
    "global_demandpriority": "Global_DemandPriority",
    "m1_initialinventory": "M1_InitialInventory",
    "m1_initialinventory_30d": "M1_InitialInventory_30D",
    "sheet1": "Sheet1",
    "m1_demandforecast": "M1_DemandForecast",
    "m1_forecasterror": "M1_ForecastError",
    "m1_ordercalendar": "M1_OrderCalendar",
    "m1_aoconfig": "M1_AOConfig",
    "m1_dpsconfig": "M1_DPSConfig",
    "m1_supplychoiceconfig": "M1_SupplyChoiceConfig",
    "m3_safetystock": "M3_SafetyStock",
    "covalidation": "COValidation",
    "m4_materiallocationlinecfg": "M4_MaterialLocationLineCfg",
    "m4_linecapacity": "M4_LineCapacity",
    "m4_changeovermatrix": "M4_ChangeoverMatrix",
    "m4_changeoverdefinition": "M4_ChangeoverDefinition",
    "m4_productionreliability": "M4_ProductionReliability",
    "m5_pushpullmodel": "M5_PushPullModel",
    "m5_deployconfig": "M5_DeployConfig",
    "m6_truckreleasecon": "M6_TruckReleaseCon",
    "m6_materialmd": "M6_MaterialMD",
    "m6_deliverydelaydistribution": "M6_DeliveryDelayDistribution",
    "m6_mdqbypassrules": "M6_MDQBypassRules",
    "m6_trucktypespecs": "M6_TruckTypeSpecs",
    "m6_truckcapacityplan": "M6_TruckCapacityPlan",
}

_DB_COLUMN_MAPPING = {
    "material": "material",
    "location": "location",
    "sourcing": "sourcing",
    "location_type": "location_type",
    "quantity": "quantity",
    "date": "date",
    "week": "week",
    "day": "day",
    "seed": "seed",
    "eff_from": "eff_from",
    "eff_to": "eff_to",
    "demand_element": "demand_element",
    "priority": "priority",
    "order_type": "order_type",
    "error_std_percent": "error_std_percent",
    "order_day_flag": "order_day_flag",
    "advance_days": "advance_days",
    "ao_percent": "ao_percent",
    "dps_location": "dps_location",
    "dps_percent": "dps_percent",
    "safety_stock_qty": "safety_stock_qty",
    "key": "key",
    "sending": "sending",
    "receiving": "receiving",
    "pdt": "PDT",
    "gr": "GR",
    "mct": "MCT",
    "otd": "OTD",
    "delegate_line": "delegate_line",
    "prd_rate": "prd_rate",
    "min_batch": "min_batch",
    "rv": "rv",
    "ptf": "ptf",
    "lsk": "lsk",
    "line": "line",
    "capacity": "capacity",
    "from_material": "from_material",
    "to_material": "to_material",
    "changeover_id": "changeover_id",
    "from_line": "from line",
    "to_line": "to line",
    "time": "time",
    "cost": "cost",
    "mu_loss": "mu_loss",
    "pr": "pr",
    "model": "model",
    "moq": "moq",
    "truck_type": "truck_type",
    "optimal_type": "optimal_type",
    "wfr": "WFR",
    "vfr": "VFR",
    "mdq": "MDQ",
    "weight": "weight",
    "volume": "volume",
    "demand_unit_to_weight": "demand_unit_to_weight",
    "demand_unit_to_volume": "demand_unit_to_volume",
    "delay_days": "delay_days",
    "probability": "probability",
    "condition_logic": "condition_logic",
    "rule_id": "rule_id",
    "max_weight": "max_weight",
    "max_volume": "max_volume",
    "capacity_qty_in_weight": "capacity_qty_in_weight",
    "capacity_qty_in_volume": "capacity_qty_in_volume",
}

_DB_METADATA_COLUMNS = {"config_name", "config_type", "db_write_time"}
_REQUIRED_SHEET_ORDER = [
    "M1_InitialInventory",
    "Global_SpaceCapacity",
    "Global_Network",
    "Global_LeadTime",
    "Global_DemandPriority",
]
_REQUIRED_SHEETS = set(_REQUIRED_SHEET_ORDER)
_IDENTIFIER_COLUMNS = [
    "material",
    "location",
    "sending",
    "receiving",
    "sourcing",
    "dps_location",
    "from_material",
    "to_material",
    "line",
    "delegate_line",
    "changeover_id",
]
_MODULE4_MAPPINGS = {
    "M4_MaterialLocationLineCfg": "MaterialLocationLineCfg",
    "M4_LineCapacity": "LineCapacity",
    "M4_ChangeoverMatrix": "ChangeoverMatrix",
    "M4_ChangeoverDefinition": "ChangeoverDefinition",
    "M4_ProductionReliability": "ProductionReliability",
}


def load_configuration_from_dict(
    config_data: dict,
    config_name: str = "DB_Config",
) -> dict[str, pd.DataFrame]:
    """Convert DB-origin configuration frames into local sheet-shaped tables."""
    logger.info(
        "Processing configuration data: %s (%d tables)",
        config_name,
        len(config_data),
    )

    config_dict: dict[str, pd.DataFrame] = {}
    for db_name, df in config_data.items():
        if not isinstance(df, pd.DataFrame):
            continue
        sheet_name = _DB_SHEET_MAPPING.get(str(db_name).lower(), str(db_name))
        converted_df = _convert_db_config_frame(df)
        config_dict[sheet_name] = converted_df
        logger.info(
            "  Loaded configuration table: %s (%d rows)",
            sheet_name,
            len(converted_df),
        )
    return config_dict


def _convert_db_config_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Restore local column names and drop DB-only metadata columns."""
    df_copy = df.copy()
    df_copy.columns = [
        _DB_COLUMN_MAPPING.get(str(column).lower(), column)
        for column in df_copy.columns
    ]

    drop_cols = [
        column for column in df_copy.columns
        if _should_drop_db_column(column)
    ]
    if drop_cols:
        df_copy = df_copy.drop(columns=drop_cols, errors="ignore")

    cols_before = list(df_copy.columns)
    df_copy = df_copy.dropna(axis=1, how="all")
    for column in cols_before:
        if column not in df_copy.columns:
            df_copy[column] = pd.NA
    return df_copy


def _should_drop_db_column(column: object) -> bool:
    """Return True when a DB-origin column should not reach simulation modules."""
    column_name = str(column)
    return (
        column_name in _DB_METADATA_COLUMNS
        or column_name.lower().startswith("unnamed")
        or (
            not column_name.isascii()
            and column_name not in _DB_COLUMN_MAPPING.values()
        )
    )


def _align_csv_dtypes_to_excel(
    csv_df: pd.DataFrame,
    xl: pd.ExcelFile,
    sheet_name: str,
) -> list[str]:
    """Align CSV columns to the same-sheet Excel dtypes in place."""
    aligned: list[str] = []
    try:
        sample = xl.parse(sheet_name, nrows=1)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[CSV-DtypeAlign] Failed to inspect Excel sheet '%s': %s",
            sheet_name,
            exc,
        )
        return aligned

    for column in sample.columns:
        if column not in csv_df.columns:
            continue
        target_dtype = sample[column].dtype
        if csv_df[column].dtype == target_dtype:
            continue
        try:
            if pd.api.types.is_datetime64_any_dtype(target_dtype):
                csv_df[column] = pd.to_datetime(csv_df[column], errors="coerce")
                aligned.append(f"{column}->datetime64")
            elif pd.api.types.is_integer_dtype(target_dtype):
                numeric = pd.to_numeric(csv_df[column], errors="coerce")
                csv_df[column] = numeric.astype(target_dtype)
                aligned.append(f"{column}->{target_dtype}")
            elif pd.api.types.is_float_dtype(target_dtype):
                numeric = pd.to_numeric(csv_df[column], errors="coerce")
                csv_df[column] = numeric.astype(target_dtype)
                aligned.append(f"{column}->{target_dtype}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[CSV-DtypeAlign] Failed to align sheet '%s' column '%s' to %s: %s",
                sheet_name,
                column,
                target_dtype,
                exc,
            )
    return aligned


def load_configuration(
    config
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
        "Loading config directory: %s (Excel: %s, CSV: %d)",
        cfg_dir.dir_path,
        cfg_dir.excel_path.name,
        len(cfg_dir.csv_map),
    )

    xl: pd.ExcelFile | None = None
    try:
        xl = pd.ExcelFile(str(cfg_dir.excel_path))
        sheet_names = tuple(xl.sheet_names)
        config_dict = _load_excel_authoritative_tables(cfg_dir, xl, sheet_names)
        config_dict.update(_load_csv_extension_tables(cfg_dir, sheet_names))
        logger.info(
            "Configuration load summary: tables=%d, excel_sheets=%d, csv_total=%d",
            len(config_dict),
            len(sheet_names),
            len(cfg_dir.csv_map),
        )

        # # ---- input DQ：CSV dtype 对齐之后、identifier normalize 之前 ----
        # try:
        #     checker = input_quality_checker
        #     if checker is None:
        #         from ...utils.data_quality import ConfigInputDataQualityChecker

        #         checker = ConfigInputDataQualityChecker.from_defaults()
        #     dq_context = input_quality_context or {}
        #     dq_result = checker.validate(
        #         config_dict,
        #         config_name=cfg_dir.excel_path.stem,
        #         sub_node="input_pre.config_loader",
        #         write_reports=bool(dq_context.get("report_dir")),
        #         output_dir=dq_context.get("report_dir"),
        #     )
        #     config_dict = dq_result["cleaned_tables"]
        #     logger.info(
        #         "✅ input DQ 完成: issues=%s, ignored_sheets=%s, ignored_columns=%s, blocked=%s",
        #         dq_result["summary"].get("issues"),
        #         dq_result["summary"].get("ignored_sheets"),
        #         dq_result["summary"].get("ignored_columns"),
        #         dq_result["blocked"],
        #     )
        #     if dq_result["blocked"]:
        #         from ...utils.data_quality import DataQualityError

        #         raise DataQualityError("Input DQ blocked config loading")
        # except Exception:
        #     logger.exception("❌ input DQ 执行失败")
        #     raise

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
    except Exception as exc:
        logger.error("Configuration load failed: %s", exc)
        raise
    finally:
        if xl is not None:
            try:
                xl.close()
            except Exception:  # noqa: BLE001
                pass


def _coerce_config_dir(config) -> "ConfigDir":
    """Convert supported config inputs to a ConfigDir."""
    if hasattr(config, "excel_path") and hasattr(config, "csv_map"):
        return config

    from ..run.config_dir import ConfigDir

    return ConfigDir.from_excel_path(config)


def _load_excel_authoritative_tables(
    cfg_dir: "ConfigDir",
    xl: pd.ExcelFile,
    sheet_names: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Load every Excel sheet, replacing a sheet with same-stem CSV when present."""
    config_dict: dict[str, pd.DataFrame] = {}
    applied_csv_count = 0

    for sheet_name in sheet_names:
        csv_path = cfg_dir.csv_for_sheet(sheet_name)
        if csv_path is None:
            config_dict[sheet_name] = xl.parse(sheet_name)
            logger.info(
                "  Loaded Excel sheet: %s (%d rows)",
                sheet_name,
                len(config_dict[sheet_name]),
            )
            continue

        csv_df = pd.read_csv(csv_path, float_precision="round_trip")
        aligned_cols = _align_csv_dtypes_to_excel(csv_df, xl, sheet_name)
        config_dict[sheet_name] = csv_df
        applied_csv_count += 1
        logger.info(
            "  Loaded CSV override: %s <- %s (%d rows, aligned=%s)",
            sheet_name,
            csv_path.name,
            len(csv_df),
            aligned_cols,
        )

    logger.info("CSV overrides applied: %d", applied_csv_count)
    return config_dict


def _load_csv_extension_tables(
    cfg_dir: "ConfigDir",
    sheet_names: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Load CSV files that do not match any Excel sheet name."""
    loaded_lower = {sheet_name.lower() for sheet_name in sheet_names}
    extension_tables: dict[str, pd.DataFrame] = {}
    for stem_lower, csv_path in cfg_dir.csv_map.items():
        if stem_lower in loaded_lower:
            continue
        extension_tables[csv_path.stem] = pd.read_csv(
            csv_path,
            float_precision="round_trip",
        )
        logger.info(
            "  Loaded CSV extension: %s <- %s (%d rows)",
            csv_path.stem,
            csv_path.name,
            len(extension_tables[csv_path.stem]),
        )
    return extension_tables


def validate_input_quality(
    config_dict: dict[str, pd.DataFrame],
    *,
    config_name: str,
    report_dir: str | Path | None = None,
    checker=None,
) -> dict[str, Any]:
    """Run input data quality checks and return the checker result (non-blocking)."""
    if checker is None:
        from ...utils.data_quality import ConfigInputDataQualityChecker

        checker = ConfigInputDataQualityChecker()

    dq_result = checker.validate(
        config_dict,
        config_name=config_name,
        report_dir=report_dir,
    )
    logger.info(
        "Input DQ complete: issues=%s, errors=%s, warnings=%s, passed=%s",
        dq_result["summary"].get("issues"),
        dq_result["summary"].get("errors"),
        dq_result["summary"].get("warnings"),
        dq_result["passed"],
    )
    return dq_result


def ensure_required_sheets(
    config_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return a config dict with required sheets present as empty frames."""
    prepared = dict(config_dict)
    loaded_lower = {sheet_name.lower() for sheet_name in prepared}
    missing_sheets = [
        sheet_name for sheet_name in _REQUIRED_SHEET_ORDER
        if sheet_name.lower() not in loaded_lower
    ]
    if missing_sheets:
        logger.warning("Missing required configuration sheets: %s", missing_sheets)
    for sheet_name in missing_sheets:
        prepared[sheet_name] = pd.DataFrame()
    return prepared


def normalize_configuration_identifiers(
    config_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return a config dict with identifier columns normalized."""
    prepared = dict(config_dict)
    standardized_count = 0
    for sheet_name, df in config_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if not any(column in df.columns for column in _IDENTIFIER_COLUMNS):
            continue

        original_dtypes = {
            column: str(df[column].dtype)
            for column in _IDENTIFIER_COLUMNS
            if column in df.columns
        }
        normalized_df = normalize_identifiers(df)
        prepared[sheet_name] = normalized_df
        new_dtypes = {
            column: str(normalized_df[column].dtype)
            for column in _IDENTIFIER_COLUMNS
            if column in normalized_df.columns
        }
        changed_fields = [
            f"{column}({original_dtypes[column]}->{new_dtypes[column]})"
            for column in original_dtypes
            if original_dtypes[column] != new_dtypes[column]
        ]
        if changed_fields:
            logger.info("  Normalized %s: %s", sheet_name, ", ".join(changed_fields))
            standardized_count += 1

    logger.info("Identifier normalization completed for %d tables", standardized_count)
    return prepared


def deduplicate_changeover_configuration(
    config_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return a config dict with duplicate Changeover rows removed."""
    prepared = dict(config_dict)
    _deduplicate_changeover_matrix(prepared)
    _deduplicate_changeover_definition(prepared)
    return prepared


def _deduplicate_changeover_matrix(config_dict: dict[str, pd.DataFrame]) -> None:
    """Deduplicate M4_ChangeoverMatrix rows in place."""
    sheet_name = "M4_ChangeoverMatrix"
    if sheet_name not in config_dict or config_dict[sheet_name].empty:
        return

    co_matrix = config_dict[sheet_name]
    duplicates = co_matrix[
        co_matrix.duplicated(subset=["from_material", "to_material"], keep=False)
    ]
    if duplicates.empty:
        logger.info("  Changeover Matrix has no duplicate definitions")
        return

    logger.warning("  Found %d duplicate changeover matrix rows", len(duplicates))
    for (from_mat, to_mat), group in duplicates.groupby(["from_material", "to_material"]):
        unique_coids = group["changeover_id"].unique()
        if len(unique_coids) > 1:
            logger.error(
                "    %s -> %s has multiple changeover_id values: %s",
                from_mat,
                to_mat,
                list(unique_coids),
            )
        else:
            logger.warning(
                "    %s -> %s has %d duplicate rows (changeover_id=%s)",
                from_mat,
                to_mat,
                len(group),
                unique_coids[0],
            )
    config_dict[sheet_name] = co_matrix.drop_duplicates(
        subset=["from_material", "to_material"],
        keep="first",
    )


def _deduplicate_changeover_definition(config_dict: dict[str, pd.DataFrame]) -> None:
    """Deduplicate M4_ChangeoverDefinition rows in place."""
    sheet_name = "M4_ChangeoverDefinition"
    if sheet_name not in config_dict or config_dict[sheet_name].empty:
        return

    co_def = config_dict[sheet_name]
    duplicates = co_def[
        co_def.duplicated(subset=["changeover_id", "line"], keep=False)
    ]
    if duplicates.empty:
        logger.info("  Changeover Definition has no duplicate definitions")
        return

    logger.warning("  Found %d duplicate changeover definition rows", len(duplicates))
    for (coid, line), group in duplicates.groupby(["changeover_id", "line"]):
        unique_times = group["time"].unique()
        if len(unique_times) > 1:
            logger.error(
                "    changeover_id=%s, line=%s has multiple time values: %s",
                coid,
                line,
                list(unique_times),
            )
        else:
            logger.warning(
                "    changeover_id=%s, line=%s has %d duplicate rows (time=%s)",
                coid,
                line,
                len(group),
                unique_times[0],
            )
    config_dict[sheet_name] = co_def.drop_duplicates(
        subset=["changeover_id", "line"],
        keep="first",
    )


def map_module4_configuration_keys(
    config_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return a config dict with Module4 backward-compatible aliases."""
    prepared = dict(config_dict)
    mapped_count = 0
    for original_key, mapped_key in _MODULE4_MAPPINGS.items():
        if original_key in prepared and not prepared[original_key].empty:
            prepared[mapped_key] = prepared[original_key]
            logger.info("  Mapped %s -> %s", original_key, mapped_key)
            mapped_count += 1
    logger.info("Mapped %d Module4 configuration tables", mapped_count)
    return prepared


def prepare_configuration(
    config_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Prepare loaded configuration tables for simulation consumers."""
    prepared = ensure_required_sheets(config_dict)
    prepared = normalize_configuration_identifiers(prepared)
    prepared = deduplicate_changeover_configuration(prepared)
    prepared = map_module4_configuration_keys(prepared)
    _validate_config_dict(prepared)
    return prepared


def _sample_quantity_warning_values(
    series: pd.Series,
    mask: pd.Series,
    limit: int = 5,
) -> list[str]:
    """Return sample values for quantity validation warnings."""
    samples: list[str] = []
    for idx, value in series[mask].head(limit).items():
        samples.append(f"{idx}={value!r}")
    return samples


def _warn_bad_quantity_columns(config_dict: dict) -> None:
    """Print warnings for empty or abnormal values in columns named quantity."""
    for sheet_name, df in config_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        quantity_cols = [
            column for column in df.columns
            if str(column).strip().lower() == "quantity"
        ]
        for column in quantity_cols:
            series = df[column]
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
                column,
                empty_count,
                non_numeric_count,
                infinite_count,
                _sample_quantity_warning_values(series, bad_mask),
            )


def _validate_config_dict(
    config_dict: dict,
    required: set[str] | None = None,
) -> None:
    """Validate loaded config table presence and obvious empty tables."""
    _warn_bad_quantity_columns(config_dict)
    if required is None:
        required = _REQUIRED_SHEETS
    loaded_lower = {key.lower() for key in config_dict}
    missing = sorted(sheet for sheet in required if sheet.lower() not in loaded_lower)
    if missing:
        logger.warning(
            "[ConfigValidation] config_dict is missing required sheets: %s",
            missing,
        )
    for name, df in config_dict.items():
        if isinstance(df, pd.DataFrame) and df.empty:
            logger.info(
                "[ConfigValidation] sheet '%s' is an empty DataFrame after loading",
                name,
            )

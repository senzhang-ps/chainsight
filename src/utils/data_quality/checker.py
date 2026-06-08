"""配置表输入数据质量检测模块。

本模块基于固定的 cfg_* 入库表契约，对 Excel/CSV 读取后的配置表执行
字段投影、质量检测、类型转换和报告输出。只有 schema 中声明的 Sheet
和入库字段会进入检测与入库链路。
"""
from __future__ import annotations

import numbers
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from pgsql_db.config_comments import is_primary_key_marker
from pgsql_db.config_table_schema import get_config_table_mapping
from src.utils.normalization import normalize_location, normalize_material


class DataQualityError(RuntimeError):
    """检测到阻断级数据质量问题时抛出的异常。"""


# 面向操作员的 Excel 输出列；内部诊断字段仅保留在内存 issue 明细中。
_ISSUE_COLUMNS = [
    "配置名",
    "配置表Sheet",
    "字段名",
    "问题类型",
    "问题代码",
    "严重级别",
    "处理动作",
    "转换后值",
]
_ISSUE_TYPE_LABELS_ZH = {
    "missing_sheet": "缺失Sheet",
    "missing_import_column": "缺失列",
    "empty_table": "空表",
    "null_value": "空值",
    "mixed_column_type": "同字段多格式",
    "type_mismatch": "类型/格式不可解析",
    "negative_value": "负数",
    "out_of_range": "超出范围",
    "invalid_enum": "非法枚举",
    "invalid_date_range": "日期范围非法",
    "duplicate_value": "重复值",
    "duplicate_conflict": "冲突重复",
}
_MISSING_SHEET_COLUMNS = ["sheet", "db_table", "reason", "severity", "action"]
_IGNORED_SHEET_COLUMNS = ["sheet", "reason", "action"]
_IGNORED_COLUMN_COLUMNS = ["sheet", "column", "reason", "action"]


@dataclass(frozen=True)
class _TableContract:
    """单张配置表的固定入库契约。"""

    key: str
    local_sheet: str
    db_table: str
    import_enabled: bool
    data_quality_enabled: bool
    primary_key: tuple[str, ...]
    fields: tuple[dict[str, Any], ...]

    @property
    def db_key(self) -> str:
        return self.db_table[4:] if self.db_table.startswith("cfg_") else self.db_table

    @property
    def local_fields(self) -> list[str]:
        return [str(field["local_name"]) for field in self.fields]


@dataclass
class _TableRuleContext:
    """表级专属质量规则的执行上下文。"""

    contract: _TableContract
    df: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    issues: list[dict[str, Any]]
    config_name: str | None
    sub_node: str


class ConfigTableQualityRules:
    """配置表专属质量规则调度器。

    字段级通用规则由 ``ConfigInputDataQualityChecker`` 统一执行；
    本类只承载依赖具体配置表业务语义的检测规则，并保持一个 Sheet
    对应一个显式检测函数。
    """

    # 固定维护 Sheet 与专属规则函数的映射，作为表级规则的唯一调度入口。
    TABLE_METHODS = {
        "Global_seed": "check_global_seed",
        "Global_Network": "check_global_network",
        "Global_SpaceCapacity": "check_global_spacecapacity",
        "Global_LeadTime": "check_global_leadtime",
        "Global_DemandPriority": "check_global_demandpriority",
        "M1_InitialInventory": "check_m1_initialinventory",
        "M1_DemandForecast": "check_m1_demandforecast",
        "M1_ForecastError": "check_m1_forecasterror",
        "M1_OrderCalendar": "check_m1_ordercalendar",
        "M1_AOConfig": "check_m1_aoconfig",
        "M1_DPSConfig": "check_m1_dpsconfig",
        "M1_SupplyChoiceConfig": "check_m1_supplychoiceconfig",
        "M3_SafetyStock": "check_m3_safetystock",
        "M4_MaterialLocationLineCfg": "check_m4_materiallocationlinecfg",
        "M4_LineCapacity": "check_m4_linecapacity",
        "M4_ChangeoverMatrix": "check_m4_changeovermatrix",
        "M4_ChangeoverDefinition": "check_m4_changeoverdefinition",
        "M4_ProductionReliability": "check_m4_productionreliability",
        "M5_PushPullModel": "check_m5_pushpullmodel",
        "M5_DeployConfig": "check_m5_deployconfig",
        "M6_TruckReleaseCon": "check_m6_truckreleasecon",
        "M6_MaterialMD": "check_m6_materialmd",
        "M6_DeliveryDelayDistribution": "check_m6_deliverydelaydistribution",
        "M6_MDQBypassRules": "check_m6_mdqbypassrules",
        "M6_TruckTypeSpecs": "check_m6_trucktypespecs",
        "M6_TruckCapacityPlan": "check_m6_truckcapacityplan",
    }

    def __init__(self, *, issue_factory, missing_mask, sample_limit: int) -> None:
        """初始化表级专属质量规则调度器。

        Args:
            issue_factory: 构造统一问题明细记录的工厂函数，复用主检测器的 ``_issue``。
            missing_mask: 识别空值和空白字符串的掩码函数，复用主检测器的 ``_missing_mask``。
            sample_limit: 单类问题最多记录的样例数量。
        """
        self._issue_factory = issue_factory  # 构造统一问题明细记录的工厂函数
        self._missing_mask = missing_mask  # 识别空值和空白字符串的掩码函数
        self.sample_limit = sample_limit  # 单类问题最多记录的样例数量

    def method_name_for_sheet(self, sheet_name: str) -> str | None:
        """返回某个 Sheet 对应的专属规则函数名。

        Args:
            sheet_name: 本地配置表 Sheet 名。

        Returns:
            映射到的规则函数名；该 Sheet 无专属规则时返回 None。
        """
        return self.TABLE_METHODS.get(sheet_name)

    def validate_all(
        self,
        tables: dict[str, pd.DataFrame],
        contracts: list[_TableContract],
        issues: list[dict[str, Any]],
        *,
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """对所有已开启质量检测的表契约执行专属规则。

        Args:
            tables: 已完成字段投影和通用检测的配置表数据，key 为本地 Sheet 名。
            contracts: 需要执行表级专属规则的配置表契约列表。
            issues: 质量问题明细列表；规则函数会在该列表中追加问题记录。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        for contract in contracts:
            df = tables.get(contract.local_sheet)
            if df is None:
                continue
            self.validate_table(
                contract,
                df,
                tables,
                issues,
                config_name=config_name,
                sub_node=sub_node,
            )

    def validate_table(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        tables: dict[str, pd.DataFrame],
        issues: list[dict[str, Any]],
        *,
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """解析并调用单张配置表对应的专属规则函数。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 当前配置表经过字段投影后的数据。
            tables: 所有已投影配置表数据，用于需要跨表上下文的规则。
            issues: 质量问题明细列表；命中的规则会在该列表中追加记录。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        method_name = self.method_name_for_sheet(contract.local_sheet)
        if method_name is None:
            return
        context = _TableRuleContext(
            contract=contract,
            df=df,
            tables=tables,
            issues=issues,
            config_name=config_name,
            sub_node=sub_node,
        )
        getattr(self, method_name)(context)

    def check_global_seed(self, context: _TableRuleContext) -> None:
        """Global_seed 表专属规则：无业务级专属约束，仅占位以保持一表一函数。"""
        pass

    def check_global_network(self, context: _TableRuleContext) -> None:
        """Global_Network 表专属规则：校验生效区间 eff_to 不早于 eff_from。"""
        self._check_date_order(context, "eff_from", "eff_to")

    def check_global_spacecapacity(self, context: _TableRuleContext) -> None:
        """Global_SpaceCapacity 表专属规则：校验生效区间顺序，并要求 capacity 非负。"""
        self._check_date_order(context, "eff_from", "eff_to")
        self._check_non_negative_numbers(context, ("capacity",))

    def check_global_leadtime(self, context: _TableRuleContext) -> None:
        """Global_LeadTime 表专属规则：要求各项提前期 PDT/GR/MCT/OTD 非负。"""
        self._check_non_negative_numbers(context, ("PDT", "GR", "MCT", "OTD"))

    def check_global_demandpriority(self, context: _TableRuleContext) -> None:
        """Global_DemandPriority 表专属规则：要求需求优先级 priority 非负。"""
        self._check_non_negative_numbers(context, ("priority",))

    def check_m1_initialinventory(self, context: _TableRuleContext) -> None:
        """M1_InitialInventory 表专属规则：要求期初库存数量 quantity 非负。"""
        self._check_non_negative_numbers(context, ("quantity",))

    def check_m1_demandforecast(self, context: _TableRuleContext) -> None:
        """M1_DemandForecast 表专属规则：要求需求预测数量 quantity 非负。"""
        self._check_non_negative_numbers(context, ("quantity",))

    def check_m1_forecasterror(self, context: _TableRuleContext) -> None:
        """M1_ForecastError 表专属规则：要求预测误差标准差百分比 error_std_percent 非负。"""
        self._check_non_negative_numbers(context, ("error_std_percent",))

    def check_m1_ordercalendar(self, context: _TableRuleContext) -> None:
        """M1_OrderCalendar 表专属规则：无业务级专属约束，仅占位以保持一表一函数。"""
        pass

    def check_m1_aoconfig(self, context: _TableRuleContext) -> None:
        """M1_AOConfig 表专属规则：要求提前下单天数 advance_days 非负，AO 占比 ao_percent 落在 [0, 1]。"""
        self._check_non_negative_numbers(context, ("advance_days",))
        self._check_number_range(context, ("ao_percent",), min_value=0, max_value=1)

    def check_m1_dpsconfig(self, context: _TableRuleContext) -> None:
        """M1_DPSConfig 表专属规则：要求 DPS 占比 dps_percent 落在 [0, 1] 区间。"""
        self._check_number_range(context, ("dps_percent",), min_value=0, max_value=1)

    def check_m1_supplychoiceconfig(self, context: _TableRuleContext) -> None:
        """M1_SupplyChoiceConfig 表专属规则：无业务级专属约束，仅占位以保持一表一函数。"""
        pass

    def check_m3_safetystock(self, context: _TableRuleContext) -> None:
        """M3_SafetyStock 表专属规则：要求安全库存数量 safety_stock_qty 非负。"""
        self._check_non_negative_numbers(context, ("safety_stock_qty",))

    def check_m4_materiallocationlinecfg(self, context: _TableRuleContext) -> None:
        """M4_MaterialLocationLineCfg 表专属规则：要求产能、批量等各项数值参数非负。"""
        self._check_non_negative_numbers(
            context,
            ("prd_rate", "min_batch", "rv", "ptf", "lsk", "day", "MCT"),
        )

    def check_m4_linecapacity(self, context: _TableRuleContext) -> None:
        """M4_LineCapacity 表专属规则：要求产线产能 capacity 非负。"""
        self._check_non_negative_numbers(context, ("capacity",))

    def check_m4_changeovermatrix(self, context: _TableRuleContext) -> None:
        """M4_ChangeoverMatrix 表专属规则：要求同一切换物料对的 changeover_id 取值一致、无冲突。"""
        self._check_conflicting_values(
            context,
            key_columns=("from_material", "to_material"),
            value_columns=("changeover_id",),
        )

    def check_m4_changeoverdefinition(self, context: _TableRuleContext) -> None:
        """M4_ChangeoverDefinition 表专属规则：要求切换耗时 time、成本 cost、产能损失 mu_loss 非负。"""
        self._check_non_negative_numbers(context, ("time", "cost", "mu_loss"))

    def check_m4_productionreliability(self, context: _TableRuleContext) -> None:
        """M4_ProductionReliability 表专属规则：要求生产可靠率 pr 落在 [0, 1] 区间。"""
        self._check_number_range(context, ("pr",), min_value=0, max_value=1)

    def check_m5_pushpullmodel(self, context: _TableRuleContext) -> None:
        """M5_PushPullModel 表专属规则：要求 model 取 push、pull 或 soft push 之一。"""
        self._check_enum(
            context,
            "model",
            allowed={"push", "pull", "soft push"},
            rule_id="m5_pushpullmodel.model.enum",
            message="model must be one of push, pull, soft push",
        )

    def check_m5_deployconfig(self, context: _TableRuleContext) -> None:
        """M5_DeployConfig 表专属规则：要求 moq、rv、lsk、day 等部署参数非负。"""
        self._check_non_negative_numbers(context, ("moq", "rv", "lsk", "day"))

    def check_m6_truckreleasecon(self, context: _TableRuleContext) -> None:
        """M6_TruckReleaseCon 表专属规则：要求 WFR/VFR 为非空数值，且最小发货量 MDQ 非负。"""
        self._check_m6_wfr_vfr_numeric(context)
        self._check_non_negative_numbers(context, ("MDQ",))

    def check_m6_materialmd(self, context: _TableRuleContext) -> None:
        """M6_MaterialMD 表专属规则：要求需求单位到重量、到体积的换算系数非负。"""
        self._check_non_negative_numbers(
            context,
            ("demand_unit_to_weight", "demand_unit_to_volume"),
        )

    def check_m6_deliverydelaydistribution(self, context: _TableRuleContext) -> None:
        """M6_DeliveryDelayDistribution 表专属规则：要求延误天数 delay_days 非负，概率 probability 落在 [0, 1]。"""
        self._check_non_negative_numbers(context, ("delay_days",))
        self._check_number_range(context, ("probability",), min_value=0, max_value=1)

    def check_m6_mdqbypassrules(self, context: _TableRuleContext) -> None:
        """M6_MDQBypassRules 表专属规则：无业务级专属约束，仅占位以保持一表一函数。"""
        pass

    def check_m6_trucktypespecs(self, context: _TableRuleContext) -> None:
        """M6_TruckTypeSpecs 表专属规则：要求车型按重量、体积计的运力 capacity_qty_* 非负。"""
        self._check_non_negative_numbers(
            context,
            ("capacity_qty_in_weight", "capacity_qty_in_volume"),
        )

    def check_m6_truckcapacityplan(self, context: _TableRuleContext) -> None:
        """M6_TruckCapacityPlan 表专属规则：要求车辆数量 truck_number 非负。"""
        self._check_non_negative_numbers(context, ("truck_number",))

    def _check_non_negative_numbers(
        self,
        context: _TableRuleContext,
        columns: tuple[str, ...],
    ) -> None:
        """校验指定数值字段是否满足非负约束。

        Args:
            context: 表级规则执行上下文，包含当前表数据和问题列表。
            columns: 需要执行非负约束检测的字段名集合。
        """
        self._check_number_range(
            context,
            columns,
            min_value=0,
            issue_type="negative_value",
            rule_suffix="non_negative",
        )

    def _check_number_range(
        self,
        context: _TableRuleContext,
        columns: tuple[str, ...],
        *,
        min_value: float | None = None,
        max_value: float | None = None,
        issue_type: str = "out_of_range",
        rule_suffix: str = "range",
    ) -> None:
        """校验指定数值字段是否落在闭区间范围内。

        Args:
            context: 表级规则执行上下文，包含当前表数据和问题列表。
            columns: 需要检测的字段名集合。
            min_value: 允许的最小值；为 None 时不校验下界。
            max_value: 允许的最大值；为 None 时不校验上界。
            issue_type: 范围检测失败时写入的问题类型编码。
            rule_suffix: 规则 ID 后缀，用于区分不同范围类规则。
        """
        df = context.df
        sheet_key = context.contract.local_sheet.lower()
        for column in columns:
            if column not in df.columns:
                continue
            missing = self._missing_mask(df[column])
            numeric = pd.to_numeric(df[column], errors="coerce")
            valid = (~missing) & numeric.notna()
            invalid = pd.Series(False, index=df.index)
            if min_value is not None:
                invalid |= valid & (numeric < min_value)
            if max_value is not None:
                invalid |= valid & (numeric > max_value)
            for idx in list(df.index[invalid])[: self.sample_limit]:
                if min_value is not None and max_value is not None:
                    message = f"{column} must be between {min_value} and {max_value}"
                elif min_value is not None:
                    message = f"{column} must be >= {min_value}"
                else:
                    message = f"{column} must be <= {max_value}"
                context.issues.append(
                    self._issue_factory(
                        config_name=context.config_name,
                        sub_node=context.sub_node,
                        sheet=context.contract.local_sheet,
                        column=column,
                        row_index=self._row_index(idx),
                        issue_type=issue_type,
                        severity="ERROR",
                        action="block",
                        message=message,
                        original_value=df.at[idx, column],
                        rule_id=f"{sheet_key}.{column}.{rule_suffix}",
                    )
                )

    def _check_date_order(
        self,
        context: _TableRuleContext,
        start_column: str,
        end_column: str,
    ) -> None:
        """校验结束日期字段不早于开始日期字段。

        Args:
            context: 表级规则执行上下文，包含当前表数据和问题列表。
            start_column: 开始日期字段名。
            end_column: 结束日期字段名。
        """
        df = context.df
        if start_column not in df.columns or end_column not in df.columns:
            return
        start_missing = self._missing_mask(df[start_column])
        end_missing = self._missing_mask(df[end_column])
        start = pd.to_datetime(df[start_column], errors="coerce")
        end = pd.to_datetime(df[end_column], errors="coerce")
        valid = (~start_missing) & (~end_missing) & start.notna() & end.notna()
        invalid = valid & (end < start)
        sheet_key = context.contract.local_sheet.lower()
        for idx in list(df.index[invalid])[: self.sample_limit]:
            context.issues.append(
                self._issue_factory(
                    config_name=context.config_name,
                    sub_node=context.sub_node,
                    sheet=context.contract.local_sheet,
                    row_index=self._row_index(idx),
                    issue_type="invalid_date_range",
                    severity="ERROR",
                    action="block",
                    message=f"{end_column} must be greater than or equal to {start_column}",
                    original_value={
                        start_column: df.at[idx, start_column],
                        end_column: df.at[idx, end_column],
                    },
                    rule_id=f"{sheet_key}.{start_column}_{end_column}.order",
                )
            )

    def _check_parseable_dates(
        self,
        context: _TableRuleContext,
        columns: tuple[str, ...],
    ) -> None:
        """校验指定字段值是否可解析为日期时间。

        Args:
            context: 表级规则执行上下文，包含当前表数据和问题列表。
            columns: 需要进行日期解析校验的字段名集合。
        """
        df = context.df
        sheet_key = context.contract.local_sheet.lower()
        for column in columns:
            if column not in df.columns:
                continue
            missing = self._missing_mask(df[column])
            parsed = pd.to_datetime(df[column], errors="coerce")
            invalid = (~missing) & parsed.isna()
            for idx in list(df.index[invalid])[: self.sample_limit]:
                context.issues.append(
                    self._issue_factory(
                        config_name=context.config_name,
                        sub_node=context.sub_node,
                        sheet=context.contract.local_sheet,
                        column=column,
                        row_index=self._row_index(idx),
                        issue_type="type_mismatch",
                        severity="ERROR",
                        action="block",
                        message=f"{column} must be parseable as a date",
                        original_value=df.at[idx, column],
                        rule_id=f"{sheet_key}.{column}.date_parse",
                    )
                )

    def _check_enum(
        self,
        context: _TableRuleContext,
        column: str,
        *,
        allowed: set[str],
        rule_id: str,
        message: str,
    ) -> None:
        """按忽略大小写的方式校验字符串字段是否属于允许值集合。

        Args:
            context: 表级规则执行上下文，包含当前表数据和问题列表。
            column: 需要进行枚举校验的字段名。
            allowed: 允许值集合，应使用小写标准值。
            rule_id: 枚举规则命中时写入的问题规则 ID。
            message: 枚举规则命中时写入的问题说明。
        """
        df = context.df
        if column not in df.columns:
            return
        values = df[column].astype("object")
        invalid = (~self._missing_mask(values)) & ~values.map(
            lambda value: str(value).strip().lower() in allowed
        )
        for idx in list(df.index[invalid])[: self.sample_limit]:
            context.issues.append(
                self._issue_factory(
                    config_name=context.config_name,
                    sub_node=context.sub_node,
                    sheet=context.contract.local_sheet,
                    column=column,
                    row_index=self._row_index(idx),
                    issue_type="invalid_enum",
                    severity="ERROR",
                    action="block_on_enforce",
                    message=message,
                    original_value=df.at[idx, column],
                    rule_id=rule_id,
                )
            )

    def _check_conflicting_values(
        self,
        context: _TableRuleContext,
        *,
        key_columns: tuple[str, ...],
        value_columns: tuple[str, ...],
    ) -> None:
        """检测同一业务键组合下是否存在互相冲突的字段取值。

        Args:
            context: 表级规则执行上下文，包含当前表数据和问题列表。
            key_columns: 用于分组识别同一业务对象的字段集合。
            value_columns: 在同一业务键下必须保持一致的字段集合。
        """
        df = context.df
        if df.empty or not set(key_columns + value_columns).issubset(df.columns):
            return
        sheet_key = context.contract.local_sheet.lower()
        grouped = df.groupby(list(key_columns), dropna=False)
        added = 0
        for key_values, group in grouped:
            for value_column in value_columns:
                values = {
                    str(value).strip()
                    for value in group[value_column]
                    if not self._missing_mask(pd.Series([value])).iloc[0]
                }
                if len(values) <= 1:
                    continue
                context.issues.append(
                    self._issue_factory(
                        config_name=context.config_name,
                        sub_node=context.sub_node,
                        sheet=context.contract.local_sheet,
                        column=value_column,
                        issue_type="duplicate_conflict",
                        severity="ERROR",
                        action="block",
                        message=(
                            f"{value_column} has conflicting values for key {key_values}"
                        ),
                        original_value=sorted(values),
                        rule_id=f"{sheet_key}.{value_column}.conflict",
                    )
                )
                added += 1
                if added >= self.sample_limit:
                    return

    @staticmethod
    def _row_index(idx: Any) -> int | str:
        """将 DataFrame 行索引规范化为问题明细可用的整数或字符串。

        Args:
            idx: 原始 DataFrame 行索引值。

        Returns:
            整数索引保持为 int，其余索引转换为 str。
        """
        return int(idx) if isinstance(idx, int) else str(idx)

    def _check_m6_wfr_vfr_numeric(self, context: _TableRuleContext) -> None:
        """校验 WFR/VFR 是否非空且为数值格式，不施加取值范围约束。

        Args:
            context: M6_TruckReleaseCon 表级规则执行上下文。
        """
        df = context.df
        for column in ("WFR", "VFR"):
            if column not in df.columns:
                continue
            missing = self._missing_mask(df[column])
            numeric = pd.to_numeric(df[column], errors="coerce")
            invalid = (~missing) & numeric.isna()
            for idx in list(df.index[invalid])[: self.sample_limit]:
                context.issues.append(
                    self._issue_factory(
                        config_name=context.config_name,
                        sub_node=context.sub_node,
                        sheet=context.contract.local_sheet,
                        column=column,
                        row_index=self._row_index(idx),
                        issue_type="type_mismatch",
                        severity="ERROR",
                        action="block",
                        message=f"{column} must be numeric",
                        original_value=df.at[idx, column],
                        rule_id="m6_truckreleasecon.wfr_vfr.numeric",
                    )
                )

class ConfigInputDataQualityChecker:
    """基于固定 schema 契约执行配置表输入数据质量检测。"""

    def __init__(
        self,
        mapping_config_path: str | Path | None = None,
        *,
        schema_config: dict[str, Any] | None = None,
        mode: str = "audit_only",
        fail_on_error: bool = False,
        sample_limit: int = 20,
        report_dir: str | Path | None = None,
        required_import_tables: list[str] | tuple[str, ...] | None = None,
        optional_import_tables: list[str] | tuple[str, ...] | None = None,
        quality_check_enabled: dict[str, bool] | None = None,
    ) -> None:
        """初始化配置表输入数据质量检测器。

        Args:
            mapping_config_path: 已废弃参数。配置表字段结构已固定在 Python
                schema 中，传入非空值会抛出异常。
            schema_config: 测试或特殊场景注入的 schema 配置；为空时读取
                ``pgsql_db.config_table_schema`` 中的固定 schema。
            mode: 数据质量检测模式，支持 ``off``、``audit_only`` 和
                ``enforce``。
            fail_on_error: 在非强制模式下是否遇到 ERROR 级问题即阻断。
            sample_limit: 单类问题最多记录的样例数量。
            report_dir: 默认质量检测报告输出目录。
            required_import_tables: 必需入库配置表名集合，缺失时按阻断处理。
            optional_import_tables: 非必需入库配置表名集合，缺失时按告警处理。
            quality_check_enabled: 按配置表 Sheet 名控制质量检测是否开启。

        Raises:
            ValueError: 当传入已废弃的 mapping_config_path 或不支持的 mode。
        """
        # 表字段、主键和类型等结构元数据由 Python schema 固定维护。
        if mapping_config_path is not None:
            raise ValueError(
                "mapping_config_path is no longer supported; config table fields "
                "are fixed in pgsql_db.config_table_schema"
            )
        self.mapping_config_path = None  # 已废弃的外部映射配置路径，固定为 None
        self._schema_config_override = schema_config  # 测试或特殊场景注入的 schema 覆盖配置
        self.mode = str(mode or "audit_only")  # 数据质量检测模式：off/audit_only/enforce
        if self.mode not in {"off", "audit_only", "enforce"}:
            raise ValueError(f"Unsupported data_quality mode: {self.mode}")
        self.fail_on_error = bool(fail_on_error)  # 非强制模式下遇 ERROR 级问题是否即阻断
        self.sample_limit = int(sample_limit)  # 单类问题最多记录的样例数量
        self.report_dir = Path(report_dir) if report_dir is not None else None  # 默认报告输出目录
        self.required_import_tables = {str(x) for x in (required_import_tables or [])}  # 必需入库表集合，缺失按阻断
        self.optional_import_tables = {str(x) for x in (optional_import_tables or [])}  # 非必需入库表集合，缺失按告警
        self.quality_check_enabled = {
            # 按 Sheet 名控制质量检测是否开启的开关映射
            str(table): self._coerce_bool(enabled)
            for table, enabled in (quality_check_enabled or {}).items()
        }

        self._raw_config = self._load_mapping_config()  # 加载后的原始固定 schema 映射
        self._contracts = self._load_contracts()  # 规范化后的表级入库契约列表
        self._contracts_by_sheet = {
            # 本地 Sheet 名到入库契约的索引，便于按 Sheet 快速查找
            contract.local_sheet: contract for contract in self._contracts
        }
        self._table_rules = ConfigTableQualityRules(  # 表级专属质量规则调度器
            issue_factory=self._issue,
            missing_mask=self._missing_mask,
            sample_limit=self.sample_limit,
        )

    @classmethod
    def from_defaults(
        cls,
        mapping_config_path: str | Path | None = None,
        *,
        schema_config: dict[str, Any] | None = None,
        report_dir: str | Path | None = None,
        force_fail_on_error: bool | None = None,
    ) -> "ConfigInputDataQualityChecker":
        """基于 ``config/defaults.yaml#data_quality`` 配置创建检测器。

        Args:
            mapping_config_path: 已废弃参数，仅保留调用兼容性。
            schema_config: 测试或特殊场景注入的 schema 配置。
            report_dir: 覆盖默认报告输出目录。
            force_fail_on_error: 覆盖 defaults 中的 fail_on_error 配置。

        Returns:
            已按 defaults 初始化的数据质量检测器实例。
        """
        try:
            from src.utils.defaults import DATA_QUALITY_CONFIG
        except Exception:
            DATA_QUALITY_CONFIG = {}

        cfg = dict(DATA_QUALITY_CONFIG or {})
        fail_on_error = cfg.get("fail_on_error", False)
        if force_fail_on_error is not None:
            fail_on_error = force_fail_on_error
        config_tables = cfg.get("config_tables") or {}

        return cls(
            mapping_config_path=mapping_config_path,
            schema_config=schema_config,
            mode=cfg.get("mode", "audit_only") if cfg.get("enabled", True) else "off",
            fail_on_error=fail_on_error,
            sample_limit=cfg.get("sample_limit", 20),
            report_dir=report_dir or cfg.get("report_dir"),
            required_import_tables=config_tables.get("required_import") or (),
            optional_import_tables=config_tables.get("optional_import") or (),
            quality_check_enabled=config_tables.get("quality_check_enabled") or {},
        )

    def _load_mapping_config(self) -> dict[str, Any]:
        """加载固定 schema 映射；测试场景可通过参数注入覆盖配置。

        Returns:
            包含 ``tables`` 节点的 schema 映射。

        Raises:
            ValueError: 当 schema 缺少合法的 ``tables`` 映射时抛出。
        """
        loaded = self._schema_config_override or get_config_table_mapping()
        if not isinstance(loaded.get("tables"), dict):
            raise ValueError(
                "Config table schema must contain a tables mapping"
            )
        return loaded

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """将配置值转换为布尔值。

        Args:
            value: 待转换的配置值，支持 bool、数值和常见字符串表示。

        Returns:
            转换后的布尔值。
        """
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes", "y", "on", "是"}

    def _load_contracts(self) -> list[_TableContract]:
        """将原始 schema 定义规范化为表级入库契约。

        Returns:
            按固定 schema 解析得到的配置表契约列表。
        """
        contracts: list[_TableContract] = []
        for key, table_cfg in self._raw_config.get("tables", {}).items():
            fields = tuple(table_cfg.get("fields") or ())
            local_sheet = str(table_cfg.get("local_sheet") or key)
            data_quality_enabled = bool(table_cfg.get("data_quality_enabled", True))
            if local_sheet in self.quality_check_enabled:
                data_quality_enabled = self.quality_check_enabled[local_sheet]
            contracts.append(
                _TableContract(
                    key=str(key),
                    local_sheet=local_sheet,
                    db_table=str(table_cfg.get("db_table") or key),
                    import_enabled=bool(table_cfg.get("import_enabled", True)),
                    data_quality_enabled=data_quality_enabled,
                    primary_key=self._primary_key_columns_from_config(table_cfg),
                    fields=fields,
                )
            )
        return contracts

    @staticmethod
    def _primary_key_columns_from_config(table_cfg: dict[str, Any]) -> tuple[str, ...]:
        """从表级配置和字段级标记中解析主键字段集合。

        Args:
            table_cfg: 单张配置表的 schema 定义。

        Returns:
            去重且保序的主键字段名元组。
        """
        columns: list[str] = []

        def add_column(column: Any) -> None:
            text = str(column).strip() if column is not None else ""
            if text and text not in columns:
                columns.append(text)

        table_primary_key = table_cfg.get("primary_key") or []
        if isinstance(table_primary_key, (list, tuple)):
            for column in table_primary_key:
                add_column(column)
        elif table_primary_key:
            add_column(table_primary_key)

        for field in table_cfg.get("fields") or []:
            if not isinstance(field, dict):
                continue
            db_name = field.get("db_name") or field.get("local_name")
            if db_name and is_primary_key_marker(field.get("primary_key")):
                add_column(db_name)

        return tuple(columns)

    def table_name_for_sheet(self, sheet_name: str) -> str:
        """根据本地 Sheet 名返回物理数据库表名。

        Args:
            sheet_name: 本地配置表 Sheet 名。

        Returns:
            对应的 cfg_* 物理表名。
        """
        return self._contracts_by_sheet[sheet_name].db_table

    def db_key_for_sheet(self, sheet_name: str) -> str:
        """根据本地 Sheet 名返回去除 cfg_ 前缀的数据库表 key。

        Args:
            sheet_name: 本地配置表 Sheet 名。

        Returns:
            去除 ``cfg_`` 前缀后的数据库表 key。
        """
        return self._contracts_by_sheet[sheet_name].db_key

    def validate(
        self,
        tables: dict[str, pd.DataFrame],
        *,
        config_name: str | None = None,
        sub_node: str = "input_pre",
        write_reports: bool = False,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """对输入配置表执行校验、字段投影和入库类型转换。

        Args:
            tables: 从 Excel/CSV 读取出的配置表数据，key 为 Sheet 名。
            config_name: 当前配置名，用于问题归属、报告输出和入库字段追加。
            sub_node: 当前检测节点标识，用于内部问题追踪。
            write_reports: 是否写出 ``input_quality.xlsx`` 质量报告。
            output_dir: 本次质量报告输出目录；为空时使用实例默认目录。

        Returns:
            包含检测结果、问题明细、清洗后表、待入库表和摘要统计的字典。
        """
        empty_result = {
            "passed": True,
            "blocked": False,
            "issues": [],
            "cleaned_tables": dict(tables),
            "db_ready_tables": {},
            "db_ready_by_db_key": {},
            "missing_sheets": [],
            "ignored_sheets": [],
            "ignored_columns": [],
            "summary": {"mode": self.mode, "errors": 0, "warnings": 0, "infos": 0},
        }
        if self.mode == "off":
            return empty_result

        issues: list[dict[str, Any]] = []
        missing_sheets: list[dict[str, Any]] = []
        ignored_sheets: list[dict[str, Any]] = []
        ignored_columns: list[dict[str, Any]] = []
        cleaned_tables: dict[str, pd.DataFrame] = {}
        db_ready_tables: dict[str, pd.DataFrame] = {}
        db_ready_by_db_key: dict[str, pd.DataFrame] = {}

        # 流程图步骤 0：读取数据配置表。
        # Excel/CSV 读取已由调用入口完成，本函数接收读取后的 tables 集合。
        mapped_sheets = {
            c.local_sheet for c in self._contracts if c.import_enabled or c.data_quality_enabled
        }
        # 非流程节点：未纳入固定 schema 的 Sheet 不参与投影、检测或入库。
        for sheet_name in sorted(set(tables) - mapped_sheets):
            ignored_sheets.append(
                {"sheet": sheet_name, "reason": "not_declared_in_fixed_schema", "action": "ignore"}
            )

        for contract in self._contracts:
            if not (contract.import_enabled or contract.data_quality_enabled):
                continue

            # 流程图步骤 1：检测核心 Sheet 是否存在。
            if contract.local_sheet not in tables:
                if not contract.data_quality_enabled:
                    continue
                severity, action = self._missing_sheet_policy(contract.local_sheet)
                # _missing_sheet_policy()：按必需/非必需入库表返回缺失 Sheet 的级别和动作。
                item = {
                    "sheet": contract.local_sheet,
                    "db_table": contract.db_table,
                    "reason": "mapped_sheet_missing",
                    "severity": severity,
                    "action": action,
                }
                missing_sheets.append(item)
                issues.append(
                    self._issue(
                        config_name=config_name,
                        sub_node=sub_node,
                        sheet=contract.local_sheet,
                        issue_type="missing_sheet",
                        severity=severity,
                        action=action,
                        message=f"Mapped sheet is missing: {contract.local_sheet}",
                        rule_id="sheet.missing_mapped",
                    )
                )
                # _issue()：构造统一的问题明细记录。
                continue

            source_df = tables[contract.local_sheet]
            if source_df is None:
                source_df = pd.DataFrame()

            # 流程图步骤 2-6：空表、结构、字段类型、全量去重、主键唯一性。
            projected, table_issues, table_ignored = self._validate_table(
                contract,
                source_df,
                config_name=config_name,
                sub_node=sub_node,
                run_quality_checks=contract.data_quality_enabled,
            )
            # _validate_table()：执行单表通用检测并返回只包含入库字段的数据。
            issues.extend(table_issues)
            ignored_columns.extend(table_ignored)
            cleaned_tables[contract.local_sheet] = projected

            # 非流程节点：DB-ready 构造只负责字段顺序、命名和类型落地；
            # 类型问题已在流程图步骤 4 记录，避免在此重复生成 issue。
            db_ready = self._build_db_ready_table(
                contract,
                projected,
                issues,
                config_name=config_name,
                sub_node=sub_node,
                record_conversion_issues=False,
            )
            # _build_db_ready_table()：构造字段名、顺序和类型符合数据库契约的入库表。
            db_ready_tables[contract.local_sheet] = db_ready
            db_ready_by_db_key[contract.db_key] = db_ready

        # 流程图步骤 4 补充：执行表级专项规则，例如非负数、枚举、日期区间。
        # 当前不做跨表引用缺失检测。
        self._table_rules.validate_all(
            cleaned_tables,
            [contract for contract in self._contracts if contract.data_quality_enabled],
            issues,
            config_name=config_name,
            sub_node=sub_node,
        )
        # _table_rules.validate_all()：按“一表一函数”执行表级专项业务规则。

        # 流程图步骤 7：是否通过质量检测。
        summary = self._summarize(issues, ignored_sheets, ignored_columns, missing_sheets)
        # _summarize()：汇总错误、告警、提示和忽略项数量。
        blocked = self._should_block(summary)
        # _should_block()：根据汇总结果判断是否阻断后续入库。
        result = {
            "passed": not blocked,
            "blocked": blocked,
            "issues": issues,
            "cleaned_tables": cleaned_tables,
            "db_ready_tables": db_ready_tables,
            "db_ready_by_db_key": db_ready_by_db_key,
            "missing_sheets": missing_sheets,
            "ignored_sheets": ignored_sheets,
            "ignored_columns": ignored_columns,
            "summary": summary,
        }

        report_target = Path(output_dir) if output_dir is not None else self.report_dir
        if write_reports and report_target is not None:
            # 流程图步骤 8：不通过时记录 error 报错；报告先写出，再由上层决定是否终止或继续入库。
            self.write_reports(result, report_target)
            # write_reports()：输出 input_quality.xlsx 的 issues 工作表。

        return result

    def _missing_sheet_policy(self, sheet_name: str) -> tuple[str, str]:
        """返回缺失必需/可选配置表时对应的严重级别和处理动作。

        Args:
            sheet_name: 缺失的本地配置表 Sheet 名。

        Returns:
            ``(severity, action)`` 元组。
        """
        if sheet_name in self.optional_import_tables:
            return "WARNING", "warn_keep"
        return "ERROR", "block_on_enforce"

    def validate_or_raise(self, *args, **kwargs) -> dict[str, Any]:
        """执行质量检测，并在出现阻断问题时抛出异常。

        Args:
            *args: 透传给 ``validate`` 的位置参数。
            **kwargs: 透传给 ``validate`` 的关键字参数。

        Returns:
            ``validate`` 返回的检测结果字典。

        Raises:
            DataQualityError: 当检测结果需要阻断后续流程时抛出。
        """
        result = self.validate(*args, **kwargs)
        if result["blocked"]:
            raise DataQualityError(
                "Input configuration data quality check failed; "
                f"errors={result['summary'].get('errors', 0)}"
            )
        return result

    def _validate_table(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        *,
        config_name: str | None,
        sub_node: str,
        run_quality_checks: bool = True,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        """按流程图顺序校验单张配置表并返回投影后的入库字段数据。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 原始 Sheet 数据。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
            run_quality_checks: 是否执行质量检测；关闭时仍执行字段投影。

        Returns:
            ``(projected, issues, ignored_columns)`` 三元组，其中 projected
            是只包含入库字段的数据，issues 是本表问题明细，ignored_columns
            是被忽略的非入库字段明细。

        Notes:
            单表内部检测顺序严格对应流程图：
            1. 是否空表；
            2. 数据结构校验；
            3. 字段类型校验；
            4. 数据全量去重；
            5. 检查主键唯一性。
        """
        issues: list[dict[str, Any]] = []
        ignored_columns: list[dict[str, Any]] = []
        expected_fields = contract.local_fields

        # 流程图步骤 3 补充：仅固定 schema 字段进入检测与入库链路；
        # 输入表多余列记录为 ignored column，不参与阻断。
        for col in df.columns:
            if str(col) not in expected_fields:
                ignored_columns.append(
                    {
                        "sheet": contract.local_sheet,
                        "column": str(col),
                        "reason": "not_declared_in_fixed_schema",
                        "action": "ignore",
                    }
                )

        present_fields = [name for name in expected_fields if name in df.columns]
        projected = df.loc[:, present_fields].copy()
        projected = self._drop_empty_mapped_rows(projected)
        # _drop_empty_mapped_rows()：删除所有入库字段均为空的填充行。

        if run_quality_checks:
            # 流程图步骤 2：是否空表。
            self._validate_empty_table(contract, projected, issues, config_name, sub_node)
            # _validate_empty_table()：检查投影后的配置表是否无有效数据行。

            # 流程图步骤 3：数据结构校验，检查固定入库列是否缺失。
            self._validate_missing_import_columns(
                contract, df, issues, config_name, sub_node
            )
            # _validate_missing_import_columns()：检查固定 schema 入库字段是否缺失。

            # 流程图步骤 4：字段类型校验，按空值、同字段多格式、db_type 转换依次执行。
            self._validate_all_field_nulls(contract, projected, issues, config_name, sub_node)
            # _validate_all_field_nulls()：检查所有入库字段的空值和空白字符串。
            self._validate_mixed_column_types(contract, projected, issues, config_name, sub_node)
            # _validate_mixed_column_types()：检查同一字段内是否存在多种输入格式。
            self._validate_db_type_conversion(contract, projected, issues, config_name, sub_node)
            # _validate_db_type_conversion()：按固定 db_type 检查字段是否可转换。

            # 流程图步骤 5：数据全量去重。
            projected = self._deduplicate_full_rows(
                contract, projected, issues, config_name, sub_node
            )
            # _deduplicate_full_rows()：记录并删除完全重复的入库字段行。

            # 流程图步骤 6：检查主键唯一性；主键空值先于重复组合判断。
            self._validate_primary_key_nulls(contract, projected, issues, config_name, sub_node)
            # _validate_primary_key_nulls()：检查主键字段是否为空。
            self._validate_primary_key_duplicates(
                contract, projected, issues, config_name, sub_node
            )
            # _validate_primary_key_duplicates()：检查主键组合是否唯一。
        return projected, issues, ignored_columns

    def _validate_missing_import_columns(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """按固定 schema 检查当前表缺失的入库字段。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 原始 Sheet 数据。
            issues: 质量问题明细列表；缺失入库字段时追加阻断问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        for field_name in contract.local_fields:
            if field_name not in df.columns:
                issues.append(
                    self._issue(
                        config_name=config_name,
                        sub_node=sub_node,
                        sheet=contract.local_sheet,
                        column=field_name,
                        issue_type="missing_import_column",
                        severity="ERROR",
                        action="block",
                        message=f"Mapped field is missing: {field_name}",
                        rule_id="field.missing_mapped",
                    )
                )

    def _drop_empty_mapped_rows(self, projected: pd.DataFrame) -> pd.DataFrame:
        """移除所有入库字段均为空的电子表格填充行。

        Args:
            projected: 已按固定 schema 投影后的入库字段数据。

        Returns:
            删除填充空行后的 DataFrame。
        """
        if projected.empty or len(projected.columns) == 0:
            return projected
        missing = pd.DataFrame(
            {column: self._missing_mask(projected[column]) for column in projected.columns},
            index=projected.index,
        )
        has_any_mapped_value = ~missing.all(axis=1)
        return projected.loc[has_any_mapped_value].copy()

    def _deduplicate_full_rows(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
    ) -> pd.DataFrame:
        """记录非阻断问题后删除完全重复的入库字段行。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 已投影并去除填充空行后的配置表数据。
            issues: 质量问题明细列表；完全重复行会追加 INFO 级问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。

        Returns:
            删除完全重复行后的 DataFrame。
        """
        if df.empty or len(df.columns) == 0:
            return df
        duplicate_mask = df.duplicated(keep="first")
        if not duplicate_mask.any():
            return df
        for idx in list(df.index[duplicate_mask])[: self.sample_limit]:
            issues.append(
                self._issue(
                    config_name=config_name,
                    sub_node=sub_node,
                    sheet=contract.local_sheet,
                    row_index=int(idx) if isinstance(idx, int) else str(idx),
                    issue_type="duplicate_value",
                    severity="INFO",
                    action="warn_deduplicate",
                    message="Full duplicate row is removed before primary key check",
                    original_value={
                        column: df.at[idx, column]
                        for column in df.columns
                    },
                    rule_id="row.full_duplicate",
                )
            )
        return df.loc[~duplicate_mask].copy()

    def _validate_db_type_conversion(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """按固定 db_type 提前执行字段类型转换校验。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 已投影并去除填充空行后的配置表数据。
            issues: 质量问题明细列表；类型转换失败时追加阻断问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        for field in self._ordered_fields(contract):
            local_name = str(field.get("local_name"))
            if local_name not in df.columns:
                continue
            self._convert_series(
                df[local_name],
                str(field.get("db_type") or "str"),
                sheet=contract.local_sheet,
                column=local_name,
                issues=issues,
                config_name=config_name,
                sub_node=sub_node,
                record_issues=True,
            )

    @staticmethod
    def _ordered_fields(contract: _TableContract) -> list[dict[str, Any]]:
        """按固定 schema 中的入库顺序返回字段定义。

        Args:
            contract: 当前配置表的固定入库契约。

        Returns:
            已按 ``db_order`` 或 ``local_order`` 排序的字段定义列表。
        """
        return sorted(
            contract.fields,
            key=lambda field: int(field.get("db_order") or field.get("local_order") or 0),
        )

    def _build_db_ready_table(
        self,
        contract: _TableContract,
        projected: pd.DataFrame,
        issues: list[dict[str, Any]],
        *,
        config_name: str | None,
        sub_node: str,
        record_conversion_issues: bool = True,
    ) -> pd.DataFrame:
        """按 schema 定义的字段顺序和类型构造待入库 DataFrame。

        Args:
            contract: 当前配置表的固定入库契约。
            projected: 已按入库字段投影后的配置表数据。
            issues: 质量问题明细列表；类型转换失败会追加问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
            record_conversion_issues: 是否记录类型转换失败问题。

        Returns:
            字段名、顺序和类型均符合数据库表契约的 DataFrame。
        """
        db_ready = pd.DataFrame(index=projected.index)
        for field in self._ordered_fields(contract):
            local_name = str(field.get("local_name"))
            db_name = str(field.get("db_name") or local_name)
            if local_name not in projected.columns:
                db_ready[db_name] = pd.Series(dtype="object")
                continue
            db_ready[db_name] = self._convert_series(
                projected[local_name],
                str(field.get("db_type") or "str"),
                sheet=contract.local_sheet,
                column=local_name,
                issues=issues,
                config_name=config_name,
                sub_node=sub_node,
                record_issues=record_conversion_issues,
            )
        return db_ready.reset_index(drop=True)

    def _convert_series(
        self,
        series: pd.Series,
        db_type: str,
        *,
        sheet: str,
        column: str,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
        record_issues: bool = True,
    ) -> pd.Series:
        """按 schema 的 db_type 转换字段值，并记录不可转换数据。

        Args:
            series: 待转换的字段值序列。
            db_type: 固定 schema 中声明的数据库字段类型。
            sheet: 当前字段所属的本地 Sheet 名。
            column: 当前字段名。
            issues: 质量问题明细列表；转换失败时追加问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
            record_issues: 是否记录类型转换失败问题。

        Returns:
            转换为目标数据库类型语义后的 pandas Series。
        """
        normalized_type = db_type.strip().lower()
        missing = self._missing_mask(series)

        if normalized_type in {"str", "text", "varchar", "character varying"}:
            converted = series.astype("object").copy()
            converted[missing] = None
            converted[~missing] = converted[~missing].map(
                lambda value: self._normalize_text_value(column, value)
            )
            return converted

        if normalized_type in {"bigint", "int", "integer"}:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = (~missing) & numeric.isna()
            fractional = (~missing) & numeric.notna() & ((numeric % 1) != 0)
            self._record_conversion_errors(
                series, invalid | fractional, issues, config_name, sub_node, sheet, column,
                record_issues=record_issues,
            )
            numeric[invalid | fractional] = pd.NA
            return numeric.astype("Int64")

        if normalized_type in {"double precision", "float", "float64", "numeric", "real"}:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = (~missing) & numeric.isna()
            self._record_conversion_errors(
                series, invalid, issues, config_name, sub_node, sheet, column,
                record_issues=record_issues,
            )
            return numeric.astype("float64")

        if normalized_type in {"bool", "boolean"}:
            return self._convert_bool(
                series, missing, issues, config_name, sub_node, sheet, column,
                record_issues=record_issues,
            )

        if "timestamp" in normalized_type or normalized_type == "date":
            converted = pd.to_datetime(series, errors="coerce")
            invalid = (~missing) & converted.isna()
            self._record_conversion_errors(
                series, invalid, issues, config_name, sub_node, sheet, column,
                record_issues=record_issues,
            )
            return converted

        converted = series.astype("object").copy()
        converted[missing] = None
        return converted

    def _convert_bool(
        self,
        series: pd.Series,
        missing: pd.Series,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
        sheet: str,
        column: str,
        record_issues: bool = True,
    ) -> pd.Series:
        """将常见布尔输入表示规范化为 pandas 可空布尔类型。

        Args:
            series: 待转换的字段值序列。
            missing: 标识空值或空白字符串的布尔掩码。
            issues: 质量问题明细列表；转换失败时追加问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
            sheet: 当前字段所属的本地 Sheet 名。
            column: 当前字段名。
            record_issues: 是否记录类型转换失败问题。

        Returns:
            pandas 可空布尔类型 Series。
        """
        truthy = {"true", "1", "y", "yes", "t"}
        falsy = {"false", "0", "n", "no", "f"}
        result = pd.Series(pd.NA, index=series.index, dtype="boolean")
        invalid = pd.Series(False, index=series.index)
        for idx, value in series.items():
            if missing.at[idx]:
                continue
            if isinstance(value, bool):
                result.at[idx] = value
                continue
            token = str(value).strip().lower()
            if token in truthy:
                result.at[idx] = True
            elif token in falsy:
                result.at[idx] = False
            else:
                invalid.at[idx] = True
        self._record_conversion_errors(
            series, invalid, issues, config_name, sub_node, sheet, column,
            record_issues=record_issues,
        )
        return result

    @staticmethod
    def _normalize_text_value(column: str, value: Any) -> str:
        """按字段业务语义规范化文本入库值。

        Args:
            column: 当前字段名。
            value: 原始字段值。

        Returns:
            规范化后的文本值。地点类字段纯数字补零至 4 位，物料类字段去除
            数值型 ``.0`` 后缀，其他文本字段执行普通字符串转换。
        """
        normalized_column = str(column).strip().lower()
        if normalized_column in {
            "location",
            "dps_location",
            "sending",
            "receiving",
            "sourcing",
        }:
            return normalize_location(value)
        if normalized_column in {"material", "from_material", "to_material"}:
            return normalize_material(value)
        return str(value)

    def _record_conversion_errors(
        self,
        series: pd.Series,
        mask: pd.Series,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
        sheet: str,
        column: str,
        record_issues: bool = True,
    ) -> None:
        """在启用问题记录时追加字段类型转换失败明细。

        Args:
            series: 原始字段值序列。
            mask: 标识转换失败位置的布尔掩码。
            issues: 质量问题明细列表；转换失败时追加问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
            sheet: 当前字段所属的本地 Sheet 名。
            column: 当前字段名。
            record_issues: 是否记录类型转换失败问题。
        """
        if not record_issues:
            return
        for idx in list(series.index[mask])[: self.sample_limit]:
            issues.append(
                self._issue(
                    config_name=config_name,
                    sub_node=sub_node,
                    sheet=sheet,
                    column=column,
                    row_index=int(idx) if isinstance(idx, int) else str(idx),
                    issue_type="type_mismatch",
                    severity="ERROR",
                    action="block",
                    message=f"Value cannot be converted to mapped db_type: {series.loc[idx]!r}",
                    original_value=series.loc[idx],
                    rule_id="field.type_mismatch",
                )
            )

    def _validate_empty_table(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """当投影后的入库表无有效数据行时记录阻断问题。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 已投影并去除填充空行后的配置表数据。
            issues: 质量问题明细列表；空表时追加阻断问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        if not df.empty:
            return
        issues.append(
            self._issue(
                config_name=config_name,
                sub_node=sub_node,
                sheet=contract.local_sheet,
                issue_type="empty_table",
                severity="ERROR",
                action="block",
                message=f"Mapped import table is empty: {contract.local_sheet}",
                rule_id="table.not_empty",
            )
        )

    def _validate_all_field_nulls(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """对所有入库字段中的空值或空白字符串记录阻断问题。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 已投影并去除填充空行后的配置表数据。
            issues: 质量问题明细列表；空值命中时追加阻断问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        for column in contract.local_fields:
            if column not in df.columns:
                continue
            mask = self._missing_mask(df[column])
            for idx in list(df.index[mask])[: self.sample_limit]:
                issues.append(
                    self._issue(
                        config_name=config_name,
                        sub_node=sub_node,
                        sheet=contract.local_sheet,
                        column=column,
                        row_index=int(idx) if isinstance(idx, int) else str(idx),
                        issue_type="null_value",
                        severity="ERROR",
                        action="block",
                        message=f"Mapped import field is empty: {column}",
                        original_value=df.at[idx, column],
                        rule_id="field.not_null",
                    )
                )

    def _validate_mixed_column_types(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """检测同一入库字段内是否存在多种输入格式类别。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 已投影并去除填充空行后的配置表数据。
            issues: 质量问题明细列表；多格式命中时追加阻断问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        for column in contract.local_fields:
            if column not in df.columns:
                continue
            if self._is_text_db_type(self._field_db_type(contract, column)):
                continue
            series = df[column]
            non_missing = series[~self._missing_mask(series)]
            if non_missing.empty:
                continue
            categories = self._column_input_format_categories(non_missing)
            if len(categories) <= 1:
                continue
            issues.append(
                self._issue(
                    config_name=config_name,
                    sub_node=sub_node,
                    sheet=contract.local_sheet,
                    column=column,
                    issue_type="mixed_column_type",
                    severity="ERROR",
                    action="block",
                    message=(
                        f"Mapped import field has mixed input formats: {column}; "
                        f"formats={sorted(categories)}"
                    ),
                    original_value=[
                        self._stringify(value)
                        for value in non_missing.head(self.sample_limit).tolist()
                    ],
                    rule_id="field.single_type",
                )
            )

    @staticmethod
    def _field_db_type(contract: _TableContract, column: str) -> str:
        """返回固定 schema 中字段对应的数据库类型。

        Args:
            contract: 当前配置表的固定入库契约。
            column: 本地入库字段名。

        Returns:
            字段对应的数据库类型；未声明时按字符串类型处理。
        """
        for field in contract.fields:
            if str(field.get("local_name")) == column:
                return str(field.get("db_type") or "str")
        return "str"

    @staticmethod
    def _is_text_db_type(db_type: str) -> bool:
        """判断数据库字段类型是否属于文本类型。

        Args:
            db_type: 固定 schema 中声明的数据库字段类型。

        Returns:
            True 表示字段入库目标类型为文本，False 表示非文本类型。
        """
        return db_type.strip().lower() in {
            "str",
            "text",
            "varchar",
            "character varying",
        }

    @staticmethod
    def _column_input_format_categories(series: pd.Series) -> set[str]:
        """返回非空字段值中出现的输入格式类别集合。

        Args:
            series: 已过滤空值后的字段值序列。

        Returns:
            字段值中出现的输入格式类别集合。
        """
        if is_bool_dtype(series):
            return {"bool:native"}
        if is_numeric_dtype(series):
            return {"number:native"}
        if is_datetime64_any_dtype(series):
            return {"datetime:native"}

        categories: set[str] = set()
        for value in series:
            categories.add(ConfigInputDataQualityChecker._value_input_format_category(value))
        return categories

    @staticmethod
    def _value_input_format_category(value: Any) -> str:
        """根据原生类型或可解析字符串格式识别单元格输入类别。

        Args:
            value: 单个单元格值。

        Returns:
            输入格式类别编码，例如 ``number:native`` 或 ``datetime:string``。
        """
        if isinstance(value, bool):
            return "bool:native"
        if isinstance(value, numbers.Number):
            return "number:native"
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return "datetime:native"

        text = str(value).strip()
        lower = text.lower()
        if lower in {"true", "false", "y", "n", "yes", "no", "t", "f"}:
            return "bool:string"
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
            return "number:string"
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}([ tT].*)?", text):
            return "datetime:string:yyyy-mm-dd"
        if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}([ tT].*)?", text):
            return "datetime:string:yyyy/mm/dd"
        if re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}([ tT].*)?", text):
            return "datetime:string:yyyy.mm.dd"
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}([ tT].*)?", text):
            return "datetime:string:mm/dd/yyyy"
        parsed = pd.to_datetime(pd.Series([text]), errors="coerce").iloc[0]
        if pd.notna(parsed):
            return "datetime:string:parseable"
        return "string"

    def _validate_primary_key_nulls(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """对配置为主键的字段空值记录阻断问题。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 已投影并去除填充空行后的配置表数据。
            issues: 质量问题明细列表；主键空值命中时追加阻断问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        for key_col in contract.primary_key:
            if key_col not in df.columns:
                continue
            mask = self._missing_mask(df[key_col])
            for idx in list(df.index[mask])[: self.sample_limit]:
                issues.append(
                    self._issue(
                        config_name=config_name,
                        sub_node=sub_node,
                        sheet=contract.local_sheet,
                        column=key_col,
                        row_index=int(idx) if isinstance(idx, int) else str(idx),
                        issue_type="null_value",
                        severity="ERROR",
                        action="block",
                        message=f"Primary key field is empty: {key_col}",
                        original_value=df.at[idx, key_col],
                        rule_id="primary_key.not_null",
                    )
                )

    def _validate_primary_key_duplicates(
        self,
        contract: _TableContract,
        df: pd.DataFrame,
        issues: list[dict[str, Any]],
        config_name: str | None,
        sub_node: str,
    ) -> None:
        """对主键组合重复的数据行记录阻断问题。

        Args:
            contract: 当前配置表的固定入库契约。
            df: 已投影并去除填充空行、完全重复行后的配置表数据。
            issues: 质量问题明细列表；主键重复命中时追加阻断问题。
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
        """
        keys = [col for col in contract.primary_key if col in df.columns]
        if not keys or df.empty:
            return
        dup_mask = df.duplicated(subset=keys, keep=False)
        for idx in list(df.index[dup_mask])[: self.sample_limit]:
            key_values = {col: df.at[idx, col] for col in keys}
            issues.append(
                self._issue(
                    config_name=config_name,
                    sub_node=sub_node,
                    sheet=contract.local_sheet,
                    row_index=int(idx) if isinstance(idx, int) else str(idx),
                    issue_type="duplicate_value",
                    severity="ERROR",
                    action="block",
                    message=f"Duplicate primary key: {key_values}",
                    original_value=key_values,
                    rule_id="primary_key.unique",
                )
            )

    @staticmethod
    def _missing_mask(series: pd.Series) -> pd.Series:
        """识别字段序列中的空值和空白字符串。

        Args:
            series: 待识别空值的字段值序列。

        Returns:
            标识空值位置的布尔 Series。
        """
        return series.isna() | series.map(
            lambda value: isinstance(value, str) and value.strip() == ""
        )

    def _issue(
        self,
        *,
        config_name: str | None,
        sub_node: str,
        sheet: str,
        issue_type: str,
        severity: str,
        action: str,
        message: str,
        column: str | None = None,
        row_index: int | str | None = None,
        original_value: Any = None,
        converted_value: Any = None,
        rule_id: str = "",
    ) -> dict[str, Any]:
        """构造各检测规则统一使用的内存问题明细记录。

        Args:
            config_name: 当前配置名，用于问题归属和报告输出。
            sub_node: 当前检测节点标识，用于内部问题追踪。
            sheet: 问题所属的本地 Sheet 名。
            issue_type: 问题类型编码。
            severity: 问题严重级别。
            action: 建议处理动作或阻断动作。
            message: 内部问题说明。
            column: 问题所属字段名；表级问题可为空。
            row_index: 问题样例所在的源数据行索引；汇总报告不输出。
            original_value: 问题命中的原始值。
            converted_value: 类型转换后的值；无转换结果时为空。
            rule_id: 命中的规则 ID。

        Returns:
            标准化内存问题明细字典。
        """
        return {
            "run_id": config_name or "",
            "sub_node": sub_node,
            "config_name": config_name or "",
            "sheet": sheet,
            "column": column or "",
            "row_index": "" if row_index is None else row_index,
            "issue_type": issue_type,
            "severity": severity,
            "action": action,
            "message": message,
            "original_value": self._stringify(original_value),
            "converted_value": self._stringify(converted_value),
            "rule_id": rule_id,
        }

    @staticmethod
    def _stringify(value: Any) -> str:
        """将报告字段值转换为空值安全的字符串。

        Args:
            value: 待转换的任意字段值。

        Returns:
            字符串化后的字段值；None 和 NaN 返回空字符串。
        """
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value)

    def _summarize(
        self,
        issues: list[dict[str, Any]],
        ignored_sheets: list[dict[str, Any]],
        ignored_columns: list[dict[str, Any]],
        missing_sheets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """汇总报告输出和阻断判断所需的问题数量指标。

        Args:
            issues: 质量问题明细列表。
            ignored_sheets: 被忽略的非 schema Sheet 明细列表。
            ignored_columns: 被忽略的非入库字段明细列表。
            missing_sheets: 缺失的 schema Sheet 明细列表。

        Returns:
            包含错误、告警、提示、阻断和忽略项数量的摘要字典。
        """
        errors = sum(1 for issue in issues if issue.get("severity") in {"ERROR", "CRITICAL"})
        warnings = sum(1 for issue in issues if issue.get("severity") == "WARNING")
        infos = sum(1 for issue in issues if issue.get("severity") == "INFO")
        hard_blocks = sum(1 for issue in issues if issue.get("action") == "block")
        return {
            "mode": self.mode,
            "fail_on_error": self.fail_on_error,
            "issues": len(issues),
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "hard_blocks": hard_blocks,
            "missing_sheets": len(missing_sheets),
            "ignored_sheets": len(ignored_sheets),
            "ignored_columns": len(ignored_columns),
        }

    def _should_block(self, summary: dict[str, Any]) -> bool:
        """根据执行模式和问题级别判断是否阻断后续流程。

        Args:
            summary: ``_summarize`` 生成的问题摘要。

        Returns:
            True 表示需要阻断后续流程，False 表示允许继续。
        """
        has_errors = int(summary.get("errors") or 0) > 0
        has_hard_blocks = int(summary.get("hard_blocks") or 0) > 0
        return has_hard_blocks or (
            has_errors and (self.fail_on_error or self.mode == "enforce")
        )

    def write_reports(self, result: dict[str, Any], output_dir: str | Path) -> None:
        """写出面向操作员的数据质量检测报告工作簿。

        Args:
            result: ``validate`` 返回的检测结果字典。
            output_dir: 报告输出目录。
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for legacy_name in (
            "input_quality_issues.csv",
            "input_quality_missing_sheets.csv",
            "input_quality_ignored_sheets.csv",
            "input_quality_ignored_columns.csv",
            "input_quality_report.csv",
        ):
            try:
                (output_path / legacy_name).unlink(missing_ok=True)
            except OSError:
                pass

        workbook_path = output_path / "input_quality.xlsx"
        issues_df = self._build_report_issues_frame(result["issues"])
        sheets = {"issues": issues_df}
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = "A2"
                if df.shape[1] > 0:
                    worksheet.auto_filter.ref = worksheet.dimensions
                for column_cells in worksheet.columns:
                    max_len = max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in column_cells
                    )
                    width = min(max(max_len + 2, 10), 60)
                    worksheet.column_dimensions[column_cells[0].column_letter].width = width

    def _build_report_issues_frame(self, issues: list[dict[str, Any]]) -> pd.DataFrame:
        """构造每张配置表一行的操作员问题汇总 DataFrame。

        Args:
            issues: 内存质量问题明细列表。

        Returns:
            面向操作员的 issues 工作表 DataFrame。
        """
        raw = pd.DataFrame(issues)
        if raw.empty:
            return pd.DataFrame(columns=_ISSUE_COLUMNS)

        raw["问题类型"] = raw["issue_type"].map(
            lambda value: _ISSUE_TYPE_LABELS_ZH.get(str(value), str(value))
        )

        rows: list[dict[str, Any]] = []
        for (config_name, sheet), group in raw.groupby(["config_name", "sheet"], dropna=False):
            rows.append(
                {
                    "配置名": config_name,
                    "配置表Sheet": sheet,
                    "字段名": self._join_unique(group.get("column")),
                    "问题类型": self._join_unique(group.get("问题类型")),
                    "问题代码": self._join_unique(group.get("issue_type")),
                    "严重级别": self._join_unique(group.get("severity")),
                    "处理动作": self._join_unique(group.get("action")),
                    "转换后值": self._join_unique(group.get("converted_value")),
                }
            )
        return pd.DataFrame(rows, columns=_ISSUE_COLUMNS)

    @staticmethod
    def _join_unique(series: pd.Series | None) -> str:
        """使用报告分隔符合并去重后的非空文本值。

        Args:
            series: 待合并的字段值序列；为空时返回空字符串。

        Returns:
            使用中文分号连接的去重非空文本。
        """
        if series is None:
            return ""
        values: list[str] = []
        for value in series:
            if pd.isna(value):
                continue
            text = str(value)
            if text == "" or text in values:
                continue
            values.append(text)
        return "；".join(values)

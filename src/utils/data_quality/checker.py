"""配置表输入数据质量检测模块。
本模块只负责“发现并记录问题”，不负责数据转换、数据清洗或流程阻断
检测规则由 ``pgsql_db.config_table_schema.CONFIG_TABLE_SCHEMAS`` 中的字段属性
驱动，包括 ``notnull``、``enumerate``、``range`` 和 ``date_flag``
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from pgsql_db.config_table_schema import get_config_table_schemas

logger = logging.getLogger(__name__)

# 单类问题最多记录的样例数量，避免异常数据过多时报告无限膨胀
SAMPLE_LIMIT = 20

# 质量报告issues工作表的固定列顺序，面向业务/操作人员展示
_ISSUE_COLUMNS = [
    "配置名",
    "配置表Sheet",
    "字段名",
    "检测项",
    "问题类型",
    "问题代码",
    "严重级别",
]
# 内部问题类型编码到报告中文标签的映射；未收录编码在报告中原样展示
_ISSUE_TYPE_LABELS_ZH = {
    "missing_sheet": "缺失Sheet",
    "missing_import_column": "缺失列",
    "empty_table": "空表",
    "null_value": "空值",
    "type_mismatch": "类型/格式不可解析",
    "out_of_range": "超出范围",
    "invalid_enum": "非法枚举",
    "invalid_date_range": "日期范围非法",
    "duplicate_value": "重复值",
    "checker_exception": "检测器内部异常",
}
# 检测步骤标识与检测方法名一一对应（check_0_prepare 为数据准备阶段），
# 报告"检测项"列直接展示该英文标识。
_CHECK_IDS = (
    "check_0_prepare",
    "check_1_sheets",
    "check_2_empty",
    "check_3_duplicates",
    "check_4_columns",
    "check_5_fields",
    "check_6_pkeys",
)

class ConfigTableQualityRules:
    """基于 schema 字段属性构建配置表字段级检测规则

    初始化时把 ``CONFIG_TABLE_SCHEMAS`` 解析成逐表规则配置，包括：
    必需列、主键列、字段规则（db_type/notnull/enumerate/range）以及
    date_flag 起止对。

    解析阶段会同步校验schema属性组合：
    - range 只能绑定数值类型
    - date_flag 只能绑定日期/时间类型
    - enumerate 必须是非空集合类配置
    不合法配置会被忽略并记录 warning，不会中断检测流程
    """

    # db_type 归类集合：用于选择字段解析方式，并校验schema属性是否匹配
    _INTEGER_DB_TYPES = {"bigint", "int", "integer"}
    _FLOAT_DB_TYPES = {"double precision", "float", "float64", "numeric", "real"}
    _NUMERIC_DB_TYPES = _INTEGER_DB_TYPES | _FLOAT_DB_TYPES
    _BOOL_DB_TYPES = {"bool", "boolean"}
    # 布尔字段允许的真/假字符串表示；比较时忽略大小写和首尾空白
    _BOOL_TRUTHY = {"true", "1", "y", "yes", "t"}
    _BOOL_FALSY = {"false", "0", "n", "no", "f"}

    @staticmethod
    def _is_datetime_db_type(db_type: str) -> bool:
        """判断 db_type 是否属于日期时间类型"""
        return "timestamp" in db_type or db_type == "date"

    @staticmethod
    def _missing_mask(series: pd.Series) -> pd.Series:
        """识别字段序列中的空值和空白字符串（空值判定规则）

        Args:
            series: 待识别空值的字段值序列

        Returns:
            标识空值位置的布尔 Series
        """
        # 除 NaN/None 外，纯空白字符串同样按空值处理
        # map 在 object 列上返回 object dtype，显式转 bool 以避免
        # pandas 对 bool 与 object 间逻辑运算的弃用告警
        return series.isna() | series.map(
            lambda value: isinstance(value, str) and value.strip() == ""
        ).astype(bool)

    def __init__(self, schemas: dict[str, Any] | None = None) -> None:
        """解析 schema 并构建逐表规则配置。

        Args:
            schemas: 测试或特殊场景注入的 schema；为空时读取
                ``CONFIG_TABLE_SCHEMAS``，兼容带 ``tables`` 节点的完整映射
        """
        raw = get_config_table_schemas() if schemas is None else schemas
        # 兼容get_config_table_mapping()风格的完整映射注入
        if isinstance(raw.get("tables"), dict):
            raw = raw["tables"]
        self._tables: dict[str, dict[str, Any]] = {}
        for key, table_cfg in raw.items():
            sheet = str(table_cfg.get("local_sheet") or key)
            self._tables[sheet] = self._parse_table(sheet, table_cfg)

    @classmethod
    def _parse_table(cls, sheet: str, table_cfg: dict[str, Any]) -> dict[str, Any]:
        """解析单张配置表的 schema 定义为规则配置

        Args:
            sheet: 本地配置表Sheet名
            table_cfg: 该表的schema定义

        Returns:
            包含 db_table、columns、primary_key、fields、date_pair的规则配置
        """
        fields: dict[str, dict[str, Any]] = {}
        date_flags: dict[str, str] = {}
        primary_key: list[str] = []

        # 来源一：表级 primary_key 列表（保序去重）。
        for column in table_cfg.get("primary_key") or []:
            text = str(column).strip()
            if text and text not in primary_key:
                primary_key.append(text)

        for field in table_cfg.get("fields") or ():
            if not isinstance(field, dict):
                continue
            column = str(field.get("local_name"))
            db_type = str(field.get("db_type") or "str").strip().lower()
            rule: dict[str, Any] = {
                "db_type": db_type,
                "notnull": bool(field.get("notnull")),
            }

            # 来源二：字段级 primary_key 标记，补充表级未覆盖的主键字段。
            if field.get("primary_key") is True and column not in primary_key:
                primary_key.append(column)

            # enumerate：须为非空列表/元组/集合，统一小写去空白后存为集合
            allowed = field.get("enumerate")
            if allowed is not None:
                if isinstance(allowed, (list, tuple, set, frozenset)) and allowed:
                    rule["enumerate"] = {str(v).strip().lower() for v in allowed}
                else:
                    logger.warning(
                        "%s.%s: invalid enumerate %r ignored", sheet, column, allowed
                    )

            # range：须配数值 db_type，且为 (min, max) 二元组（None 表示无界）
            value_range = field.get("range")
            if value_range is not None:
                if (
                    db_type in cls._NUMERIC_DB_TYPES
                    and isinstance(value_range, (list, tuple))
                    and len(value_range) == 2
                ):
                    rule["range"] = (value_range[0], value_range[1])
                else:
                    logger.warning(
                        "%s.%s: range %r requires a numeric db_type; ignored",
                        sheet,
                        column,
                        value_range,
                    )

            # date_flag：须配日期 db_type，且取值为 start/end
            date_flag = field.get("date_flag")
            if date_flag is not None:
                if cls._is_datetime_db_type(db_type) and str(date_flag) in {"start", "end"}:
                    date_flags[column] = str(date_flag)
                else:
                    logger.warning(
                        "%s.%s: date_flag %r requires a datetime db_type; ignored",
                        sheet,
                        column,
                        date_flag,
                    )

            fields[column] = rule

        # date_flag起止对：有且仅有一个start和一个end时才生效
        starts = [col for col, flag in date_flags.items() if flag == "start"]
        ends = [col for col, flag in date_flags.items() if flag == "end"]
        date_pair: tuple[str, str] | None = None
        if len(starts) == 1 and len(ends) == 1:
            date_pair = (starts[0], ends[0])
        elif date_flags:
            logger.warning(
                "%s: date_flag must form exactly one start/end pair; got %r",
                sheet,
                date_flags,
            )

        return {
            "db_table": str(table_cfg.get("db_table") or sheet),
            "columns": tuple(fields),
            "primary_key": tuple(primary_key),
            "fields": fields,
            "date_pair": date_pair,
        }

    def sheet_names(self) -> list[str]:
        """返回 schema 中声明的全部配置表Sheet名"""
        return list(self._tables)

    def has_sheet(self, sheet: str) -> bool:
        """判断某个 Sheet 是否在 schema 中声明"""
        return sheet in self._tables

    def required_columns(self, sheet: str) -> list[str]:
        """返回某张表 schema 声明的必需字段名列表"""
        return list(self._tables[sheet]["columns"])

    def primary_key(self, sheet: str) -> tuple[str, ...]:
        """返回某张表的主键字段名元组"""
        return self._tables[sheet]["primary_key"]

    def db_table(self, sheet: str) -> str:
        """返回某张表对应的物理数据库表名"""
        return self._tables[sheet]["db_table"]

    def validate_fields(
        self,
        sheet: str,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """对单张表执行全部字段级规则

        执行顺序固定为：非空检查、类型检查、枚举检查、范围检查、日期顺序检查
        缺列问题由 ``check_4_columns`` 统一报告，本函数只校验已经存在
        的字段值

        Args:
            sheet: 本地配置表Sheet名（用于查询该表的规则配置）
            df: 已投影到 schema 声明列并去除填充空行的数据

        Returns:
            轻量命中明细列表；Sheet 未在 schema 声明时返回空列表。
            问题归属（config_name/sheet/check）与标准记录格式由调用方
            （Checker）补全，本类不依赖任何流程上下文。
        """
        # 获取当前 Sheet 的规则配置；未知 Sheet 不参与字段级检测
        table = self._tables.get(sheet)
        if table is None:
            return []

        # 准备问题收集器和主键集合，后续非空检查需要区分主键字段
        issues: list[dict[str, Any]] = []
        pk_columns = set(table["primary_key"])

        # 按 schema 字段声明顺序逐列校验。
        for column, rule in table["fields"].items():
            # 缺失字段由 check_4 统一报告，避免同一缺列重复产生多类问题
            if column not in df.columns:
                continue

            # 取出当前字段值，并复用同一份缺失掩码，保持各规则口径一致
            series = df[column]
            missing = self._missing_mask(series)

            # 非空规则只在 schema 声明 notnull=True 时执行
            if rule["notnull"]:
                issues.extend(
                    self.validate_notnull(
                        column, series, missing,
                        is_primary_key=column in pk_columns,
                    )
                )

            # 类型规则按 db_type 解析非空值；空值留给非空规则处理
            issues.extend(
                self.validate_type(
                    column, series, missing,
                    db_type=rule["db_type"],
                )
            )

            # 枚举规则只在 schema 声明 enumerate 时执行。
            if "enumerate" in rule:
                issues.extend(
                    self.validate_enum(
                        column, series, missing,
                        allowed=rule["enumerate"],
                    )
                )

            # 范围规则只在 schema 声明 range 时执行。
            if "range" in rule:
                issues.extend(
                    self.validate_range(
                        column, series, missing,
                        min_value=rule["range"][0],
                        max_value=rule["range"][1],
                    )
                )

        # 表级日期起止字段成对存在时，额外校验 start <= end。
        if table["date_pair"] is not None:
            issues.extend(self.validate_date_order(sheet, df))
        return issues

    @staticmethod
    def validate_notnull(
        column: str,
        series: pd.Series,
        missing: pd.Series,
        *,
        is_primary_key: bool = False,
    ) -> list[dict[str, Any]]:
        """校验字段非空。

        ``missing`` 由调用方统一计算，已把 ``NaN``、``None`` 和纯空白字符串
        都视为缺失值。

        Args:
            column: 字段名。
            series: 字段值序列。
            missing: 空值掩码（由调用方统一计算复用）。
            is_primary_key: 是否主键字段，主键空值使用独立规则 ID。

        Returns:
            轻量命中明细列表（最多 SAMPLE_LIMIT 条样例）；问题归属与
            标准记录格式由调用方（Checker）补全。
        """
        # 初始化命中列表；本函数只返回当前字段的非空命中。
        issues: list[dict[str, Any]] = []

        # 提取命中缺失掩码的行索引，并限制样例数量。
        for idx in list(series.index[missing])[:SAMPLE_LIMIT]:
            # 构造轻量命中；主键字段使用更明确的规则 ID 和提示。
            issues.append(
                {
                    "column": column,
                    "row_index": idx,
                    "issue_type": "null_value",
                    "severity": "ERROR",
                    "message": (
                        f"Primary key field is empty: {column}"
                        if is_primary_key
                        else f"Mapped import field is empty: {column}"
                    ),
                    "original_value": series.loc[idx],
                    "rule_id": (
                        "primary_key.not_null" if is_primary_key else "field.not_null"
                    ),
                }
            )
        return issues

    @classmethod
    def validate_type(
        cls,
        column: str,
        series: pd.Series,
        missing: pd.Series,
        *,
        db_type: str,
    ) -> list[dict[str, Any]]:
        """按 db_type 解析字段值，并报告不可解析的非空值。

        数值类型用 ``pd.to_numeric``（整数类型额外检查小数部分）、日期
        类型用 ``pd.to_datetime``、布尔类型按常见真假值字面量识别；文本
        及未识别类型不做检查。

        Args:
            column: 字段名。
            series: 字段值序列。
            missing: 空值掩码（空值由非空规则负责，类型规则跳过）。
            db_type: schema 声明的数据库字段类型（已小写规整）。

        Returns:
            轻量命中明细列表（最多 SAMPLE_LIMIT 条样例）；问题归属与
            标准记录格式由调用方（Checker）补全。
        """
        # 根据 db_type 选择解析策略，并生成 invalid 掩码。
        if db_type in cls._NUMERIC_DB_TYPES:
            # 数值类型：无法解析为数字的非空值判定为类型错误。
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = (~missing) & numeric.isna()

            # 整数类型：在可解析为数字的基础上，额外要求没有小数部分。
            if db_type in cls._INTEGER_DB_TYPES:
                invalid |= (~missing) & numeric.notna() & ((numeric % 1) != 0)
            message = f"{column} must be {db_type} compatible"
        elif cls._is_datetime_db_type(db_type):
            # 日期/时间类型：无法被 pandas 解析成日期的非空值判定为错误。
            parsed = pd.to_datetime(series, errors="coerce")
            invalid = (~missing) & parsed.isna()
            message = f"{column} must be parseable as a date"
        elif db_type in cls._BOOL_DB_TYPES:
            # 布尔类型：允许原生 bool，以及配置导入中常见的真假字符串。
            invalid = (~missing) & ~series.map(
                lambda value: isinstance(value, bool)
                or str(value).strip().lower()
                in (cls._BOOL_TRUTHY | cls._BOOL_FALSY)
            ).astype(bool)
            message = f"{column} must be a boolean literal"
        else:
            # 文本及未识别类型不做格式解析，避免误报自由文本字段。
            return []

        # 把 invalid 掩码转换为轻量命中明细，并限制样例数量。
        issues: list[dict[str, Any]] = []
        for idx in list(series.index[invalid])[:SAMPLE_LIMIT]:
            issues.append(
                {
                    "column": column,
                    "row_index": idx,
                    "issue_type": "type_mismatch",
                    "severity": "ERROR",
                    "message": message,
                    "original_value": series.loc[idx],
                    "rule_id": "field.type_mismatch",
                }
            )
        return issues

    @staticmethod
    def validate_enum(
        column: str,
        series: pd.Series,
        missing: pd.Series,
        *,
        allowed: set[str],
    ) -> list[dict[str, Any]]:
        """校验字段值是否属于schema声明的枚举集合。
        校验时会忽略大小写和首尾空白；空值由非空规则处理，枚举规则跳过。
        Args:
            column: 字段名。
            series: 字段值序列。
            missing: 空值掩码（空值由非空规则负责，枚举规则跳过）。
            allowed: 允许值集合（已小写规整）。
        Returns:
            轻量命中明细列表（最多 SAMPLE_LIMIT 条样例）；问题归属与
            标准记录格式由调用方（Checker）补全。
        """
        # 将非空值标准化为小写去空白文本，再与允许值集合比较。
        invalid = (~missing) & ~series.map(
            lambda value: str(value).strip().lower() in allowed
        ).astype(bool)

        # 把非法枚举值转换为轻量命中明细，并限制样例数量。
        issues: list[dict[str, Any]] = []
        for idx in list(series.index[invalid])[:SAMPLE_LIMIT]:
            issues.append(
                {
                    "column": column,
                    "row_index": idx,
                    "issue_type": "invalid_enum",
                    "severity": "ERROR",
                    "message": f"{column} must be one of {sorted(allowed)}",
                    "original_value": series.loc[idx],
                    "rule_id": "field.enum",
                }
            )
        return issues

    @staticmethod
    def validate_range(
        column: str,
        series: pd.Series,
        missing: pd.Series,
        *,
        min_value: float | None,
        max_value: float | None,
    ) -> list[dict[str, Any]]:
        """校验数值字段是否落在 schema 声明的闭区间范围内。
        ``None`` 表示对应一侧无边界。无法解析为数值的内容由类型规则报告，
        本函数只检查已经能够解析为数值的非空值。

        Args:
            column: 字段名。
            series: 字段值序列。
            missing: 空值掩码；范围判断只针对能解析为数值的有效值。
            min_value: 允许的最小值；为 None 时不校验下界。
            max_value: 允许的最大值；为 None 时不校验上界。

        Returns:
            轻量命中明细列表（最多 SAMPLE_LIMIT 条样例）；问题归属与
            标准记录格式由调用方（Checker）补全。
        """
        # 先尝试解析为数值；不可解析值不在本函数重复报错。
        numeric = pd.to_numeric(series, errors="coerce")

        # 仅保留非空且可解析为数值的行作为范围检查对象。
        valid = (~missing) & numeric.notna()

        # 初始化越界掩码，再分别叠加下界和上界条件。
        invalid = pd.Series(False, index=series.index)
        if min_value is not None:
            invalid |= valid & (numeric < min_value)
        if max_value is not None:
            invalid |= valid & (numeric > max_value)

        # 根据上下界配置生成清晰的操作员提示。
        if min_value is not None and max_value is not None:
            message = f"{column} must be between {min_value} and {max_value}"
        elif min_value is not None:
            message = f"{column} must be >= {min_value}"
        else:
            message = f"{column} must be <= {max_value}"

        # 把越界样例转换为轻量命中明细。
        issues: list[dict[str, Any]] = []
        for idx in list(series.index[invalid])[:SAMPLE_LIMIT]:
            issues.append(
                {
                    "column": column,
                    "row_index": idx,
                    "issue_type": "out_of_range",
                    "severity": "ERROR",
                    "message": message,
                    "original_value": series.loc[idx],
                    "rule_id": "field.range",
                }
            )
        return issues

    def validate_date_order(
        self,
        sheet: str,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """校验 date_flag 标记的起止字段满足 ``start <= end``。

        空值和日期格式问题分别由非空规则、类型规则报告；本函数只负责在
        两端都能解析为日期时比较先后顺序。

        Args:
            sheet: 本地配置表 Sheet 名（用于查询起止字段规则配置）。
            df: 已投影的配置表数据。

        Returns:
            轻量命中明细列表（最多 SAMPLE_LIMIT 条样例）；问题归属与
            标准记录格式由调用方（Checker）补全。
        """
        # 读取当前表的 date_flag 起止字段配置；未配置则无需检查。
        table = self._tables.get(sheet)
        if table is None or table["date_pair"] is None:
            return []

        # 拆出 start/end 字段名。
        start_column, end_column = table["date_pair"]

        # 任一端缺列时无法比较，缺列问题交由 check_4 报告。
        if start_column not in df.columns or end_column not in df.columns:
            return []

        # 分别计算起止字段的缺失掩码。
        start_missing = self._missing_mask(df[start_column])
        end_missing = self._missing_mask(df[end_column])

        # 尝试解析起止日期；解析失败的行由类型规则负责。
        start = pd.to_datetime(df[start_column], errors="coerce")
        end = pd.to_datetime(df[end_column], errors="coerce")

        # 只比较两端均非空且均可解析为日期的行。
        valid = (~start_missing) & (~end_missing) & start.notna() & end.notna()

        # 结束日期早于开始日期即为非法日期范围。
        invalid = valid & (end < start)

        # 把非法日期范围样例转换为轻量命中明细（表级命中不带 column）。
        issues: list[dict[str, Any]] = []
        for idx in list(df.index[invalid])[:SAMPLE_LIMIT]:
            issues.append(
                {
                    "row_index": idx,
                    "issue_type": "invalid_date_range",
                    "severity": "ERROR",
                    "message": (
                        f"{end_column} must be greater than or equal to {start_column}"
                    ),
                    "original_value": {
                        start_column: df.at[idx, start_column],
                        end_column: df.at[idx, end_column],
                    },
                    "rule_id": "field.date_order",
                }
            )
        return issues


class ConfigInputDataQualityChecker:
    """配置表输入数据质量检测器（六步检测，只记录问题不阻断）。
    实例属性只有 ``issues``（问题明细列表）和 ``result``（检测结果汇总）；
    规则配置、逐表开关等均在 ``validate`` 内部按需构建，不落为实例状态。
    """

    def __init__(self) -> None:
        self.issues: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}

    #  流程与记录工具方法（静态，无实例状态；标准 issue 由 Checker 统一构造）

    @staticmethod
    def _stringify(value: Any) -> str:
        """将问题明细字段值转换为空值安全的字符串。
        Args:
            value: 待转换的任意字段值。
        Returns:
            字符串化后的字段值；None 和 NaN 返回空字符串。
        """
        # None 与浮点 NaN 统一展示为空字符串，避免出现 "nan" 字样。
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value)

    @staticmethod
    def _normalize_row_index(idx: Any) -> int | str:
        """将 DataFrame 行索引规范化为问题明细可用的整数或字符串。"""
        # 非整数索引（如字符串索引）统一转字符串，保证可序列化。
        return int(idx) if isinstance(idx, int) else str(idx)

    @staticmethod
    def _join_unique(series: pd.Series | None) -> str:
        """使用中文分号合并去重后的非空文本值，保持首次出现顺序。"""
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

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """将配置值转换为布尔值，支持 bool、数值和常见字符串表示。"""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes", "y", "on", "是"}

    @staticmethod
    def _load_quality_check_enabled() -> dict[str, bool]:
        """读取 defaults.yaml#data_quality 中的逐表质量检测开关。
        Returns:
            Sheet 名到开关布尔值的映射；defaults 不可用时返回空映射
            （未声明的表默认开启检测）。
        """
        try:
            from src.utils.defaults import data_quality_config
        except ImportError:
            logger.warning(
                "defaults module unavailable; quality check enabled for all sheets"
            )
            return {}
        cfg = dict(data_quality_config or {})
        raw = (cfg.get("config_tables") or {}).get("quality_check_enabled") or {}
        return {
            str(sheet): ConfigInputDataQualityChecker._coerce_bool(enabled)
            for sheet, enabled in raw.items()
        }

    @staticmethod
    def _load_optional_sheets() -> set[str]:
        """读取 defaults.yaml#data_quality 的可选表清单（optional_import）。

        可选表在输入中存在时才执行检测；缺失时不做任何检测、不记缺表
        问题。未列入该清单的 schema 表一律按必需表处理（缺失记 ERROR），
        防止清单遗漏导致漏检。defaults 不可用时返回空集合。

        Returns:
            可选配置表 Sheet 名集合。
        """
        try:
            from src.utils.defaults import data_quality_config
        except ImportError:
            logger.warning(
                "defaults module unavailable; all sheets treated as required"
            )
            return set()
        cfg = dict(data_quality_config or {})
        raw = (cfg.get("config_tables") or {}).get("optional_import") or ()
        return {str(sheet) for sheet in raw}

    @staticmethod
    def _drop_empty_rows(
        df: pd.DataFrame, rules: ConfigTableQualityRules
    ) -> pd.DataFrame:
        """移除所有列均为空的电子表格填充行（仅用于检测口径）。

        空值判定复用规则实例的 ``_missing_mask``，保证检测口径与字段级
        规则的空值定义一致；本方法只构造检测视图，不修改调用方数据。
        """
        if df.empty or len(df.columns) == 0:
            return df
        missing = pd.DataFrame(
            {column: rules._missing_mask(df[column]) for column in df.columns},
            index=df.index,
        )
        return df.loc[~missing.all(axis=1)].copy()

    # pylint: disable=too-many-arguments
    @staticmethod
    def _build_issue(
        *,
        config_name: str | None,
        sheet: str,
        check: str,
        issue_type: str,
        severity: str,
        message: str,
        column: str | None = None,
        row_index: Any = None,
        original_value: Any = None,
        rule_id: str = "",
    ) -> dict[str, Any]:
        """构造各检测规则统一使用的内存问题明细记录。

        Args:
            config_name: 当前配置名，用于问题归属和报告输出。
            sheet: 问题所属的配置表 Sheet 名。
            check: 命中问题的检测步骤标识，与检测方法名一致
                （见 ``_CHECK_IDS``：check_0_prepare ~ check_6_pkeys）。
            issue_type: 问题类型编码（见 ``_ISSUE_TYPE_LABELS_ZH``）。
            severity: 严重级别（INFO/WARNING/ERROR）。
            message: 问题说明。
            column: 问题字段名；表级问题可不带。
            row_index: 问题行索引，接受原始 DataFrame 索引值并在此统一
                规范化；表级问题可不带。
            original_value: 原始取值快照。
            rule_id: 命中的规则 ID。

        Returns:
            标准化内存问题明细字典。
        """
        return {
            "config_name": config_name or "",
            "sheet": sheet,
            "column": column or "",
            # 行索引在此统一规范化，规则层只需透传原始索引值。
            "row_index": (
                ""
                if row_index is None
                else ConfigInputDataQualityChecker._normalize_row_index(row_index)
            ),
            "check": check,
            "issue_type": issue_type,
            "severity": severity,
            "message": message,
            "original_value": ConfigInputDataQualityChecker._stringify(original_value),
            "rule_id": rule_id,
        }

    def validate(
        self,
        tables: dict[str, pd.DataFrame],
        *,
        config_name: str | None = None,
        report_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """对输入配置表执行完整的数据质量检测流程。

        只有 ``defaults.yaml#data_quality.config_tables.quality_check_enabled``
        中为 true（或未声明）的表才参与检测；检测不修改调用方数据，也不
        阻断后续流程。

        Args:
            tables: 从 Excel/CSV 读取出的配置表数据，key 为 Sheet 名。
            config_name: 当前配置名，用于问题归属和报告输出。
            report_dir: 质量报告输出目录；为空时不写报告。

        Returns:
            检测结果字典：``{"passed": 无 ERROR 即 True, "issues": [...],
            "summary": {...}}``，同时存于 ``self.result``。
        """
        # 重置本次检测状态，避免复用实例时混入上一次结果。
        self.issues = []
        self.result = {}

        # 加载 schema 规则；字段、主键、枚举、范围等检测依据均来自这里。
        rules = ConfigTableQualityRules()

        # 读取逐表检测开关；未配置的 Sheet 默认启用检测。
        enabled = self._load_quality_check_enabled()

        # 生成本次实际参与检测的 Sheet 列表；关闭的表整体跳过。
        enabled_sheets = [
            sheet for sheet in rules.sheet_names() if enabled.get(sheet, True)
        ]

        # 可选表清单（optional_import）：存在时才检测，缺失时不报缺表问题。
        optional_sheets = self._load_optional_sheets()

        # 识别输入中存在、但 schema 未声明的 Sheet；这类 Sheet 不参与检测。
        undeclared = sorted(set(tables) - set(rules.sheet_names()))
        if undeclared:
            logger.info("Sheets not declared in schema are ignored: %s", undeclared)

        # 准备检测口径数据。
        # - 只保留 schema 声明且输入实际存在的列。
        # - 删除电子表格常见的全空填充行。
        # - 不修改调用方传入的原始 DataFrame。
        prepared: dict[str, pd.DataFrame] = {}
        for sheet in enabled_sheets:
            df = tables.get(sheet)

            # 缺失 Sheet 由 check_1 统一报告，这里只准备已存在的 Sheet。
            if df is None:
                continue
            try:
                # 投影到已存在的 schema 列，避免字段级检测处理无关列。
                present = [c for c in rules.required_columns(sheet) if c in df.columns]

                # 去掉全空行，避免 Excel/CSV 末尾填充行被误判为数据问题。
                prepared[sheet] = self._drop_empty_rows(
                    df.loc[:, present].copy(), rules
                )
            except Exception as exc:  # noqa: BLE001
                # 单表准备失败时记录内部异常，并继续处理其他 Sheet。
                logger.exception("Data quality preparation failed for sheet %s", sheet)
                self.issues.append(
                    self._internal_error_issue(
                        config_name=config_name,
                        sheet=sheet,
                        check="check_0_prepare",
                        exc=exc,
                    )
                )

        # 声明六个检测项的固定执行顺序。
        # 每个检测项只负责一种问题类型，便于报告定位和后续扩展。
        steps: tuple[tuple[str, Callable[[], None]], ...] = (
            # 检查 schema 声明且已启用的 Sheet 是否存在于输入数据中。
            (
                "check_1_sheets",
                lambda: self.check_1_sheets(
                    tables,
                    rules,
                    enabled_sheets=enabled_sheets,
                    optional_sheets=optional_sheets,
                    config_name=config_name,
                ),
            ),
            # 检查完成字段投影和空行过滤后的 Sheet 是否没有有效数据。
            (
                "check_2_empty",
                lambda: self.check_2_empty(prepared, rules, config_name=config_name),
            ),
            # 检查同一 Sheet 内是否存在全字段完全相同的重复数据行。
            (
                "check_3_duplicates",
                lambda: self.check_3_duplicates(
                    prepared, rules, config_name=config_name
                ),
            ),
            # 检查原始输入 Sheet 是否缺少 schema 声明的必需字段。
            (
                "check_4_columns",
                lambda: self.check_4_columns(
                    tables, rules, enabled_sheets=enabled_sheets, config_name=config_name
                ),
            ),
            # 检查字段级规则，包括非空、类型、枚举、范围和日期顺序。
            (
                "check_5_fields",
                lambda: self.check_5_fields(prepared, rules, config_name=config_name),
            ),
            # 检查 schema 主键字段组合是否在同一 Sheet 内重复。
            (
                "check_6_pkeys",
                lambda: self.check_6_pkeys(
                    prepared, rules, config_name=config_name
                ),
            ),
        )

        # 按顺序执行所有检测项。
        # 任一检测项异常都会转成 checker_exception 问题，不影响后续检测项。
        for check_name, run_check in steps:
            try:
                run_check()
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Data quality %s crashed; continue with next check", check_name
                )
                self.issues.append(
                    self._internal_error_issue(
                        config_name=config_name,
                        sheet="",
                        check=check_name,
                        exc=exc,
                    )
                )

        # 汇总问题明细，生成 passed 标记和按维度统计的 summary。
        self.result = self._build_result()

        # 按需写出操作员报告；报告写出失败只记日志，不影响返回结果。
        if report_dir is not None:
            try:
                self.write_report(report_dir)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to write data quality report to %s", report_dir
                )

        # 检测完成后统一记录错误摘要，便于日志侧快速感知质量状态。
        self._report_errors_at_end()

        # 返回本次检测结果，调用方可继续读取 self.result。
        return self.result

    def check_1_sheets(
        self,
        tables: dict[str, pd.DataFrame],
        rules: ConfigTableQualityRules,
        *,
        enabled_sheets: list[str] | None = None,
        optional_sheets: set[str] | None = None,
        config_name: str | None = None,
    ) -> None:
        """检测 1：schema 声明且检测开启的必需表在输入中是否存在。

        ``optional_sheets``（defaults.yaml#optional_import）中的可选表
        存在时才进行后续检测；缺失时不做任何检测、不记缺表问题。
        """
        # 确定本次需要检查的 Sheet 范围；默认使用 schema 全量 Sheet。
        sheets = rules.sheet_names() if enabled_sheets is None else enabled_sheets

        # 逐一确认必需 Sheet 是否出现在输入 tables 中。
        for sheet in sheets:
            if sheet in tables:
                continue

            # 可选表缺失：不做任何检测、不记缺表问题，仅日志留痕。
            if optional_sheets and sheet in optional_sheets:
                logger.info("Optional sheet is absent; all checks skipped: %s", sheet)
                continue

            # 必需表缺失记录为 ERROR，后续检测项会自动跳过该表。
            self.issues.append(
                self._build_issue(
                    config_name=config_name,
                    sheet=sheet,
                    check="check_1_sheets",
                    issue_type="missing_sheet",
                    severity="ERROR",
                    message=f"Mapped sheet is missing: {sheet}",
                    rule_id="sheet.missing_mapped",
                )
            )

    def check_2_empty(
        self,
        prepared: dict[str, pd.DataFrame],
        rules: ConfigTableQualityRules,
        *,
        config_name: str | None = None,
    ) -> None:
        """检测 2：投影并去掉填充空行后是否为空表。"""
        del rules  # 接口统一保留参数；空表判断无需规则配置。

        # 只检查已完成检测口径准备的 Sheet。
        for sheet, df in prepared.items():
            if not df.empty:
                continue

            # 准备后无有效数据行的 Sheet 记录为空表问题。
            self.issues.append(
                self._build_issue(
                    config_name=config_name,
                    sheet=sheet,
                    check="check_2_empty",
                    issue_type="empty_table",
                    severity="ERROR",
                    message=f"Mapped import table is empty: {sheet}",
                    rule_id="table.not_empty",
                )
            )

    def check_3_duplicates(
        self,
        prepared: dict[str, pd.DataFrame],
        rules: ConfigTableQualityRules,
        *,
        config_name: str | None = None,
    ) -> None:
        """检测 3：全列完全重复的数据行（只检测不删除）。"""
        del rules  # 接口统一保留参数；全行重复判断无需规则配置。

        # 逐表检查全行重复；空表和无列 DataFrame 无需处理。
        for sheet, df in prepared.items():
            if df.empty or len(df.columns) == 0:
                continue

            # keep=False 会标记每个重复组内的所有冲突行。
            duplicate_mask = df.duplicated(keep=False)

            # 将重复行样例写入标准问题明细，不修改原始数据。
            for idx in list(df.index[duplicate_mask])[:SAMPLE_LIMIT]:
                self.issues.append(
                    self._build_issue(
                        config_name=config_name,
                        sheet=sheet,
                        row_index=idx,
                        check="check_3_duplicates",
                        issue_type="duplicate_value",
                        severity="ERROR",
                        message="Row is a full duplicate of another row",
                        original_value={
                            column: df.at[idx, column] for column in df.columns
                        },
                        rule_id="row.full_duplicate",
                    )
                )

    def check_4_columns(
        self,
        tables: dict[str, pd.DataFrame],
        rules: ConfigTableQualityRules,
        *,
        enabled_sheets: list[str] | None = None,
        config_name: str | None = None,
    ) -> None:
        """检测 4：schema 声明的必需字段是否在原始 Sheet 列中缺失。"""
        # 确定需要检查字段完整性的 Sheet 范围。
        sheets = rules.sheet_names() if enabled_sheets is None else enabled_sheets

        # 字段缺失必须基于原始输入表检查，不能基于已投影 prepared 表。
        for sheet in sheets:
            df = tables.get(sheet)

            # 缺表由 check_1 负责报告，此处只处理已存在 Sheet 的缺列问题。
            if df is None:
                continue

            # 逐个 schema 必需字段确认是否存在于原始导入列中。
            for column in rules.required_columns(sheet):
                if column in df.columns:
                    continue

                # 缺失字段记录为 ERROR，字段级规则不会再重复报告该列。
                self.issues.append(
                    self._build_issue(
                        config_name=config_name,
                        sheet=sheet,
                        column=column,
                        check="check_4_columns",
                        issue_type="missing_import_column",
                        severity="ERROR",
                        message=f"Mapped field is missing: {column}",
                        rule_id="field.missing_mapped",
                    )
                )

    def check_5_fields(
        self,
        prepared: dict[str, pd.DataFrame],
        rules: ConfigTableQualityRules,
        *,
        config_name: str | None = None,
    ) -> None:
        """检测 5：字段级规则（非空/类型/枚举/范围/日期顺序）。"""
        # 逐表委托给 ConfigTableQualityRules，保持字段规则集中管理。
        # 规则层只返回轻量命中（不含流程上下文），此处补全问题归属
        # （config_name/sheet/check）后落为标准问题记录。
        for sheet, df in prepared.items():
            for hit in rules.validate_fields(sheet, df):
                self.issues.append(
                    self._build_issue(
                        config_name=config_name,
                        sheet=sheet,
                        check="check_5_fields",
                        **hit,
                    )
                )

    def check_6_pkeys(
        self,
        prepared: dict[str, pd.DataFrame],
        rules: ConfigTableQualityRules,
        *,
        config_name: str | None = None,
    ) -> None:
        """检测 6：主键字段组合分组下是否存在重复行。"""
        # 逐表检查主键组合唯一性。
        for sheet, df in prepared.items():
            # 只用实际存在的主键字段做判断；缺失主键列由 check_4 报告。
            keys = [col for col in rules.primary_key(sheet) if col in df.columns]
            if not keys or df.empty:
                continue

            # keep=False 标记所有主键重复行，便于报告完整呈现冲突。
            duplicate_mask = df.duplicated(subset=keys, keep=False)

            # 记录重复主键样例，并把主键取值作为 original_value。
            for idx in list(df.index[duplicate_mask])[:SAMPLE_LIMIT]:
                key_values = {col: df.at[idx, col] for col in keys}
                self.issues.append(
                    self._build_issue(
                        config_name=config_name,
                        sheet=sheet,
                        row_index=idx,
                        check="check_6_pkeys",
                        issue_type="duplicate_value",
                        severity="ERROR",
                        message=f"Duplicate primary key: {key_values}",
                        original_value=key_values,
                        rule_id="primary_key.unique",
                    )
                )

    def _build_result(self) -> dict[str, Any]:
        """汇总问题明细为检测结果字典（无 ERROR 即视为通过）。"""
        errors = sum(
            1 for issue in self.issues if issue["severity"] in {"ERROR", "CRITICAL"}
        )
        warnings = sum(1 for issue in self.issues if issue["severity"] == "WARNING")
        infos = sum(1 for issue in self.issues if issue["severity"] == "INFO")
        summary = {
            "issues": len(self.issues),
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "by_check": dict(Counter(issue["check"] for issue in self.issues)),
            "by_issue_type": dict(
                Counter(issue["issue_type"] for issue in self.issues)
            ),
            "by_sheet": dict(Counter(issue["sheet"] for issue in self.issues)),
        }
        return {"passed": errors == 0, "issues": list(self.issues), "summary": summary}

    @staticmethod
    def _internal_error_issue(
        *,
        config_name: str | None,
        sheet: str,
        check: str,
        exc: Exception,
    ) -> dict[str, Any]:
        """构造检测器内部异常的问题明细（保证检测流程不中断）。

        Args:
            config_name: 当前配置名。
            sheet: 出错时正在处理的 Sheet；步骤级异常可为空字符串。
            check: 出错的检测步骤标识（prepare 或 check_1 ~ check_6）。
            exc: 捕获到的异常对象。

        Returns:
            issue_type 为 ``checker_exception`` 的标准问题明细字典。
        """
        return ConfigInputDataQualityChecker._build_issue(
            config_name=config_name,
            sheet=sheet,
            check=check,
            issue_type="checker_exception",
            severity="ERROR",
            message=f"{check} crashed and was skipped: {exc!r}",
            rule_id="checker.internal_error",
        )

    def _report_errors_at_end(self) -> None:
        """全部检测完成后统一上报错误汇总（只记日志，不抛异常）。"""
        summary = self.result.get("summary", {})
        errors = int(summary.get("errors") or 0)
        if errors == 0:
            return
        logger.error(
            "[DQ] 输入数据质量检测完成：共 %s 个问题（errors=%s, warnings=%s）；"
            "按检测项：%s；按 Sheet：%s；明细见 result['issues'] 与 input_quality.xlsx",
            summary.get("issues"),
            errors,
            summary.get("warnings"),
            summary.get("by_check"),
            summary.get("by_sheet"),
        )

    def write_report(self, output_dir: str | Path) -> None:
        """写出面向操作员的数据质量检测报告工作簿（input_quality.xlsx）。

        Args:
            output_dir: 报告输出目录，不存在时自动创建。
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 报告列宽自适应参数：内容长度加 PAD，再夹在 MIN/MAX 之间。
        _COLUMN_WIDTH_PAD = 2
        _COLUMN_WIDTH_MIN = 10
        _COLUMN_WIDTH_MAX = 60

        workbook_path = output_path / "input_quality.xlsx"
        issues_df = self._build_report_issues_frame()
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            issues_df.to_excel(writer, sheet_name="issues", index=False)

            # 冻结表头并开启自动筛选，方便操作员浏览与过滤。
            worksheet = writer.sheets["issues"]
            worksheet.freeze_panes = "A2"
            if issues_df.shape[1] > 0:
                worksheet.auto_filter.ref = worksheet.dimensions
            # 按内容最大长度自适应列宽，并限制在最小/最大宽度之间。
            for column_cells in worksheet.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                width = min(
                    max(max_len + _COLUMN_WIDTH_PAD, _COLUMN_WIDTH_MIN),
                    _COLUMN_WIDTH_MAX,
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = width

    def _build_report_issues_frame(self) -> pd.DataFrame:
        """构造每张配置表一行的问题汇总 DataFrame。"""
        # 无问题时仍输出带固定表头的空表，保证报告结构稳定。
        raw = pd.DataFrame(self.issues)
        if raw.empty:
            return pd.DataFrame(columns=_ISSUE_COLUMNS)

        # 将问题类型编码翻译为中文标签，未收录的编码原样保留。
        raw["问题类型"] = raw["issue_type"].map(
            lambda value: _ISSUE_TYPE_LABELS_ZH.get(str(value), str(value))
        )
        # 按"配置名 + Sheet"聚合为一行，每列合并组内去重后的取值。
        rows: list[dict[str, Any]] = []
        for (config_name, sheet), group in raw.groupby(
            ["config_name", "sheet"], dropna=False
        ):
            rows.append(
                {
                    "配置名": config_name,
                    "配置表Sheet": sheet,
                    "字段名": self._join_unique(group.get("column")),
                    # 检测项直接展示英文标识（与检测方法名一致）。
                    "检测项": self._join_unique(group.get("check")),
                    "问题类型": self._join_unique(group.get("问题类型")),
                    "问题代码": self._join_unique(group.get("issue_type")),
                    "严重级别": self._join_unique(group.get("severity")),
                }
            )
        return pd.DataFrame(rows, columns=_ISSUE_COLUMNS)

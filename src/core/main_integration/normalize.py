"""
normalize.py

标识符规范化函数模块。

注意：这些实现是从 orchestrator/normalize.py 复制过来的，
但处理了更多列（实现有所不同）：
- dps_location、from_material、to_material
- line、delegate_line、changeover_id

本版本用于 main_integration 的配置表标准化。
"""

import pandas as pd

from src.utils.normalization_common import (
    normalize_identifiers_scalar,
    normalize_location_preserve_non_numeric,
    normalize_material_numeric_token_cleanup,
)


def _normalize_location(location_str) -> str:
    """规范化库位标识符

    目的/逻辑：
    - 纯数字字符串：左补零至4位
    - 非数字（如 A888）：原样保留
    - 直接调用本函数时，空值/None：返回空字符串

    入参：`location_str` 任意类型标识符
    出参：规范化后的字符串库位标识符
    """
    return normalize_location_preserve_non_numeric(location_str)


def _normalize_material(material_str) -> str:
    """规范化物料编码

    目的/逻辑：
    - 数值/浮点型物料编码转换为整数字符串，去除小数点后缀
    - 空值/None/NAN：返回空字符串
    - 其他情况：去除首尾空格
    - 与 code_v0 实现保持一致

    入参：`material_str` 任意类型标识符
    出参：规范化后的字符串物料编码
    """
    return normalize_material_numeric_token_cleanup(material_str)


def _normalize_sending(sending_str) -> str:
    """规范化发货库位标识符

    目的/逻辑：
    - 纯数字：补零至4位
    - 非数字：原样保留
    - 直接调用本函数时，空值/None：返回空字符串

    入参：`sending_str`
    出参：规范化后的字符串
    """
    return normalize_location_preserve_non_numeric(sending_str)


def _normalize_receiving(receiving_str) -> str:
    """规范化收货库位标识符

    目的/逻辑：与 `_normalize_sending` 相同
    入参：`receiving_str`
    出参：规范化后的字符串
    """
    return normalize_location_preserve_non_numeric(receiving_str)


def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """在 DataFrame 层面规范化标识符字段

    目的：
    - 统一将多个标识符列转换为字符串，并执行必要的格式化
      （库位补零、物料去小数点等），确保跨模块数据类型一致性。

    Args:
        df: 输入 DataFrame

    Returns:
        pd.DataFrame: 标识符已规范化的 DataFrame；非空输入返回副本，空表原样返回

    输入数据：
        - 包含 `material/location/sending/receiving/sourcing/...` 列的表

    输出/副作用：
        - 非空输入时返回新 DataFrame，不修改原始数据；空表直接原样返回

    逻辑：
        - 对每个标识符列先执行 astype(str)，再按列应用专项规范化；
          因此前置为空值的内容在该路径下会先转为字符串，再交由列级规则处理。
        - 空表直接原样返回。
    """
    identifier_cols = (
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
    )
    return normalize_identifiers_scalar(
        df,
        identifier_cols=identifier_cols,
        location_cols=("location", "dps_location", "sending", "receiving"),
        material_cols=("material", "from_material", "to_material"),
        passthrough_str_cols=("changeover_id", "line", "delegate_line"),
    )

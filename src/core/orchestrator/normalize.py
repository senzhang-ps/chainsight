# normalize.py
# 标识符规范化函数集
#
# 将 material / location / sending / receiving 等标识符列
# 转换为统一的字符串格式，确保跨模块数据一致性。

import pandas as pd


def _normalize_material(material_str) -> str:
    """规范化物料字符串——移除数值物料的 .0 后缀。

    Args:
        material_str: 原始物料标识（int/float/str/None）

    Returns:
        规范化后的字符串
    """
    if material_str is None or pd.isna(material_str):
        return ""

    try:
        if (
            isinstance(material_str, (int, float))
            or str(material_str)
            .replace('.', '')
            .replace('-', '')
            .isdigit()
        ):
            return str(int(float(material_str)))
        else:
            return str(material_str)
    except (ValueError, TypeError):
        return str(material_str)


def _normalize_location(location_str) -> str:
    """规范化地点字符串：纯数字补齐到 4 位。

    Args:
        location_str: 原始地点标识

    Returns:
        规范化后的字符串
    """
    if pd.isna(location_str) or location_str is None:
        return ""

    location_str = str(location_str).strip()

    try:
        if location_str.isdigit():
            return str(int(location_str)).zfill(4)
        else:
            return location_str
    except (ValueError, TypeError):
        return str(location_str)


def _normalize_sending(sending_str) -> str:
    """规范化发货地字符串：纯数字补齐到 4 位。

    Args:
        sending_str: 原始发货地标识

    Returns:
        规范化后的字符串
    """
    if pd.isna(sending_str) or sending_str is None:
        return ""

    sending_str = str(sending_str).strip()

    try:
        if sending_str.isdigit():
            return str(int(sending_str)).zfill(4)
        else:
            return sending_str
    except (ValueError, TypeError):
        return str(sending_str)


def _normalize_receiving(receiving_str) -> str:
    """规范化收货地字符串：纯数字补齐到 4 位。

    Args:
        receiving_str: 原始收货地标识

    Returns:
        规范化后的字符串
    """
    if pd.isna(receiving_str) or receiving_str is None:
        return ""

    receiving_str = str(receiving_str).strip()

    try:
        if receiving_str.isdigit():
            return str(int(receiving_str)).zfill(4)
        else:
            return receiving_str
    except (ValueError, TypeError):
        return str(receiving_str)


def _normalize_identifiers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """将标识符列规范化为字符串并按规则格式化。

    使用向量化操作提升性能，与 Dev 版本行为一致：
    - material 列：移除数值物料的 .0 后缀
    - location/sending/receiving/sourcing 列：纯数字补齐到 4 位

    Args:
        df: 待规范化的 DataFrame

    Returns:
        规范化后的 DataFrame 副本
    """
    if df.empty:
        return df

    df = df.copy()

    # 向量化处理 material 列
    if 'material' in df.columns:
        df['material'] = df['material'].astype(str)
        df['material'] = df['material'].replace(
            ['nan', 'None', '<NA>', 'NaN'], ''
        )
        df['material'] = df['material'].str.replace(
            r'\.0$', '', regex=True
        )

    # 向量化处理 location 类列
    location_cols = [
        'location', 'sending', 'receiving', 'sourcing',
    ]
    for col in location_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                ['nan', 'None', '<NA>', 'NaN'], ''
            )
            is_numeric = df[col].str.match(
                r'^\d+$', na=False
            )
            df.loc[is_numeric, col] = (
                df.loc[is_numeric, col].str.zfill(4)
            )

    return df

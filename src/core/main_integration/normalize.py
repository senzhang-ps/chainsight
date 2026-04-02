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


def _normalize_location(location_str) -> str:
    """规范化库位标识符

    目的/逻辑：
    - 纯数字字符串：左补零至4位
    - 非数字（如 A888）：原样保留
    - 直接调用本函数时，空值/None：返回空字符串

    入参：`location_str` 任意类型标识符
    出参：规范化后的字符串库位标识符
    """
    if pd.isna(location_str) or location_str is None:
        return ""
    
    location_str = str(location_str).strip()
    
    try:
        # 判断是否为纯数字字符串
        if location_str.isdigit():
            return str(int(location_str)).zfill(4)
        else:
            # 非数字库位（如 A888），原样返回，不补零
            return location_str
    except (ValueError, TypeError):
        return str(location_str)


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
    if material_str is None or material_str == '' or str(
        material_str).lower() in ['nan', 'none', '<na>']:
        return ""

    try:
        # 若为数值类型（int 或 float），转为整数字符串以移除 .0 后缀
        if isinstance(material_str, (int, float)) or str(
            material_str).replace('.', '').replace('-', '').isdigit():
            return str(int(float(material_str)))
        else:
            # 非数字物料编码，原样转字符串返回
            return str(material_str).strip()
    except (ValueError, TypeError):
        # 转换失败时，原样转字符串返回
        return str(material_str).strip()


def _normalize_sending(sending_str) -> str:
    """规范化发货库位标识符

    目的/逻辑：
    - 纯数字：补零至4位
    - 非数字：原样保留
    - 直接调用本函数时，空值/None：返回空字符串

    入参：`sending_str`
    出参：规范化后的字符串
    """
    if pd.isna(sending_str) or sending_str is None:
        return ""
    
    sending_str = str(sending_str).strip()
    
    try:
        # 判断是否为纯数字字符串
        if sending_str.isdigit():
            return str(int(sending_str)).zfill(4)
        else:
            # 非数字发货库位（如 A888），原样返回，不补零
            return sending_str
    except (ValueError, TypeError):
        return str(sending_str)


def _normalize_receiving(receiving_str) -> str:
    """规范化收货库位标识符

    目的/逻辑：与 `_normalize_sending` 相同
    入参：`receiving_str`
    出参：规范化后的字符串
    """
    if pd.isna(receiving_str) or receiving_str is None:
        return ""
    
    receiving_str = str(receiving_str).strip()
    
    try:
        # 判断是否为纯数字字符串
        if receiving_str.isdigit():
            return str(int(receiving_str)).zfill(4)
        else:
            # 非数字收货库位（如 A888），原样返回，不补零
            return receiving_str
    except (ValueError, TypeError):
        return str(receiving_str)


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
    if df.empty:
        return df
    
    # 定义需要字符串转换的标识符列
    identifier_cols = [
        'material', 'location', 'sending', 'receiving', 'sourcing',
        'dps_location', 'from_material', 'to_material', 'line',
        'delegate_line', 'changeover_id'
    ]
    
    df = df.copy()
    for col in identifier_cols:
        if col in df.columns:
            # 关键修复：使用 object dtype（Python str）而非 pandas StringDtype，
            # 确保与后续 astype(str) 的一致性
            df[col] = df[col].astype(str)
            # 对库位类字段执行专项规范化
            if col in ['location', 'dps_location']:
                df[col] = df[col].apply(_normalize_location)
            elif col == 'sending':
                df[col] = df[col].apply(_normalize_sending)
            elif col == 'receiving':
                df[col] = df[col].apply(_normalize_receiving)
            # 对物料类字段执行专项规范化
            elif col in ['material', 'from_material', 'to_material']:
                df[col] = df[col].apply(_normalize_material)
            # changeover_id 和 line 仅需字符串转换，无需特殊格式化
            # 其他标识符列（line、delegate_line 等）确保正确字符串格式
            elif col in ['changeover_id', 'line', 'delegate_line']:
                # 这些字段只需保持字符串类型，无需额外处理
                pass
            else:
                df[col] = df[col].apply(
                    lambda x: str(x) if pd.notna(x) else "")
    
    return df

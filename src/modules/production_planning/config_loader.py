"""
配置加载模块

负责加载和校验Module4配置文件。
"""

from typing import Dict, List, Any

import pandas as pd

from .constants import REQUIRED_CONFIG_SHEETS, SHEET_KEY_MAPPING
from .utils import cast_identifiers_to_str, validate_merge_keys


def load_config(filepath: str) -> Dict[str, Any]:
    """加载模块4配置文件。

    读取必需工作表，转换标识符类型，兼容可选工作表。

    Args:
        filepath: 配置Excel文件路径

    Returns:
        Dict[str, Any]: 配置字典

    Raises:
        KeyError: 缺少必需工作表时
    """
    xl = pd.ExcelFile(filepath)
    cfg = {}

    _load_required_sheets(xl, cfg)
    _load_optional_sheets(xl, filepath, cfg)

    return cfg


def _load_required_sheets(xl: pd.ExcelFile, cfg: Dict[str, Any]) -> None:
    """加载必需的配置工作表。

    Args:
        xl: Excel文件对象
        cfg: 配置字典（会被修改）

    Raises:
        KeyError: 缺少必需工作表时
    """
    for sheet_name in REQUIRED_CONFIG_SHEETS:
        if sheet_name not in xl.sheet_names:
            raise KeyError(f"缺少必需的工作表: {sheet_name}")

        internal_key = SHEET_KEY_MAPPING.get(sheet_name, sheet_name)
        df = cast_identifiers_to_str(xl.parse(sheet_name))
        cfg[internal_key] = df


def _load_optional_sheets(
    xl: pd.ExcelFile,
    filepath: str,
    cfg: Dict[str, Any]
) -> None:
    """加载可选的配置工作表。

    Args:
        xl: Excel文件对象
        filepath: 文件路径
        cfg: 配置字典（会被修改）
    """
    _load_net_demand_sheet(xl, cfg)
    _load_seed_sheet(xl, filepath, cfg)


def _load_net_demand_sheet(xl: pd.ExcelFile, cfg: Dict[str, Any]) -> None:
    """加载可选的NetDemand工作表。

    Args:
        xl: Excel文件对象
        cfg: 配置字典（会被修改）
    """
    if 'NetDemand' in xl.sheet_names:
        cfg['NetDemand'] = cast_identifiers_to_str(
            xl.parse('NetDemand'),
            ['material', 'location']
        )


def _load_seed_sheet(
    xl: pd.ExcelFile,
    filepath: str,
    cfg: Dict[str, Any]
) -> None:
    """加载可选的Global_seed工作表。

    Args:
        xl: Excel文件对象
        filepath: 文件路径
        cfg: 配置字典（会被修改）
    """
    if 'Global_seed' in xl.sheet_names:
        seed_df = pd.read_excel(filepath, sheet_name='Global_seed')
        if not seed_df.empty:
            cfg['RandomSeed'] = int(seed_df.iloc[0, 0])


def validate_config(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """校验配置一致性。

    检查NetDemand与MaterialLocationLineCfg的可合并性，
    提示缺少线配置或一物料地点多线情况。

    Args:
        cfg: 配置字典

    Returns:
        List[Dict[str, str]]: 问题列表（非致命）
    """
    issues = []

    _validate_net_demand(cfg, issues)
    _validate_line_config(cfg, issues)

    return issues


def _validate_net_demand(
    cfg: Dict[str, Any],
    issues: List[Dict[str, str]]
) -> None:
    """校验NetDemand配置。

    Args:
        cfg: 配置字典
        issues: 问题列表（会被修改）
    """
    if 'NetDemand' not in cfg or cfg['NetDemand'].empty:
        return

    nd = cfg['NetDemand'][['material', 'location']]
    ml = cfg['MaterialLocationLineCfg'][['material', 'location']]

    validate_merge_keys(nd, ml, ['material', 'location'])

    merged = pd.merge(
        nd, ml,
        on=['material', 'location'],
        how='left',
        indicator=True
    )

    _collect_missing_line_issues(merged, issues)


def _collect_missing_line_issues(
    merged: pd.DataFrame,
    issues: List[Dict[str, str]]
) -> None:
    """收集缺少产线配置的问题。

    Args:
        merged: 合并后的DataFrame
        issues: 问题列表（会被修改）
    """
    bad = merged[merged['_merge'] == 'left_only']
    unique_pairs = bad[['material', 'location']].drop_duplicates()

    # 优化：使用 itertuples() 替代 iterrows()
    for row in unique_pairs.itertuples():
        issues.append({
            'sheet': 'MaterialLocationLineCfg',
            'row': '',
            'issue': (
                f"物料 {row.material} 在地点 {row.location} "
                f"缺少产线配置"
            )
        })


def _validate_line_config(
    cfg: Dict[str, Any],
    issues: List[Dict[str, str]]
) -> None:
    """校验产线配置的唯一性。

    Args:
        cfg: 配置字典
        issues: 问题列表（会被修改）
    """
    if 'MaterialLocationLineCfg' not in cfg:
        return

    line_counts = cfg['MaterialLocationLineCfg'].groupby(
        ['material', 'location']
    ).size()

    for (mat, loc), cnt in line_counts.items():
        if cnt > 1:
            issues.append({
                'sheet': 'MaterialLocationLineCfg',
                'row': '',
                'issue': (
                    f"物料-地点组合存在多条产线配置: {mat}/{loc}"
                )
            })

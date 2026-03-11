# -*- coding: utf-8 -*-
"""
缓存和工具模块

提供缓存构建功能和通用工具函数。

优化历史:
- v3.0: 添加 DuckDB 加速索引构建选项
"""
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .constants import DEFAULT_PTF, DEFAULT_LSK, DEFAULT_LEAD_TIME

# DuckDB 加速开关
USE_DUCKDB_INDEX: bool = True


def _try_duckdb_build_index(
    df: pd.DataFrame,
    key_columns: List[str],
    filter_date_range: Optional[Tuple] = None
) -> Optional[Dict[tuple, pd.DataFrame]]:
    """
    尝试使用 DuckDB 构建索引。
    
    返回：
        索引字典，如果失败则返回 None
    """
    if not USE_DUCKDB_INDEX:
        return None
    
    try:
        from ...utils.duckdb_optimizer import get_duckdb_optimizer
        optimizer = get_duckdb_optimizer()
        
        # 构建简单的分组索引
        if df.empty:
            return {}
        
        df = df.copy()
        for col in key_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        result = {}
        for key, group in df.groupby(key_columns, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            result[key] = group.reset_index(drop=True)
        
        return result
    except Exception:
        return None


def build_dataframe_index(
    df: pd.DataFrame,
    key_columns: List[str]
) -> Dict[tuple, pd.DataFrame]:
    """
    构建DataFrame的GroupBy索引，实现O(1)查找。

    参数：
        df: 源DataFrame
        key_columns: 索引列名列表

    返回：
        dict: key_tuple -> 对应的DataFrame子集
    """
    if df.empty:
        return {}

    index_dict = {}
    grouped = df.groupby(key_columns, sort=False)
    for key, group in grouped:
        # 确保key总是tuple格式
        if not isinstance(key, tuple):
            key = (key,)
        index_dict[key] = group

    return index_dict


def build_sdl_index(
    supply_demand_log: pd.DataFrame
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    构建SupplyDemandLog的(material, location)索引。

    参数：
        supply_demand_log: SDL DataFrame

    返回：
        dict: (material, location) -> 对应记录
    """
    if supply_demand_log.empty:
        return {}

    # 确保material和location是字符串类型
    sdl = supply_demand_log.copy()
    sdl['material'] = sdl['material'].astype(str)
    sdl['location'] = sdl['location'].astype(str)

    return build_dataframe_index(sdl, ['material', 'location'])


def build_safety_stock_index(
    safety_stock: pd.DataFrame
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    构建SafetyStock的(material, location)索引。

    参数：
        safety_stock: `SafetyStock` DataFrame

    返回：
        dict: (material, location) -> 对应记录
    """
    if safety_stock.empty:
        return {}

    ss = safety_stock.copy()
    ss['material'] = ss['material'].astype(str)
    ss['location'] = ss['location'].astype(str)

    return build_dataframe_index(ss, ['material', 'location'])


def build_order_log_index(
    order_log: pd.DataFrame
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    构建OrderLog的(material, location)索引。

    参数：
        order_log: `OrderLog` DataFrame

    返回：
        dict: (material, location) -> 对应记录
    """
    if order_log.empty:
        return {}

    ol = order_log.copy()
    ol['material'] = ol['material'].astype(str)
    ol['location'] = ol['location'].astype(str)

    return build_dataframe_index(ol, ['material', 'location'])


def build_deploy_config_index(
    deploy_config: pd.DataFrame
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    构建DeployConfig的(material, sending)索引。

    参数：
        deploy_config: DeployConfig DataFrame

    返回：
        dict: (material, sending) -> 对应记录
    """
    if deploy_config.empty:
        return {}

    dc = deploy_config.copy()
    dc['material'] = dc['material'].astype(str)
    dc['sending'] = dc['sending'].astype(str)

    return build_dataframe_index(dc, ['material', 'sending'])


def get_from_index(
    index: Dict[tuple, pd.DataFrame],
    key: tuple
) -> pd.DataFrame:
    """
    从索引中获取数据，不存在则返回空DataFrame。

    参数：
        index: 索引字典
        key: 查找键

    返回：
        pd.DataFrame: 匹配的记录或空DataFrame
    """
    return index.get(key, pd.DataFrame())


def build_ptf_lsk_cache(
    m4_mlcfg_df: Optional[pd.DataFrame]
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    构建PTF/LSK缓存。

    用途：为Plant口径的lead time计算提供PTF/LSK值。

    参数：
        m4_mlcfg_df: M4_MaterialLocationLineCfg DataFrame

    返回：
        dict: (material, location) -> (ptf, lsk)
    """
    cache = {}
    if m4_mlcfg_df is None or m4_mlcfg_df.empty:
        return cache

    for row in m4_mlcfg_df.itertuples():
        material = getattr(row, 'material', None)
        location = getattr(row, 'location', None)
        if material is None or location is None:
            continue

        ptf = DEFAULT_PTF
        lsk = DEFAULT_LSK

        # 尝试小写和大写列名
        ptf_val = getattr(row, 'ptf', None) or getattr(row, 'PTF', None)
        lsk_val = getattr(row, 'lsk', None) or getattr(row, 'LSK', None)

        if ptf_val is not None and not pd.isna(ptf_val):
            ptf = int(ptf_val)
        if lsk_val is not None and not pd.isna(lsk_val):
            lsk = int(lsk_val)

        cache[(str(material), str(location))] = (ptf, lsk)

    return cache


def get_ptf_lsk(
    material: str,
    site: str,
    m4_mlcfg_df: Optional[pd.DataFrame],
    cache: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None
) -> Tuple[int, int]:
    """
    获取指定(material, site)的PTF/LSK。

    参数：
        material: 物料编码
        site: 站点编码
        m4_mlcfg_df: M4_MaterialLocationLineCfg DataFrame
        cache: PTF/LSK缓存

    返回：
        tuple: (ptf, lsk)
    """
    # 使用缓存
    if cache is not None:
        return cache.get(
            (str(material), str(site)),
            (DEFAULT_PTF, DEFAULT_LSK)
        )

    # 回退到DataFrame查询
    ptf, lsk = DEFAULT_PTF, DEFAULT_LSK
    if m4_mlcfg_df is None or m4_mlcfg_df.empty:
        return ptf, lsk

    ml = m4_mlcfg_df[
        (m4_mlcfg_df['material'] == material) &
        (m4_mlcfg_df['location'] == site)
    ]
    if ml.empty:
        return ptf, lsk

    row = ml.iloc[0]
    if 'ptf' in ml.columns and pd.notna(row.get('ptf')):
        ptf = int(row['ptf'])
    elif 'PTF' in ml.columns and pd.notna(row.get('PTF')):
        ptf = int(row['PTF'])

    if 'lsk' in ml.columns and pd.notna(row.get('lsk')):
        lsk = int(row['lsk'])
    elif 'LSK' in ml.columns and pd.notna(row.get('LSK')):
        lsk = int(row['LSK'])

    return ptf, lsk


def build_lead_time_cache(
    lead_time_df: pd.DataFrame
) -> Dict[Tuple[str, str], Tuple[int, int, int]]:
    """
    构建LeadTime基础参数缓存（PDT/GR/MCT）。

    参数：
        lead_time_df: LeadTime DataFrame

    返回：
        dict: (sending, receiving) -> (PDT, GR, MCT)
    """
    cache = {}
    if lead_time_df.empty:
        return cache

    for row in lead_time_df.itertuples():
        sending = getattr(row, 'sending', None)
        receiving = getattr(row, 'receiving', None)
        if sending is None or receiving is None:
            continue

        pdt = int(getattr(row, 'PDT', 0) or 0)
        gr = int(getattr(row, 'GR', 0) or 0)
        mct = int(getattr(row, 'MCT', 0) or 0)

        cache[(str(sending), str(receiving))] = (pdt, gr, mct)

    return cache


def build_active_network_cache(
    network_df: pd.DataFrame
) -> Dict[Tuple[str, str, pd.Timestamp, pd.Timestamp], object]:
    """
    构建Network活动行缓存。

    参数：
        network_df: Network DataFrame

    返回：
        dict: (material, location, eff_from, eff_to) -> row
    """
    cache = {}
    if network_df.empty:
        return cache

    for row in network_df.itertuples():
        material = getattr(row, 'material', None)
        location = getattr(row, 'location', None)
        eff_from = getattr(row, 'eff_from', None)
        eff_to = getattr(row, 'eff_to', None)

        if material is None or location is None:
            continue

        key = (str(material), str(location), eff_from, eff_to)
        cache[key] = row

    return cache


def get_active_network(
    network_df: pd.DataFrame,
    material: str,
    location: str,
    sim_date: pd.Timestamp,
    cache: Optional[dict] = None
) -> pd.DataFrame:
    """
    获取(material, location)在sim_date的活动Network行。

    参数：
        network_df: Network DataFrame
        material: 物料编码
        location: 位置编码
        sim_date: 仿真日期
        cache: Network缓存

    返回：
        pd.DataFrame: 匹配的Network行
    """
    if cache is not None:
        matching_rows = []
        for key, row in cache.items():
            if (key[0] == str(material) and
                key[1] == str(location) and
                key[2] <= sim_date <= key[3]):
                matching_rows.append(row)

        if matching_rows:
            return pd.DataFrame([matching_rows[0]._asdict()])
        return pd.DataFrame()

    # 回退到DataFrame过滤
    rows = network_df[
        (network_df['material'] == material) &
        (network_df['location'] == location) &
        (network_df['eff_from'] <= sim_date) &
        (network_df['eff_to'] >= sim_date)
    ]
    return rows


def get_upstream(
    location: str,
    material: str,
    network_df: pd.DataFrame,
    sim_date: pd.Timestamp,
    active_network_cache: Optional[dict] = None
) -> Optional[str]:
    """
    查找(material, location)的上游sourcing。

    参数：
        location: 位置编码
        material: 物料编码
        network_df: Network DataFrame
        sim_date: 仿真日期
        active_network_cache: Network缓存

    返回：
        str or None: 上游位置编码
    """
    row = get_active_network(
        network_df, material, location, sim_date,
        cache=active_network_cache
    )
    if not row.empty:
        return row.iloc[0]['sourcing']
    return None


def assign_location_layers(network_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据Network的sourcing→location关系，**按物料维度**计算层级。

    每个物料单独建图并做BFS，得到该物料下各location的layer。
    返回包含material/location/layer的DataFrame，供后续按(material, location)查询。

    参数：
        network_df: Network DataFrame

    返回：
        pd.DataFrame: 包含material, location和layer的DataFrame
    """
    if network_df.empty:
        return pd.DataFrame({'material': [], 'location': [], 'layer': []})

    layer_rows = []
    # 使用 `dropna=False` 保留 NaN 物料键，尽量与原始数据保持一致
    for material, mat_df in network_df.groupby('material', dropna=False):
        if mat_df.empty:
            continue

        children = defaultdict(list)
        parents = defaultdict(list)
        for row in mat_df.itertuples(index=False):
            sourcing_val = getattr(row, 'sourcing', None)  # type: ignore[attr-defined]
            location_val = getattr(row, 'location', None)  # type: ignore[attr-defined]
            sourcing_valid = sourcing_val is not None and pd.notna(sourcing_val) and str(sourcing_val).strip() != ''
            location_valid = location_val is not None and pd.notna(location_val) and str(location_val).strip() != ''
            if sourcing_valid and location_valid:
                children[str(sourcing_val)].append(str(location_val))
                parents[str(location_val)].append(str(sourcing_val))

        all_locations = set(mat_df['location'].dropna().astype(str)).union(
            set(mat_df['sourcing'].dropna().astype(str))
        )
        if not all_locations:
            continue

        potential_roots = [loc for loc in all_locations if not parents[loc]]
        true_roots = []
        for loc in potential_roots:
            if loc in children:
                true_roots.append(loc)
            else:
                has_incoming = any(loc in parents.get(other_loc, []) for other_loc in all_locations)
                if not has_incoming:
                    true_roots.append(loc)
        if not true_roots:
            true_roots = potential_roots if potential_roots else list(all_locations)

        layer_dict = {}
        queue = deque()
        queue.extend((root, 0) for root in true_roots)
        while queue:
            loc, layer = queue.popleft()
            if loc in layer_dict and layer_dict[loc] <= layer:
                continue
            layer_dict[loc] = layer
            for child in children.get(loc, []):
                queue.append((child, layer + 1))

        unassigned = [loc for loc in all_locations if loc not in layer_dict]
        if unassigned:
            max_layer = max(layer_dict.values()) if layer_dict else 0
            for loc in unassigned:
                layer_dict[loc] = max_layer + 1

        for loc, layer in layer_dict.items():
            # 处理 groupby 产生的 NaN material
            try:
                is_na = material is None or pd.isna(material)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                is_na = False
            material_str = '' if is_na else str(material)
            layer_rows.append({
                'material': material_str,
                'location': loc,
                'layer': layer
            })

    if not layer_rows:
        return pd.DataFrame({'material': [], 'location': [], 'layer': []})

    layer_df = pd.DataFrame(layer_rows).sort_values(['material', 'layer', 'location']).reset_index(drop=True)
    return layer_df


def determine_lead_time(
    sending: str,
    receiving: str,
    location_type: str,
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: Optional[pd.DataFrame] = None,
    material: Optional[str] = None,
    lead_time_cache: Optional[dict] = None,
    ptf_lsk_cache: Optional[dict] = None
) -> Tuple[int, str]:
    """
    计算(sending→receiving)的到货提前期。

    Plant: lead_time = max(MCT, PDT+GR) + PTF + LSK - 1
    DC: lead_time = PDT + GR

    参数：
        sending: 发送端编码
        receiving: 接收端编码
        location_type: 位置类型（Plant/DC）
        lead_time_df: LeadTime DataFrame
        m4_mlcfg_df: M4_MaterialLocationLineCfg DataFrame
        material: 物料编码
        lead_time_cache: LeadTime缓存
        ptf_lsk_cache: PTF/LSK缓存

    返回：
        tuple: (lead_time, error_message)
    """
    # 使用缓存
    if lead_time_cache is not None:
        base_values = lead_time_cache.get((str(sending), str(receiving)))
        if base_values is None:
            return DEFAULT_LEAD_TIME, 'lead_time_missing'
        pdt, gr, mct = base_values
    else:
        # 回退到DataFrame过滤
        if lead_time_df.empty:
            return DEFAULT_LEAD_TIME, 'empty_lead_time_config'

        row = lead_time_df[
            (lead_time_df['sending'] == sending) &
            (lead_time_df['receiving'] == receiving)
        ]
        if row.empty:
            return DEFAULT_LEAD_TIME, 'lead_time_missing'

        try:
            pdt = int(row.iloc[0].get('PDT', 0) or 0)
            gr = int(row.iloc[0].get('GR', 0) or 0)
            mct = int(row.iloc[0].get('MCT', 0) or 0)
        except Exception as e:
            return DEFAULT_LEAD_TIME, f'lead_time_calculation_error: {str(e)}'

    try:
        ptf, lsk = DEFAULT_PTF, DEFAULT_LSK
        if str(location_type).lower() == 'plant' and material is not None:
            ptf, lsk = get_ptf_lsk(
                material=material, site=sending,
                m4_mlcfg_df=m4_mlcfg_df, cache=ptf_lsk_cache
            )

        if str(location_type).lower() == 'plant':
            base_lt = max(mct, pdt + gr)
            leadtime = base_lt + ptf + lsk - 1
        else:
            leadtime = pdt + gr

        return max(1, int(leadtime)), ""

    except Exception as e:
        return DEFAULT_LEAD_TIME, f'lead_time_calculation_error: {str(e)}'


def get_sending_location_type(
    material: str,
    sending: str,
    sim_date: pd.Timestamp,
    network_df: pd.DataFrame,
    location_layer_map: Dict[Tuple[str, str], int]
) -> str:
    """
    识别发送端类型（与Module3一致）。

    1) Network有活动行则使用其location_type
    2) 若为根层（layer=0），视为Plant
    3) 否则默认DC

    参数：
        material: 物料编码
        sending: 发送端编码
        sim_date: 仿真日期
        network_df: Network DataFrame
        location_layer_map: 位置层级映射 dict[(material, location): layer]

    返回：
        str: 'Plant' 或 'DC'
    """
    if not sending or pd.isna(sending) or str(sending).strip() == "":
        return 'DC'

    row = get_active_network(network_df, material, sending, sim_date, cache=None)
    if not row.empty:
        return str(row.iloc[0].get('location_type', 'DC') or 'DC')

    # 未维护但被识别为根节点 → Plant
    # 使用 (material, sending) 键，与基线保持一致
    layer_key = (str(material), str(sending))
    if location_layer_map.get(layer_key, None) == 0:
        return 'Plant'

    return 'DC'


def is_review_day(dt: pd.Timestamp, lsk: str, day: int) -> bool:
    """
    判断是否为回顾日。

    参数：
        dt: 日期
        lsk: 回顾类型（daily/weekly/monthly）
        day: 日期参数

    返回：
        bool: 是否为回顾日
    """
    if lsk == 'daily':
        return True
    if lsk == 'weekly':
        return dt.weekday() == (int(day) - 1)
    if lsk == 'monthly':
        return dt.day == int(day)
    raise ValueError(f"Unknown LSK: {lsk}")

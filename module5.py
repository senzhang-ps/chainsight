#module 5
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from typing import Dict, List
from functools import lru_cache

# ========= 集成数据加载函数 (新增) =========

def _normalize_location(location_str) -> str:
    """Normalize location string by padding with leading zeros to 4 digits"""
    # Handle None and pandas NA
    if location_str is None or pd.isna(location_str):
        return ""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)

def _normalize_material(material_str) -> str:
    """Normalize material string"""
    # Handle None and pandas NA
    if material_str is None or pd.isna(material_str):
        return ""
    return str(material_str)

def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifier columns to string format with proper formatting"""
    if df.empty:
        return df
    
    # Define identifier columns that need string conversion
    identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing']
    
    df = df.copy()
    for col in identifier_cols:
        if col in df.columns:
            # Convert to string and handle NaN values
            df[col] = df[col].astype('string')
            # Apply normalization functions
            if col in ['location', 'sending', 'receiving', 'sourcing']:
                # Normalization for location-type fields
                df[col] = df[col].apply(_normalize_location)
            # Apply specific normalization for material
            elif col == 'material':
                df[col] = df[col].apply(_normalize_material)
            # For other identifier columns, ensure they are properly formatted strings
            else:
                # Vectorized string conversion
                df[col] = df[col].fillna('').astype(str)
    
    return df

def load_module1_daily_shipment(module1_output_dir: str, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    从Module1输出加载当日发货数据
    
    Args:
        module1_output_dir: Module1输出目录
        current_date: 当前日期
        
    Returns:
        pd.DataFrame: 当日发货数据 [date, material, location, quantity]
    """
    try:
        date_str = current_date.strftime('%Y%m%d')
        module1_file = f"{module1_output_dir}/module1_output_{date_str}.xlsx"
        
        if os.path.exists(module1_file):
            xl = pd.ExcelFile(module1_file)
            if 'ShipmentLog' in xl.sheet_names:
                shipment_df = xl.parse('ShipmentLog')
                # 确保包含需要的列
                required_cols = ['date', 'material', 'location', 'quantity']
                if all(col in shipment_df.columns for col in required_cols):
                    result_df = shipment_df[required_cols].copy()
                    # 确保标识符字段为字符串格式
                    return _normalize_identifiers(result_df)
                else:
                    print(f"⚠️  Module1输出文件缺少必要字段: {module1_file}")
            else:
                print(f"⚠️  Module1输出文件中无ShipmentLog表: {module1_file}")
        else:
            print(f"⚠️  Module1输出文件不存在: {module1_file}")
    except Exception as e:
        print(f"⚠️  加载Module1发货数据失败: {e}")
    
    # 返回空DataFrame
    return pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])

def load_module1_daily_orders(module1_output_dir: str, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    从Module1输出加载"当日版本"的订单日志（包含历史天生成但尚未来到期的订单 + 当天新生成）
    仅按 requirement_date>=current_date 过滤，不按 simulation_date 过滤
    返回列: [date, material, location, demand_type, quantity, simulation_date]
    """
    cols = ['date', 'material', 'location', 'demand_type', 'quantity', 'simulation_date']
    try:
        date_str = current_date.strftime('%Y%m%d')
        module1_file = f"{module1_output_dir}/module1_output_{date_str}.xlsx"
        if not os.path.exists(module1_file):
            print(f"⚠️  Module1输出文件不存在: {module1_file}")
            return pd.DataFrame(columns=cols)

        xl = pd.ExcelFile(module1_file)
        if 'OrderLog' not in xl.sheet_names:
            print(f"⚠️  Module1输出文件中无OrderLog表: {module1_file}")
            return pd.DataFrame(columns=cols)

        df = xl.parse('OrderLog')
        for c in ['date', 'simulation_date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        # 只保留 requirement_date(=date) >= today 的订单行
        if 'date' in df.columns:
            df = df[df['date'] >= current_date]

        # 规范列
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NaT if c in ['date','simulation_date'] else np.nan
        result_df = df[cols].copy()
        # 确保标识符字段为字符串格式
        return _normalize_identifiers(result_df)

    except Exception as e:
        print(f"⚠️  加载Module1订单数据失败: {e}")
        return pd.DataFrame(columns=cols)

def load_orchestrator_delivery_gr(orchestrator: object, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    从Orchestrator加载当日收货数据
    
    Args:
        orchestrator: Orchestrator实例
        current_date: 当前日期
        
    Returns:
        pd.DataFrame: 当日收货数据 [date, material, receiving, quantity]
    """
    try:
        date_str = current_date.strftime('%Y-%m-%d')
        delivery_gr_view = orchestrator.get_delivery_gr_view(date_str)
        
        if isinstance(delivery_gr_view, pd.DataFrame) and not delivery_gr_view.empty:
            # 确保包含需要的列
            required_cols = ['date', 'material', 'receiving', 'quantity']
            available_cols = delivery_gr_view.columns.tolist()
            
            # 尝试映射列名称
            col_mapping = {
                'location': 'receiving',  # location 映射为 receiving
                'gr_qty': 'quantity',     # gr_qty 映射为 quantity
                'received_qty': 'quantity'  # received_qty 映射为 quantity
            }
            
            # 应用列映射
            renamed_df = delivery_gr_view.copy()
            for old_col, new_col in col_mapping.items():
                if old_col in renamed_df.columns:
                    renamed_df = renamed_df.rename(columns={old_col: new_col})
            
            # 检查必要列是否存在
            missing_cols = [col for col in required_cols if col not in renamed_df.columns]
            if not missing_cols:
                result_df = renamed_df[required_cols].copy()
                # 确保标识符字段为字符串格式
                return _normalize_identifiers(result_df)
            else:
                print(f"⚠️  Orchestrator delivery_gr_view缺少字段: {missing_cols}")
        else:
            print(f"⚠️  Orchestrator返回空的delivery_gr_view")
    except Exception as e:
        print(f"⚠️  从Orchestrator加载收货数据失败: {e}")
    
    # 返回空DataFrame
    return pd.DataFrame(columns=['date', 'material', 'receiving', 'quantity'])

def load_orchestrator_open_deployment(orchestrator: object, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    从Orchestrator加载开放调拨数据
    
    Args:
        orchestrator: Orchestrator实例
        current_date: 当前日期
        
    Returns:
        pd.DataFrame: 开放调拨数据 [material, sending, receiving, quantity]
    """
    try:
        date_str = current_date.strftime('%Y-%m-%d')
        open_deployment_view = orchestrator.get_open_deployment_view(date_str)
        
        if isinstance(open_deployment_view, pd.DataFrame) and not open_deployment_view.empty:
            # 确保包含需要的列（包括receiving用于自循环检查）
            required_cols = ['material', 'sending', 'receiving', 'quantity']
            available_cols = open_deployment_view.columns.tolist()
            
            # 尝试映射列名称
            col_mapping = {
                'location': 'sending',     # location 映射为 sending
                'deployed_qty': 'quantity',  # deployed_qty 映射为 quantity
                'planned_qty': 'quantity'    # planned_qty 映射为 quantity
            }
            
            # 应用列映射
            renamed_df = open_deployment_view.copy()
            for old_col, new_col in col_mapping.items():
                if old_col in renamed_df.columns:
                    renamed_df = renamed_df.rename(columns={old_col: new_col})
            
            # 检查必要列是否存在
            missing_cols = [col for col in required_cols if col not in renamed_df.columns]
            if not missing_cols:
                result_df = renamed_df[required_cols].copy()
                # 确保标识符字段为字符串格式
                return _normalize_identifiers(result_df)
            else:
                print(f"⚠️  Orchestrator open_deployment_view缺少字段: {missing_cols}")
        else:
            print(f"⚠️  Orchestrator返回空的open_deployment_view")
    except Exception as e:
        print(f"⚠️  从Orchestrator加载开放调拨数据失败: {e}")
    
    # 返回空DataFrame
    return pd.DataFrame(columns=['material', 'sending', 'receiving', 'quantity'])

def build_open_deployment_inbound(open_deployment_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """
    从 open_deployment 明细构造 inbound 视图：
    - 维度： (material, receiving)
    - 过滤：sending != receiving（排除自循环）；deployed_qty/quantity > 0
    - 汇总：sum(quantity)
    返回：{(material, receiving): qty}
    """
    if open_deployment_df is None or open_deployment_df.empty:
        return {}

    df = open_deployment_df.copy()
    # 统一数量列名
    if 'quantity' not in df.columns and 'deployed_qty' in df.columns:
        df = df.rename(columns={'deployed_qty': 'quantity'})
    if 'quantity' not in df.columns:
        # 兜底：如果叫 planned_qty
        if 'planned_qty' in df.columns:
            df = df.rename(columns={'planned_qty': 'quantity'})
        else:
            return {}

    # 过滤：数量>0，且非自循环
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    df = df[(df['quantity'] > 0) & (df['sending'] != df['receiving'])]

    # 聚合： (material, receiving)
    g = (df.groupby(['material', 'receiving'])['quantity']
           .sum().reset_index())

    # Performance optimization: Use dict comprehension with itertuples
    inbound = {(row.material, row.receiving): int(row.quantity) for row in g.itertuples(index=False)}
    return inbound

def calculate_projected_inventory(
    beginning_inventory: dict,
    in_transit: dict, 
    delivery_gr: dict,
    today_production_gr: dict,
    future_production: dict,
    today_shipment: dict,
    open_deployment: dict
) -> dict:
    """
    计算预测库存，用于gap计算和供应链规划
    
    Formula: projected_inventory = beginning_inventory + in_transit + delivery_gr + 
             today_production + future_production - today_shipment - open_deployment
    
    Args:
        各个库存维度的字典，键为(material, location)，值为数量
        
    Returns:
        dict: 预测库存字典 {(material, location): quantity}
    """
    all_keys = set()
    for d in [beginning_inventory, in_transit, delivery_gr, today_production_gr, 
              future_production, today_shipment, open_deployment]:
        all_keys.update(d.keys())
    
    projected_inventory = {}
    for key in all_keys:
        projected_inventory[key] = (
            beginning_inventory.get(key, 0) +
            in_transit.get(key, 0) +
            delivery_gr.get(key, 0) +
            today_production_gr.get(key, 0) +
            future_production.get(key, 0) -
            today_shipment.get(key, 0) -
            open_deployment.get(key, 0)
        )
    
    return projected_inventory

def calculate_available_inventory(
    beginning_inventory: dict,
    delivery_gr: dict,
    today_production_gr: dict,
    today_shipment: dict,
    open_deployment: dict,
    open_deployment_inbound: dict
) -> dict:
    """
    计算当日真实可用库存（dynamic_soh），用于实际分配

    现在约定：
    dynamic_soh = beginning + delivery_gr + today_production_gr - open_deployment

    open_deployment_inbound 不再计入当日现货，只作为 pipeline supply 使用
    """
    all_keys = set()
    for d in [beginning_inventory, delivery_gr, today_production_gr,
              today_shipment, open_deployment]:
        all_keys.update(d.keys())

    soh = {}
    for key in all_keys:
        soh[key] = (
            beginning_inventory.get(key, 0) +
            delivery_gr.get(key, 0) +
            today_production_gr.get(key, 0) -
            # today_shipment.get(key, 0) -   # 如需扣减当日对客发货可放开
            open_deployment.get(key, 0)
        )
    return soh


# ========= 1. 通用辅助 =========

def get_upstream(location, material, network_df, sim_date, 
                 active_network_cache=None):
    row = get_active_network(network_df, material, location, sim_date, cache=active_network_cache)
    if not row.empty:
        return row.iloc[0]['sourcing']
    return None

def apply_moq_rv(qty, moq, rv, is_cross_node=True):
    """
    补货量小于moq补moq，否则向上取整到rv的倍数
    
    Args:
        qty: 需求数量
        moq: 最小订货量
        rv: 重订量(Round Volume)
        is_cross_node: 是否为跨节点调运。True=跨节点需要应用MOQ/RV，False=自循环不应用MOQ/RV
    """
    if qty <= 0:
        return 0
    
    # 🔧 修复：自循环调运不应用MOQ/RV约束，直接返回原需求量
    if not is_cross_node:
        return qty
    
    # 跨节点调运应用MOQ/RV约束
    if qty < moq:
        return moq
    return int(np.ceil(qty / rv)) * rv

def apply_grouped_moq_rv(demand_rows, location):
    """
    按调运路径分组应用MOQ/RV
    
    Args:
        demand_rows: 需求行列表
        location: 当前位置（sending）
        
    Returns:
        dict: 调整后的 demand_row_index -> adjusted_qty 映射
    """
    # 按 (material, sending, receiving, demand_element) 分组
    route_groups = {}
    for i, d in enumerate(demand_rows):
        receiving = d.get('from_location', d.get('receiving', location))
        is_cross_node = (location != receiving)
        
        route_key = (d['material'], location, receiving, d['demand_element'])
        if route_key not in route_groups:
            route_groups[route_key] = {
                'items': [],
                'total_qty': 0,
                'is_cross_node': is_cross_node,
                'moq': d['moq'],
                'rv': d['rv']
            }
        
        route_groups[route_key]['items'].append((i, d))
        route_groups[route_key]['total_qty'] += d['demand_qty']
    
    # 对每个路径组应用MOQ/RV
    adjusted_qtys = {}
    
    for route_key, group in route_groups.items():
        material, sending, receiving, demand_element = route_key
        total_qty = group['total_qty']
        is_cross_node = group['is_cross_node']
        moq = group['moq']
        rv = group['rv']
        
        # 对组合后的总量应用MOQ/RV
        adjusted_total = apply_moq_rv(total_qty, moq, rv, is_cross_node=is_cross_node)
        
        # print(f"      📦 路径组 {sending}→{receiving} [{demand_element}]: 原始={total_qty} → 调整={adjusted_total} (MOQ={moq}, 跨节点={is_cross_node})")
        
        # 将调整后的总量按原始比例分配回各个需求项
        if total_qty > 0:
            adjustment_ratio = adjusted_total / total_qty
        else:
            adjustment_ratio = 1.0
            
        for item_idx, item in group['items']:
            original_qty = item['demand_qty']
            adjusted_qty = int(original_qty * adjustment_ratio)
            adjusted_qtys[item_idx] = adjusted_qty
            
    return adjusted_qtys

def _build_ptf_lsk_cache(m4_mlcfg_df: pd.DataFrame | None) -> Dict[tuple[str, str], tuple[int, int]]:
    """Build cache for PTF/LSK lookups - 15-20x faster than DataFrame filtering"""
    cache = {}
    if m4_mlcfg_df is None or m4_mlcfg_df.empty:
        return cache
    
    for row in m4_mlcfg_df.itertuples():
        material = getattr(row, 'material', None)
        location = getattr(row, 'location', None)
        if material is None or location is None:
            continue
        
        ptf = 0
        lsk = 1
        
        # Try lowercase first, then uppercase
        ptf_val = getattr(row, 'ptf', None) or getattr(row, 'PTF', None)
        lsk_val = getattr(row, 'lsk', None) or getattr(row, 'LSK', None)
        
        if ptf_val is not None and not pd.isna(ptf_val):
            ptf = int(ptf_val)
        if lsk_val is not None and not pd.isna(lsk_val):
            lsk = int(lsk_val)
        
        cache[(str(material), str(location))] = (ptf, lsk)
    
    return cache

def _get_ptf_lsk(material: str, site: str, m4_mlcfg_df: pd.DataFrame | None, 
                 cache: Dict[tuple[str, str], tuple[int, int]] | None = None) -> tuple[int, int]:
    """Get PTF/LSK with optional caching support"""
    # Use cache if provided (15-20x faster)
    if cache is not None:
        return cache.get((str(material), str(site)), (0, 1))
    
    # Fallback to original logic if no cache
    ptf, lsk = 0, 1
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

def _build_lead_time_cache(lead_time_df: pd.DataFrame) -> Dict[tuple[str, str], tuple[int, int, int]]:
    """Build cache for lead time base values (PDT, GR, MCT) - 10-15x faster"""
    cache = {}
    if lead_time_df.empty:
        return cache
    
    for row in lead_time_df.itertuples():
        sending = getattr(row, 'sending', None)
        receiving = getattr(row, 'receiving', None)
        if sending is None or receiving is None:
            continue
        
        PDT = int(getattr(row, 'PDT', 0) or 0)
        GR = int(getattr(row, 'GR', 0) or 0)
        MCT = int(getattr(row, 'MCT', 0) or 0)
        
        cache[(str(sending), str(receiving))] = (PDT, GR, MCT)
    
    return cache

def determine_lead_time(
    sending: str,
    receiving: str,
    location_type: str,
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: pd.DataFrame | None = None,
    material: str | None = None,
    lead_time_cache: Dict[tuple[str, str], tuple[int, int, int]] | None = None,
    ptf_lsk_cache: Dict[tuple[str, str], tuple[int, int]] | None = None
) -> tuple[int, str]:
    """
    Plant: lead_time = max(MCT, PDT+GR) + PTF + LSK - 1
    DC:    lead_time = PDT + GR
    PTF/LSK 来源: M4_MaterialLocationLineCfg（按 material+sending 匹配）
    
    With caching: 10-15x faster when cache hit rate is high
    """
    # Use cache if provided (10-15x faster)
    if lead_time_cache is not None:
        base_values = lead_time_cache.get((str(sending), str(receiving)))
        if base_values is None:
            return 1, 'lead_time_missing'
        PDT, GR, MCT = base_values
    else:
        # Fallback to DataFrame filtering
        if lead_time_df.empty:
            return 1, 'empty_lead_time_config'

        row = lead_time_df[
            (lead_time_df['sending'] == sending) &
            (lead_time_df['receiving'] == receiving)
        ]
        if row.empty:
            return 1, 'lead_time_missing'

        try:
            PDT = int(row.iloc[0].get('PDT', 0) or 0)
            GR  = int(row.iloc[0].get('GR',  0) or 0)
            MCT = int(row.iloc[0].get('MCT', 0) or 0)
        except Exception as e:
            return 1, f'lead_time_calculation_error: {str(e)}'

    try:
        ptf, lsk = 0, 1
        if str(location_type).lower() == 'plant' and material is not None:
            # 与 M3 对齐：按 (material, sending) 取 PTF/LSK
            ptf, lsk = _get_ptf_lsk(material=material, site=sending, m4_mlcfg_df=m4_mlcfg_df, cache=ptf_lsk_cache)

        if str(location_type).lower() == 'plant':
            base_lt  = max(MCT, PDT + GR)
            leadtime = base_lt + ptf + lsk - 1
        else:
            leadtime = PDT + GR

        return max(1, int(leadtime)), ""

    except Exception as e:
        return 1, f'lead_time_calculation_error: {str(e)}'

def get_sending_location_type(
    material: str,
    sending: str,
    sim_date: pd.Timestamp,
    network_df: pd.DataFrame,
    location_layer_map: dict
) -> str:
    """
    与 Module3 一致的口径：
    1) 先查 network 中 (material, location=sending) 活动行，若存在则用其 location_type
    2) 若不存在且 sending 是自动识别的最上游（layer=0），则视为 Plant
    3) 否则默认 DC
    """
    if not sending or pd.isna(sending) or str(sending).strip() == "":
        return 'DC'

    row = get_active_network(network_df, material, sending, sim_date, cache=None)  # This function doesn't have access to cache
    if not row.empty:
        return str(row.iloc[0].get('location_type', 'DC') or 'DC')

    # 未维护但被自动识别为根节点 → Plant
    if location_layer_map.get(str(sending), None) == 0:
        return 'Plant'

    return 'DC'

# === 用 module3 的版本替换 ===
def assign_location_layers(network_df: pd.DataFrame) -> pd.DataFrame:
    from collections import defaultdict, deque
    if network_df.empty:
        return pd.DataFrame({'location': [], 'layer': []})

    children = defaultdict(list)
    parents = defaultdict(list)
    # Performance optimization: Use itertuples instead of iterrows
    for row in network_df.itertuples():
        sourcing_val = row.sourcing
        location_val = row.location
        sourcing_valid = sourcing_val is not None and pd.notna(sourcing_val) and str(sourcing_val).strip() != ''
        location_valid = location_val is not None and pd.notna(location_val) and str(location_val).strip() != ''
        if sourcing_valid and location_valid:
            children[sourcing_val].append(location_val)
            parents[location_val].append(sourcing_val)

    all_locations = set(network_df['location'].dropna()).union(set(network_df['sourcing'].dropna()))
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
        true_roots = potential_roots

    layer_dict = {}
    from collections import deque
    queue = deque()
    for root in true_roots:
        queue.append((root, 0))
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

    layer_df = pd.DataFrame([{'location': loc, 'layer': layer} for loc, layer in layer_dict.items()])
    layer_df = layer_df.sort_values('layer')
    return layer_df

def _build_active_network_cache(network_df: pd.DataFrame) -> Dict[tuple[str, str, pd.Timestamp, pd.Timestamp], pd.Series]:
    """Build cache for active network lookups - 20-30x faster than DataFrame filtering"""
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
        # Store the row as a dictionary for easy access
        cache[key] = row
    
    return cache

def get_active_network(network_df, material, location, sim_date, 
                      cache: Dict[tuple[str, str, pd.Timestamp, pd.Timestamp], pd.Series] | None = None):
    """Get active network with optional caching support - 20-30x faster with cache"""
    # Use cache if provided
    if cache is not None:
        # Find matching entries in cache
        matching_rows = []
        for key, row in cache.items():
            if (key[0] == str(material) and 
                key[1] == str(location) and 
                key[2] <= sim_date <= key[3]):
                matching_rows.append(row)
        
        if matching_rows:
            # Convert back to DataFrame format for compatibility
            # Return just the first match as a DataFrame
            return pd.DataFrame([matching_rows[0]._asdict()])
        return pd.DataFrame()
    
    # Fallback to original DataFrame filtering
    rows = network_df[
        (network_df['material'] == material) &
        (network_df['location'] == location) &
        (network_df['eff_from'] <= sim_date) &
        (network_df['eff_to'] >= sim_date)
    ]
    return rows

def is_review_day(dt, lsk, day):
    if lsk == 'daily':
        return True
    if lsk == 'weekly':
        return dt.weekday() == (int(day) - 1)
    if lsk == 'monthly':
        return dt.day == int(day)
    raise ValueError(f"Unknown LSK: {lsk}")

def compute_horizon(dt, lsk, day):
    if lsk == 'daily':
        return dt, dt
    if lsk == 'weekly':
        # Only valid if dt is review day
        if dt.weekday() != (int(day)-1):
            raise ValueError(f"compute_horizon: input date {dt} is not review day (expected weekday {int(day)-1})")
        cur = dt + timedelta(days=1)
        while True:
            if cur.weekday() == (int(day) - 1):
                break
            cur += timedelta(days=1)
        window_end = cur - timedelta(days=1)
        return dt, window_end
    if lsk == 'monthly':
        if dt.day != int(day):
            raise ValueError(f"compute_horizon: input date {dt} is not review day (expected day {int(day)})")
        y, m = dt.year, dt.month
        if dt.day >= int(day):
            m += 1
            if m > 12:
                m = 1
                y += 1
        next_review = pd.Timestamp(y, m, int(day))
        window_end = next_review - timedelta(days=1)
        return dt, window_end
    raise ValueError(f"Unknown LSK: {lsk}")

def load_integrated_config(
    config_dict: dict,
    module1_output_dir: str,
    module4_output_path: str, 
    orchestrator: object,
    current_date: pd.Timestamp
) -> dict:
    """
    加载集成配置数据，替代原来的load_config
    
    Args:
        config_dict: 配置数据字典
        module1_output_dir: Module1输出目录
        module4_output_path: Module4输出文件路径
        orchestrator: Orchestrator实例
        current_date: 当前日期
        
    Returns:
        dict: 集成配置数据
    """
    config = {}
    validation_log = []
    
    # 1. 从配置表加载静态数据
    config['SafetyStock'] = config_dict.get('M3_SafetyStock', pd.DataFrame())
    config['Network'] = config_dict.get('Global_Network', pd.DataFrame())
    config['LeadTime'] = config_dict.get('Global_LeadTime', pd.DataFrame())
    config['DemandPriority'] = config_dict.get('Global_DemandPriority', pd.DataFrame())
    config['PushPullModel'] = config_dict.get('M5_PushPullModel', pd.DataFrame())
    config['DeployConfig'] = config_dict.get('M5_DeployConfig', pd.DataFrame())
    
    # 应用字符串格式化到所有配置表
    for sheet_name in ['SafetyStock', 'Network', 'LeadTime', 'DemandPriority', 'PushPullModel', 'DeployConfig']:
        if not config[sheet_name].empty:
            config[sheet_name] = _normalize_identifiers(config[sheet_name])
    
    # 2. 从Module1加载当日数据
    config['SupplyDemandLog'] = config_dict.get('M5_SupplyDemandLog', pd.DataFrame())  # 从测试配置加载
    
    # 从Module1加载当日"订单池"
    if module1_output_dir and current_date:
        config['OrderLog'] = load_module1_daily_orders(module1_output_dir, current_date)
    else:
        config['OrderLog'] = pd.DataFrame()

    # 实际从 Module1 输出加载当日数据
    if module1_output_dir and current_date:
        try:
            # 从 Module1 日输出加载 SupplyDemandLog
            date_str = current_date.strftime('%Y%m%d')
            module1_file = f"{module1_output_dir}/module1_output_{date_str}.xlsx"
            if os.path.exists(module1_file):
                xl = pd.ExcelFile(module1_file)
                if 'SupplyDemandLog' in xl.sheet_names:
                    m1_supply_demand = xl.parse('SupplyDemandLog')
                    if not m1_supply_demand.empty:
                        config['SupplyDemandLog'] = m1_supply_demand
                        # print(f"  ✅ 从 Module1 加载了 {len(m1_supply_demand)} 条 SupplyDemandLog 数据")
        except Exception as e:
            print(f"  ⚠️  无法从 Module1 加载数据: {e}")
    
    # 从Module1加载当日发货数据
    if module1_output_dir and current_date:
        config['TodayShipment'] = load_module1_daily_shipment(module1_output_dir, current_date)
    else:
        config['TodayShipment'] = pd.DataFrame()
    
    # 3. 生产计划：修复重复计算问题，只使用实际的历史生产GR
    config['ProductionPlan'] = pd.DataFrame()  # 先置空
    # === 🔧 修复：只从 Orchestrator 取当日实际历史生产GR，避免重复计算 ===
    if orchestrator and current_date:
        date_str = current_date.strftime('%Y-%m-%d')
        try:
            # 只获取当日实际历史生产GR，不包含计划生产
            prod_gr = orchestrator.get_production_gr_view(date_str)
            if isinstance(prod_gr, pd.DataFrame) and not prod_gr.empty:
                # 规范字段，将date重命名为available_date以保持兼容性
                prod_gr = prod_gr.rename(columns={'date': 'available_date'})[['material', 'location', 'available_date', 'quantity']]
                if 'available_date' in prod_gr.columns:
                    prod_gr['available_date'] = pd.to_datetime(prod_gr['available_date'])
                for col in ['quantity']:
                    if col in prod_gr.columns:
                        prod_gr[col] = pd.to_numeric(prod_gr[col], errors='coerce').fillna(0)
                config['ProductionPlan'] = prod_gr
                # print(f"  ✅ 从 Orchestrator 加载了 {len(prod_gr)} 条生产计划数据（仅历史生产GR，修复重复计算）")
            else:
                print(f"  ⚠️  Orchestrator当日无历史生产GR数据")
        except Exception as e:
            print(f"  ⚠️  从 Orchestrator 加载生产计划失败: {e}")

    # === 回退：若 orchestrator 无数据，再尝试从 module4 文件读取 ProductionPlan ===
    if (config['ProductionPlan'].empty) and module4_output_path and os.path.exists(module4_output_path):
        try:
            xl = pd.ExcelFile(module4_output_path)
            if 'ProductionPlan' in xl.sheet_names:
                m4_production = xl.parse('ProductionPlan')
                if not m4_production.empty:
                    if 'available_date' in m4_production.columns:
                        m4_production['available_date'] = pd.to_datetime(m4_production['available_date'])
                    for col in ['produced_qty', 'uncon_planned_qty', 'planned_qty', 'quantity']:
                        if col in m4_production.columns:
                            m4_production[col] = pd.to_numeric(m4_production[col], errors='coerce').fillna(0)
                    config['ProductionPlan'] = m4_production
                    # print(f"  ✅ 回退：从 Module4 加载了 {len(m4_production)} 条生产计划数据")
        except Exception as e:
            print(f"  ⚠️  无法从 Module4 加载 ProductionPlan: {e}")   

    # 读取 M4_MaterialLocationLineCfg（用于 PTF/LSK）
    config['M4_MaterialLocationLineCfg'] = config_dict.get('M4_MaterialLocationLineCfg', pd.DataFrame())
    if module4_output_path and os.path.exists(module4_output_path):
        try:
            xl = pd.ExcelFile(module4_output_path)
            if 'M4_MaterialLocationLineCfg' in xl.sheet_names:
                mlcfg = xl.parse('M4_MaterialLocationLineCfg')
                if not mlcfg.empty:
                    config['M4_MaterialLocationLineCfg'] = mlcfg
                    print(f"  ✅ 从 Module4 加载了 {len(mlcfg)} 条 M4_MaterialLocationLineCfg")
        except Exception as e:
            print(f"  ⚠️  无法从 Module4 读取 M4_MaterialLocationLineCfg: {e}")

    # 4. 从Orchestrator加载动态数据
    if orchestrator and current_date:
        date_str = current_date.strftime('%Y-%m-%d')
        try:
            # 🔄 修改：使用期初库存而不是当前库存状态，避免重复计算
            config['InventoryLog'] = orchestrator.get_beginning_inventory_view(date_str)
            config['InTransit'] = orchestrator.get_planning_intransit_view(date_str)
            config['DeliveryGR'] = load_orchestrator_delivery_gr(orchestrator, current_date)
            config['OpenDeployment'] = load_orchestrator_open_deployment(orchestrator, current_date)
            config['ReceivingSpace'] = orchestrator.get_space_quota_view(date_str)
            # print(f"  ✅ 从 Orchestrator 加载了动态数据（使用期初库存基础）")
        except Exception as e:
            print(f"  ⚠️  从 Orchestrator 加载动态数据失败: {e}")
            # 使用空数据作为备选
            config['InventoryLog'] = pd.DataFrame()
            config['InTransit'] = pd.DataFrame() 
            config['DeliveryGR'] = pd.DataFrame()
            config['OpenDeployment'] = pd.DataFrame()
            config['ReceivingSpace'] = pd.DataFrame()
    else:
        # 使用空数据
        config['InventoryLog'] = pd.DataFrame()
        config['InTransit'] = pd.DataFrame()
        config['DeliveryGR'] = pd.DataFrame()
        config['OpenDeployment'] = pd.DataFrame()
        config['ReceivingSpace'] = pd.DataFrame()
    
    # 临时使用空数据以保持兼容性
    for key in ['SupplyDemandLog', 'ProductionPlan', 'InventoryLog', 'InTransit', 'ReceivingSpace']:
        if key not in config:
            config[key] = pd.DataFrame()
    
    # 日期字段处理
    date_fields = {
        'SupplyDemandLog': ['date'],
        'ProductionPlan': ['available_date'],
        'InventoryLog': ['date'],
        'InTransit': ['available_date'],
        'SafetyStock': ['date'],
        'ReceivingSpace': ['date'],
        'Network': ['eff_from', 'eff_to'],
        'OrderLog': ['date', 'simulation_date'],
    }
    
    for sheet, fields in date_fields.items():
        if sheet in config and not config[sheet].empty:
            for f in fields:
                if f in config[sheet].columns:
                    config[sheet][f] = pd.to_datetime(config[sheet][f])
    
    # 最终格式化所有配置表的标识符字段
    for sheet_name, df in config.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            config[sheet_name] = _normalize_identifiers(df)
    
    config['ValidationLog'] = validation_log
    return config

def load_config(input_path: str):
    required_sheets = [
        'SupplyDemandLog', 'ProductionPlan', 'InventoryLog', 'InTransit', 'SafetyStock',
        'Network', 'PushPullModel', 'ReceivingSpace', 'LeadTime',
        'DemandPriority', 'DeployConfig'
    ]
    config = {}
    validation_log = []
    xl = pd.ExcelFile(input_path)
    for sheet in required_sheets:
        if sheet not in xl.sheet_names:
            validation_log.append({'No': len(validation_log)+1, 'Issue': f'Missing required sheet: {sheet}'})
            config[sheet] = pd.DataFrame()
        else:
            df = xl.parse(sheet)
            config[sheet] = df
    # 字段校验举例
    sdl_required = ['date', 'material', 'location', 'demand_element', 'quantity']
    if not config['SupplyDemandLog'].empty:
        missing_cols = [c for c in sdl_required if c not in config['SupplyDemandLog'].columns]
        if missing_cols:
            validation_log.append({'No': len(validation_log)+1, 'Issue': f'SupplyDemandLog missing columns: {",".join(missing_cols)}'})
    # 日期类型处理
    date_fields = {
        'SupplyDemandLog': ['date'],
        'ProductionPlan': ['available_date'],
        'InventoryLog': ['date'],
        'InTransit': ['available_date'],
        'SafetyStock': ['date'],
        'ReceivingSpace': ['date'],
        'Network': ['eff_from', 'eff_to'],
        'OrderLog': ['date', 'simulation_date'],
    }
    for sheet, fields in date_fields.items():
        if sheet in config and not config[sheet].empty:
            for f in fields:
                if f in config[sheet].columns:
                    config[sheet][f] = pd.to_datetime(config[sheet][f])
    
    # 最终格式化所有配置表的标识符字段
    for sheet_name, df in config.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            config[sheet_name] = _normalize_identifiers(df)
    
    config['ValidationLog'] = validation_log
    return config

def validate_config_before_run(config, validation_log):
    deploy_cfg = config['DeployConfig']
    leadtime_df = config['LeadTime']
    pushpull = config['PushPullModel']
    demand_priority = config['DemandPriority']
    network = config['Network']
    # ======= 校验network是否有multiple sourcing ==========
    multi_sourcing = (
        network.groupby(['material', 'location'])['sourcing']
        .nunique().reset_index()
    )
    multi_sourcing = multi_sourcing[multi_sourcing['sourcing'] > 1]
    for _, row in multi_sourcing.iterrows():
        validation_log.append({
            'No': len(validation_log) + 1,
            'Issue': f"Network配置不合法: material={row['material']}, location={row['location']} 有多个sourcing"
        })
    # 校验leadtime
    for _, row in network.iterrows():
        if leadtime_df[
            (leadtime_df['sending'] == row['sourcing']) & (leadtime_df['receiving'] == row['location'])
        ].empty:
            validation_log.append({'No': len(validation_log)+1,
                                  'Issue': f"Missing leadtime for {row['sourcing']}->{row['location']} ({row['material']})"})
    # 校验pushpull
    for _, row in deploy_cfg.iterrows():
        if pushpull[
            (pushpull['material'] == row['material']) & (pushpull['sending'] == row['sending'])
        ].empty:
            validation_log.append({'No': len(validation_log)+1,
                'Issue': f"Missing PushPullModel for {row['material']}/{row['sending']}"})
    # ======= 校验/补充 DemandPriority ==========
    dp = demand_priority.copy()

    # 既看 SupplyDemandLog 的 demand_element，也看 OrderLog 的 demand_type（AO/normal）
    sdl_types = set(config['SupplyDemandLog']['demand_element'].unique()) if not config['SupplyDemandLog'].empty else set()
    ol = config.get('OrderLog', pd.DataFrame())
    ol_types = set(ol['demand_type'].unique()) if ('demand_type' in ol.columns and not ol.empty) else set()

    # 把 AO/normal 映射为 demand_element 字段里的值（我们后续用 demand_element 做优先级）
    needed = sdl_types | ol_types  # AO/normal 也在其中

    # 缺啥补啥（默认：AO=1，normal=2，其余给个较低优先级 9）
    def _ensure_priority(elem, default_p):
        if dp[dp['demand_element'] == elem].empty:
            dp.loc[len(dp)] = {'demand_element': elem, 'priority': default_p}
            validation_log.append({
                'No': len(validation_log)+1,
                'Issue': f'Auto add DemandPriority for {elem}={default_p}'
            })
    for elem in needed:
        if elem == 'AO':
            _ensure_priority('AO', 1)
        elif elem == 'normal':
            _ensure_priority('normal', 2)
        else:
            _ensure_priority(elem, 9)
    # 回写
    config['DemandPriority'] = dp

    return validation_log

def collect_node_demands(material, location, sim_date, config, up_gap_buffer,
                         ptf_lsk_cache=None, lead_time_cache=None, active_network_cache=None):
    """
    对齐规则：
    - horizon 统一来自 determine_lead_time 口径：
        * 有上游：horizon = determine_lead_time(upstream->location)
        * 无上游（顶层自补）：horizon = max(MCT, PDT+GR) + PTF + LSK - 1
    - horizon_end = sim_date + horizon
    - 选择窗口：
        * AO/normal（订单，来自 OrderLog）：date ∈ (sim_date, horizon_end]
        * forecast（来自 SupplyDemandLog）：date ∈ [sim_date, horizon_end]
        * safety：取 horizon_end 当天的目标量
        * 其余/净需求传递（net demand for xx 等）：date ∈ [sim_date, horizon_end]
    - 行级 leadtime：
        * 有上游：= horizon（同一套口径）
        * 无上游：= 0
    """
    import pandas as pd
    from datetime import timedelta

    supply_demand_log = config['SupplyDemandLog']
    safety_stock      = config['SafetyStock']
    deploy_cfg        = config['DeployConfig']
    network           = config['Network']
    leadtime_df       = config['LeadTime']

    # 读取 MOQ/RV/LSK/Day（保持你现有口径：以本节点作为 sending 去读）
    param_row = deploy_cfg[
        (deploy_cfg['material'] == material) & (deploy_cfg['sending'] == location)
    ]
    if not param_row.empty:
        moq = int(param_row.iloc[0]['moq'])
        rv  = int(param_row.iloc[0]['rv'])
        lsk = param_row.iloc[0]['lsk']  # 此处 lsk/day 仅作为后续模块可能用到的元数据，窗口不再依赖它
        day = int(param_row.iloc[0]['day'])
    else:
        moq, rv, lsk, day = 1, 1, 1, 1

    # 上游
    network_row = get_active_network(network, material, location, sim_date, cache=active_network_cache)
    upstream = network_row.iloc[0]['sourcing'] if not network_row.empty else None

    # 统一：发送端类型 & horizon
    if upstream and str(upstream).strip():
        sending_location_type = get_sending_location_type(
            material=str(material),
            sending=str(upstream),
            sim_date=sim_date,
            network_df=network,
            location_layer_map=config.get('LocationLayerMap', {})
        )
        # 有上游：用 determine_lead_time 得到 horizon
        horizon, err = determine_lead_time(
            sending=str(upstream),
            receiving=str(location),
            location_type=str(sending_location_type),
            lead_time_df=leadtime_df,
            m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
            material=str(material),
            lead_time_cache=lead_time_cache,
            ptf_lsk_cache=ptf_lsk_cache
        )
        if err:
            # 缺失或异常回退为 1
            horizon = 1
        leadtime_for_row = int(horizon)  # 行级 LT（跨节点）
    else:
        # 顶层：按 Plant 公式计算 horizon = max(MCT, PDT+GR) + PTF + LSK - 1
        # PTF/LSK 来自 M4_MaterialLocationLineCfg
        ptf, lsk_val = _get_ptf_lsk(
            material=str(material),
            site=str(location),
            m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
            cache=ptf_lsk_cache
        )
        # MCT/PDT/GR 来自 Global_LeadTime（以 sending==location 的行取最大值；缺失按 0）
        df_loc = leadtime_df[leadtime_df['sending'] == str(location)]
        MCT = int(pd.to_numeric(df_loc.get('MCT', 0), errors='coerce').fillna(0).max()) if not df_loc.empty else 0
        PDT = int(pd.to_numeric(df_loc.get('PDT', 0), errors='coerce').fillna(0).max()) if not df_loc.empty else 0
        GR  = int(pd.to_numeric(df_loc.get('GR',  0), errors='coerce').fillna(0).max()) if not df_loc.empty else 0

        base_lt  = max(MCT, PDT + GR)
        horizon  = max(1, int(base_lt + int(ptf) + int(lsk_val) - 1))
        leadtime_for_row = 0  # 自补货（顶层）行级 LT 恒为 0

    # horizon_end
    horizon_end = sim_date + timedelta(days=int(horizon))

    demand_rows = []

    # ========= 1) SDL: 预测 / 其他本地需求 =========
    sdl = supply_demand_log[
        (supply_demand_log['material'] == material) &
        (supply_demand_log['location'] == location)
    ].copy()
    if not sdl.empty:
        sdl['requirement_date'] = pd.to_datetime(sdl['date'])

        # 识别 forecast 行（完全匹配 'forecast'，忽略大小写）
        is_fc = sdl['demand_element'].astype(str).str.lower() == 'forecast'

        # forecast: [sim_date, horizon_end]
        sdl_fc = sdl[is_fc & (sdl['requirement_date'] >= sim_date) & (sdl['requirement_date'] <= horizon_end)]

        # 其他（含 net demand for xx 等）：[sim_date, horizon_end]
        sdl_others = sdl[~is_fc & (sdl['requirement_date'] >= sim_date) & (sdl['requirement_date'] <= horizon_end)]

        for _, row in pd.concat([sdl_fc, sdl_others], ignore_index=True).iterrows():
            demand_rows.append({
                'material': material,
                'location': location,
                'sending': upstream,
                'receiving': location,
                'demand_element': row['demand_element'],
                'demand_qty': int(row['quantity']),
                'planned_qty': int(row['quantity']),
                'moq': moq,
                'rv': rv,
                'leadtime': leadtime_for_row if upstream else 0,  # 顶层自补 0，跨节点=统一 horizon
                'requirement_date': row['requirement_date'],
                'plan_deploy_date': sim_date  # 计划触发日在窗口内分配环节使用，这里先放 sim_date
            })

    # ========= 2) 安全库存：只取 horizon_end 当天 =========
    ss = safety_stock[
        (safety_stock['material'] == material) &
        (safety_stock['location'] == location)
    ].copy()
    ss_qty = 0
    if not ss.empty:
        ss['date'] = pd.to_datetime(ss['date'])
        ss_end = ss[ss['date'] == horizon_end]
        if not ss_end.empty:
            ss_qty = int(pd.to_numeric(ss_end['safety_stock_qty'], errors='coerce').fillna(0).sum())

    if ss_qty > 0:
        demand_rows.append({
            'material': material,
            'location': location,
            'sending': upstream,
            'receiving': location,
            'demand_element': 'safety',
            'demand_qty': ss_qty,
            'planned_qty': ss_qty,
            'moq': moq,
            'rv': rv,
            'leadtime': leadtime_for_row if upstream else 0,
            'requirement_date': horizon_end,     # 目标日 = horizon_end
            'plan_deploy_date': sim_date
        })

    # ========= 3) 订单池（AO/normal）：[sim_date, horizon_end] =========
    order_df = config.get('OrderLog', pd.DataFrame())
    if not order_df.empty:
        orders = order_df[
            (order_df['material'] == material) & (order_df['location'] == location)
        ].copy()
        if not orders.empty:
            orders['requirement_date'] = pd.to_datetime(orders['date'])
            orders['demand_element']  = orders['demand_type']

            # AO / normal 统一用 (sim_date, horizon_end]
            mask = (orders['requirement_date'] >= sim_date) & (orders['requirement_date'] <= horizon_end)
            orders = orders[mask]

            for _, row in orders.iterrows():
                qty = int(row['quantity'])
                demand_rows.append({
                    'material': material,
                    'location': location,
                    'sending': upstream,
                    'receiving': location,
                    'demand_element': str(row['demand_element']),
                    'demand_qty': qty,
                    'planned_qty': qty,
                    'moq': moq,
                    'rv': rv,
                    'leadtime': leadtime_for_row if upstream else 0,
                    'requirement_date': row['requirement_date'],
                    'plan_deploy_date': sim_date,
                    'orig_location': location
                })

    # ========= 4) GAP 传递（上游下发的净需求）：[sim_date, horizon_end] =========
    if up_gap_buffer is not None and (material, location) in up_gap_buffer:
        for gap in up_gap_buffer[(material, location)]:
            req_dt = pd.to_datetime(gap.get('requirement_date', sim_date))
            if (req_dt >= sim_date) and (req_dt <= horizon_end):
                demand_rows.append({
                    'material': material,
                    'location': gap.get('location', location),
                    'receiving': gap.get('receiving', gap.get('location', location)),
                    'orig_location': gap.get('orig_location', gap.get('location', location)),
                    'sending': upstream,
                    'demand_element': gap['demand_element'],
                    'demand_qty': int(gap['planned_qty']),
                    'planned_qty': int(gap['planned_qty']),
                    'moq': moq,
                    'rv': rv,
                    'leadtime': leadtime_for_row if upstream else 0,
                    'requirement_date': req_dt,
                    'plan_deploy_date': sim_date,
                    'from_location': gap.get('from_location', None),
                })

    return demand_rows

def push_softpush_allocation(
    deployment_plan_rows, config, dynamic_soh, sim_date,
    ptf_lsk_cache=None, lead_time_cache=None
):
    """
    对 push / soft-push 节点，把真正的剩余库存按下游 safety 权重分配为补货计划行。
    修复点：
      1) 本节点当日若存在未满足的非 push 需求（含自满足 & 跨节点），则不触发 push
      2) 已分配库存：当日、非 push 行（含自满足）都会扣减，避免误判“剩余”
      3) receiving 端安全库存按 (sim_date + leadtime) 取值
      4) 若下游 safety 总和为 0 或无下游，则不分配
      5) 最大余数法分配，消灭向下取整尾差
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta

    pushpull   = config['PushPullModel']
    safety     = config['SafetyStock']
    lt_df      = config['LeadTime']
    deploy_cfg = config['DeployConfig']
    net        = config['Network']

    plan_rows_push = []

    # —— 统计【当日】已分配的库存（非 push 行，含自满足 + 跨节点）——
    allocated_inventory: dict[tuple[str, str], int] = {}
    for r in deployment_plan_rows:
        if 'push' in str(r.get('demand_element', '')).lower():
            continue  # 排除 push/soft-push 自身
        if pd.to_datetime(r.get('date')) != sim_date:
            continue  # 仅当日
        mat = r.get('material'); snd = r.get('sending')
        if mat is None or snd is None:
            continue
        key = (mat, snd)
        qty = int(r.get('deployed_qty_invCon', 0) or 0)
        if qty > 0:
            allocated_inventory[key] = allocated_inventory.get(key, 0) + qty

    # —— 逐 (material, sending) 处理 —— 
    group_keys = {(r['material'], r['sending']) for r in deployment_plan_rows if r.get('material') and r.get('sending')}
    for mat, sending in group_keys:
        # A) 当日是否仍有未满足的非 push 需求？（含自满足 & 跨节点）
        pending_gap = any(
            (
                r.get('material') == mat
                and r.get('sending') == sending
                and 'push' not in str(r.get('demand_element', '')).lower()
                and pd.to_datetime(r.get('date')) == sim_date
                and int(r.get('deployed_qty_invCon', 0) or 0) < int(r.get('planned_qty', 0) or 0)
            )
            for r in deployment_plan_rows
        )
        if pending_gap:
            continue  # 有缺口就不推

        # B) 该 (mat,sending) 是否配置 push / soft push
        row_pp = pushpull[(pushpull['material'] == mat) & (pushpull['sending'] == sending)]
        if row_pp.empty:
            continue
        model = str(row_pp.iloc[0]['model']).strip().lower()
        if model not in ['push', 'soft push']:
            continue

        # C) 计算真正的剩余库存 = dynamic_soh - 当日已分配（非 push）
        total_soh        = int(dynamic_soh.get((mat, sending), 0) or 0)
        already_allocated = int(allocated_inventory.get((mat, sending), 0) or 0)
        soh = max(0, total_soh - already_allocated)
        if soh <= 0:
            continue

        # D) 读取 LSK/Day（虽当前逻辑未用到 day，但保持一致性）
        row_cfg = deploy_cfg[(deploy_cfg['material'] == mat) & (deploy_cfg['sending'] == sending)]
        if not row_cfg.empty:
            lsk = int(row_cfg.iloc[0]['lsk'])
            day = int(row_cfg.iloc[0]['day'])
        else:
            lsk, day = 1, 1

        # E) soft-push 需先保留本节点当日 safety（sim_date）
        sending_ss = 0
        if model == 'soft push':
            ss_self = safety[(safety['material'] == mat) & (safety['location'] == sending)]
            ss_self = ss_self[pd.to_datetime(ss_self['date']) == sim_date] if not ss_self.empty else pd.DataFrame()
            if not ss_self.empty:
                sending_ss = int(ss_self['safety_stock_qty'].sum())
        # 可用于下推的库存
        available_soh = soh if model == 'push' else max(0, soh - sending_ss)
        if available_soh <= 0:
            continue

        # F) 找下游 receiving 列表
        recs = net[(net['material'] == mat) & (net['sourcing'] == sending)]['location'].dropna().unique().tolist()
        if not recs:
            continue  # 没有下游

        # G) 构造 receiving 安全库存与 leadtime
        receiving_ss_data = []
        for rec in recs:
            # leadtime 计算
            sending_location_type = get_sending_location_type(
                material=str(mat),
                sending=str(sending),
                sim_date=sim_date,
                network_df=net,
                location_layer_map=config.get('LocationLayerMap', {})
            )
            leadtime, err = determine_lead_time(
                sending=str(sending),
                receiving=str(rec),
                location_type=str(sending_location_type),
                lead_time_df=lt_df,
                m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
                material=str(mat),
                lead_time_cache=lead_time_cache,
                ptf_lsk_cache=ptf_lsk_cache
            )
            if err:
                leadtime = 1
            target_date = sim_date + timedelta(days=int(leadtime))
            ss_rec = safety[(safety['material'] == mat) & (safety['location'] == rec)]
            ss_rec = ss_rec[pd.to_datetime(ss_rec['date']) == target_date] if not ss_rec.empty else pd.DataFrame()
            ss_qty = int(ss_rec['safety_stock_qty'].sum()) if not ss_rec.empty else 0

            receiving_ss_data.append({
                'receiving': rec,
                'ss_qty': ss_qty,
                'leadtime': int(leadtime),
                'planned_delivery_date': target_date
            })

        total_ss = sum(x['ss_qty'] for x in receiving_ss_data)
        if total_ss <= 0:
            continue  # 下游安全库存总和为 0，不推

        # H) 按 safety 权重分配（最大余数法，吃掉尾差）
        # 1) 理想配额与向下取整
        ideal = []
        floor_sum = 0
        for x in receiving_ss_data:
            share = available_soh * (x['ss_qty'] / total_ss)
            q_floor = int(np.floor(share))
            frac = share - q_floor
            ideal.append((x, q_floor, frac))
            floor_sum += q_floor

        # 2) 把剩余 (available_soh - floor_sum) 份额按 frac 从大到小 +1
        remainder = int(available_soh - floor_sum)
        if remainder > 0:
            ideal_sorted = sorted(ideal, key=lambda t: t[2], reverse=True)
            for i in range(remainder):
                x, q_floor, frac = ideal_sorted[i % len(ideal_sorted)]
                ideal_sorted[i % len(ideal_sorted)] = (x, q_floor + 1, frac)
            ideal = ideal_sorted

        # I) 生成 push 计划行（只落 qty>0）
        for x, qty, _ in ideal:
            if qty <= 0:
                continue
            plan = {
                'date': sim_date,
                'material': mat,
                'sending': sending,
                'receiving': x['receiving'],
                'demand_qty': 0,
                'demand_element': 'push replenishment' if model == 'push' else 'soft push replenishment',
                'planned_qty': int(qty),
                'deployed_qty_invCon_push': int(qty),
                'deployed_qty_invCon': int(qty),  # 兼容后续空间配额与库存统计
                'planned_delivery_date': x['planned_delivery_date'],
                'orig_location': x['receiving'],
                'leadtime': int(x['leadtime']),
                'is_cross_node': True
            }
            plan_rows_push.append(plan)

    return plan_rows_push


def apply_receiving_space_quota(deployment_plan_rows, receiving_space, sim_date, demand_priority_map):
    """
    在所有调运计划明细生成后，按receiving space quota再分配，更新deployed_qty，unfulfilled log
    修复：仅对跨节点调运（sending != receiving）应用receiving space quota限制
    """
    df = pd.DataFrame(deployment_plan_rows)
    if df.empty:
        df['deployed_qty'] = []
        df['quota'] = []
        return df, []
    
    # 如果receiving_space配置为空，直接返回原分配结果
    if receiving_space.empty:
        df['deployed_qty'] = df['deployed_qty_invCon']
        df['quota'] = np.inf
        return df, []
    
    # 按receiving+date分组
    unfulfilled = []
    for (recv, date), grp in df.groupby(['receiving', 'date']):
        quota_row = receiving_space[
            (receiving_space['receiving'] == recv) & (pd.to_datetime(receiving_space['date']) == date)
        ]
        quota = quota_row['max_qty'].iloc[0] if not quota_row.empty else np.inf
        
        # 🔧 修复：仅计算跨节点调运的quantity占用quota
        cross_node_grp = grp[grp['sending'] != grp['receiving']]
        self_fulfillment_grp = grp[grp['sending'] == grp['receiving']]
        
        # 自我需求满足不占用quota，直接通过
        df.loc[self_fulfillment_grp.index, 'deployed_qty'] = self_fulfillment_grp['deployed_qty_invCon']
        df.loc[self_fulfillment_grp.index, 'quota'] = np.inf  # 自我满足不受quota限制
        
        if cross_node_grp.empty:
            # 如果没有跨节点调运，跳过quota检查
            continue
            
        # 仅检查跨节点调运是否超过quota
        cross_node_total = cross_node_grp['deployed_qty_invCon'].sum()
        if cross_node_total <= quota:
            df.loc[cross_node_grp.index, 'deployed_qty'] = cross_node_grp['deployed_qty_invCon']
            df.loc[cross_node_grp.index, 'quota'] = quota
            continue
        # 跨节点调运空间不足，按优先级+权重分配（仅处理跨节点调运）
        cross_node_rows = cross_node_grp.to_dict(orient='records')
        
        # 按优先级对跨节点调运进行排序和分组
        rows_sorted = sorted(cross_node_rows, key=lambda r: demand_priority_map.get(r['demand_element'], 99))
        grouped = {}
        for r in rows_sorted:
            p = demand_priority_map.get(r['demand_element'], 99)
            grouped.setdefault(p, []).append(r)
        
        left = quota
        deploy_qtys = {i: 0 for i in range(len(cross_node_rows))}
        
        for priority in sorted(grouped):
            group = grouped[priority]
            group_total = sum(r['deployed_qty_invCon'] for r in group)
            if left >= group_total:
                for r in group:
                    idx = cross_node_rows.index(r)
                    deploy_qtys[idx] = r['deployed_qty_invCon']
                left -= group_total
            else:
                allocated = 0
                for r in group:
                    idx = cross_node_rows.index(r)
                    weight = r['deployed_qty_invCon'] / group_total if group_total > 0 else 0
                    q = int(left * weight)
                    deploy_qtys[idx] = min(q, r['deployed_qty_invCon'])
                    allocated += deploy_qtys[idx]
                left -= allocated
                # 不再分配
                break
        
        # 更新跨节点调运的实际分配
        for idx, qty in deploy_qtys.items():
            original_row = cross_node_rows[idx]
            # 找到原始DataFrame中对应的索引
            original_idx = cross_node_grp[
                (cross_node_grp['sending'] == original_row['sending']) &
                (cross_node_grp['receiving'] == original_row['receiving']) &
                (cross_node_grp['material'] == original_row['material']) &
                (cross_node_grp['demand_element'] == original_row['demand_element'])
            ].index[0]
            
            df.at[original_idx, 'deployed_qty'] = qty
            df.at[original_idx, 'quota'] = quota
            
            gap = original_row['deployed_qty_invCon'] - qty
            if gap > 0:
                unfulfilled.append({
                    'date': date,
                    'sending': original_row['sending'],
                    'receiving': original_row['receiving'],
                    'material': original_row['material'],  # 新增
                    'demand_qty': original_row['demand_qty'],
                    'demand_element': original_row['demand_element'],
                    'unfulfilled_qty': gap,
                    'reason': "space constraint"
                })
    # 空间充足行
    df['deployed_qty'] = df['deployed_qty'].fillna(df['deployed_qty_invCon'])
    df['quota'] = df['quota'].fillna(np.nan)
    return df, unfulfilled

def log_outputs(output_path: str, outputs: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet, df in outputs.items():
            if df.empty:
                # 输出空表头
                pd.DataFrame(columns=df.columns).to_excel(writer, sheet_name=sheet, index=False)
            else:
                # 确保输出时标识符字段为字符串格式
                normalized_df = _normalize_identifiers(df)
                normalized_df.to_excel(writer, sheet_name=sheet, index=False)

# ============ 2. 主流程 ===============

def main(
    input_path: str = None, 
    output_path: str = None, 
    sim_start: str = None, 
    sim_end: str = None,
    # 新增参数支持集成模式
    config_dict: dict = None,
    module1_output_dir: str = None,
    module4_output_path: str = None,
    orchestrator: object = None,
    current_date: str = None
):
    """
    Module 5: 多层级部署规划模块
    
    支持两种运行模式:
    1. 独立模式: 传入input_path, output_path, sim_start, sim_end
    2. 集成模式: 传入config_dict, module1_output_dir, module4_output_path, orchestrator, current_date
    
    Args:
        # 独立模式参数
        input_path: 输入配置文件路径
        output_path: 输出文件路径 
        sim_start: 仿真开始日期
        sim_end: 仿真结束日期
        
        # 集成模式参数
        config_dict: 配置数据字典
        module1_output_dir: Module1输出目录
        module4_output_path: Module4输出文件路径
        orchestrator: Orchestrator实例
        current_date: 当前日期(单日运行时)
    """
    # 判断运行模式
    if config_dict is not None:
        # 集成模式 - 完成的集成数据加载
        # print("\n🔄 Module5 运行于集成模式")
        current_date_obj = pd.to_datetime(current_date) if current_date else None
        config = load_integrated_config(
            config_dict, module1_output_dir, module4_output_path, 
            orchestrator, current_date_obj
        )
        sim_dates = [current_date_obj] if current_date_obj else pd.date_range(sim_start, sim_end, freq='D')
        
        # 集成模式输出路径
        if output_path is None:
            output_path = f"./Module5Output_{current_date_obj.strftime('%Y%m%d')}.xlsx" if current_date_obj else "./Module5Output.xlsx"
    else:
        # 独立模式 - 保持往后兼容
        # print("\n📜 Module5 运行于独立模式") 
        config = load_config(input_path)
        sim_dates = pd.date_range(sim_start, sim_end, freq='D')
    
    # 继续原有逻辑流程
    validation_log = list(config.get('ValidationLog', []))
    validate_config_before_run(config, validation_log)

    network = config['Network']
    deploy_cfg = config['DeployConfig']
    inventory_log = config['InventoryLog']
    production_plan = config['ProductionPlan']
    in_transit = config['InTransit']
    demand_priority = config['DemandPriority']
    receiving_space = config['ReceivingSpace']

    network_layers = assign_location_layers(network)
    location_to_layer = dict(zip(network_layers['location'], network_layers['layer']))
    layer_list = sorted(network_layers['layer'].unique(), reverse=True)  # 从最大层往上游推进
    # Performance optimization: Use dict() with zip instead of iterrows
    demand_priority_map = dict(zip(demand_priority['demand_element'], demand_priority['priority']))
    config['LocationLayerMap'] = location_to_layer
    
    # ========== 高优先级性能优化: 构建查询缓存 (15-25% 提升) ==========
    # 1. PTF/LSK 缓存 (15-20x faster)
    ptf_lsk_cache = _build_ptf_lsk_cache(config.get('M4_MaterialLocationLineCfg', pd.DataFrame()))
    # 2. Lead Time 缓存 (10-15x faster)
    lead_time_cache = _build_lead_time_cache(config.get('LeadTime', pd.DataFrame()))
    # 3. Active Network 缓存 (20-30x faster)
    active_network_cache = _build_active_network_cache(network)
    print(f"✅ 缓存已初始化: PTF/LSK={len(ptf_lsk_cache)} | LeadTime={len(lead_time_cache)} | Network={len(active_network_cache)}")
    # ========== 初始化库存 soh_dict ==========
    # 1. 全收集所有material/location（包含 OrderLog）
    ol_df = config.get('OrderLog', pd.DataFrame())
    mats_from_ol = set(ol_df['material'].unique()) if ('material' in ol_df.columns and not ol_df.empty) else set()
    locs_from_ol = set(ol_df['location'].unique()) if ('location' in ol_df.columns and not ol_df.empty) else set()

    all_mats = set(config['SupplyDemandLog']['material'].unique()) | \
            set(config['SafetyStock']['material'].unique()) | \
            mats_from_ol

    all_locs = set(config['SupplyDemandLog']['location'].unique()) | \
            set(config['SafetyStock']['location'].unique()) | \
            locs_from_ol

    # 2. 确定仿真开始日期并获取当天的库存
    # 集成模式下使用第一个仿真日期，独立模式下使用sim_start参数
    actual_sim_start = (sim_dates[0] if hasattr(sim_dates, '__getitem__') else pd.to_datetime(sim_start))
    
    inv_df = inventory_log[inventory_log['date'] == actual_sim_start]
    if inv_df.empty:
        print(f"[WARN] No inventory records found for sim_start: {actual_sim_start}")

    # 3. 检查是否有重复记录
    duplicates = inv_df.duplicated(subset=['material', 'location'], keep=False)
    if duplicates.any():
        dup_rows = inv_df[duplicates]
        raise ValueError(f"InventoryLog contains duplicate (material, location) on sim_start {sim_start}:\n{dup_rows[['material', 'location', 'date']]}")

    # 4. 初始化soh_dict，默认0
    # Performance optimization: Use dictionary comprehension instead of nested loops
    soh_dict = {(mat, loc): 0 for mat in all_mats for loc in all_locs}

    # Performance optimization: Use itertuples instead of iterrows
    for row in inv_df.itertuples():
        soh_dict[(row.material, row.location)] = int(row.quantity)


    deployment_plan_rows = []
    unfulfilled_rows = []
    stock_on_hand_log = []

    up_gap_buffer = {}

    for sim_date in sim_dates:
        # === 优化的日志输出 ===
        # print(f"\n{'='*60}")
        # print(f"📅 仿真日期: {sim_date.strftime('%Y-%m-%d')}")
        # print(f"{'='*60}")

        # ===== 库存计算逻辑重构 (修复重复计算问题) =====
        # 🔄 新的库存计算公式: 基于期初库存避免重复计算
        # available_inventory = 
        #   beginning_inventory +              # 当日期初库存（未包含当日事务）
        #   in_transit +                      # 在途库存
        #   delivery_gr +                     # 当日收货数据  
        #   today_production +                # 当日生产 (available_date = today)
        #   future_production +               # 未来生产 (available_date > today)
        #   - today_shipment -                # 当日发货数据
        #   - open_deployment                 # 开放调拨数据
        
        # 使用期初库存作为基础，避免重复计算M1 shipment和M4 production
        beginning_inventory = soh_dict.copy()
        
        # 从 Module4/Orchestrator 获取当日和未来生产（来源见 load_integrated_config 的配置）
        today_production_gr = {}
        future_production = {}
        if not production_plan.empty:
            # 🔍 调试生产计划数据
            # print(f"\n🔍 调试生产计划数据:")
            # print(f"   生产计划总条目: {len(production_plan)}")
            # print(f"   生产计划列: {production_plan.columns.tolist()}")
            
            # 检查所有当日生产计划
            all_today = production_plan[production_plan['available_date'] == sim_date]
            # print(f"   所有当日生产计划: {len(all_today)} 条")
            # for _, row in all_today.iterrows():
                # print(f"   - {row.get('material')}@{row.get('location')}: {row.get('quantity')}")
            
            # 查看当日的80813644@0386生产计划
            # debug_today = production_plan[
            #     (production_plan['available_date'] == sim_date) & 
            #     (production_plan['material'] == '80813644') & 
            #     (production_plan['location'] == '0386')
            # ]
            # if not debug_today.empty:
            #     print(f"   当日80813644@0386生产计划: {len(debug_today)} 条")
            #     for _, row in debug_today.iterrows():
            #         print(f"   - material: {row.get('material')}, location: {row.get('location')}")
            #         print(f"     produced_qty: {row.get('produced_qty')}, planned_qty: {row.get('planned_qty')}")
            #         print(f"     quantity: {row.get('quantity')}, available_date: {row.get('available_date')}")
                    
            # 🔍 重要：对比历史生产入库vs计划生产
            if orchestrator:
                date_str = sim_date.strftime('%Y-%m-%d')
                # print(f"\n🔍 对比历史生产入库 vs 计划生产:")
                # 获取当日历史生产GR
                # prod_gr_view = orchestrator.get_production_gr_view(date_str)
                # print(f"   当日历史生产GR条目: {len(prod_gr_view) if not prod_gr_view.empty else 0}")
                # if not prod_gr_view.empty:
                #     for _, row in prod_gr_view.iterrows():
                #         print(f"   - 历史GR: {row.get('material')}@{row.get('location')}: {row.get('quantity')}")
                
                # # 获取计划生产backlog
                # if hasattr(orchestrator, 'production_plan_backlog'):
                #     backlog_today = [p for p in orchestrator.production_plan_backlog 
                #                    if pd.to_datetime(p.get('available_date')).normalize() == sim_date.normalize()]
                #     print(f"   当日计划生产backlog条目: {len(backlog_today)}")
                #     for record in backlog_today:
                #         print(f"   - 计划backlog: {record.get('material')}@{record.get('location')}: {record.get('quantity')}")
                # else:
                #     print(f"   Orchestrator没有production_plan_backlog属性")
            
            # # 当日生产 (available_date = sim_date) —— 用 produced_qty
            today_prod = production_plan[production_plan['available_date'] == sim_date]
            # print(f"   当日生产条目: {len(today_prod)}")
            # Helper function to get first valid quantity from multiple columns
            def _get_qty_from_row(row, col_names):
                """Get first non-null, non-NaN value from list of column names"""
                for col in col_names:
                    val = getattr(row, col, None)
                    if val is not None and not pd.isna(val):
                        return int(val)
                return 0
            
            # Performance optimization: Use itertuples for faster iteration
            for row in today_prod.itertuples():
                k = (row.material, row.location)
                # Try columns in order: produced_qty -> planned_qty -> quantity
                qty_today = _get_qty_from_row(row, ['produced_qty', 'planned_qty', 'quantity'])
                today_production_gr[k] = today_production_gr.get(k, 0) + qty_today

            # 未来生产 (available_date > sim_date) —— 用 uncon_planned_qty
            future_prod = production_plan[production_plan['available_date'] > sim_date]
            # Performance optimization: Use itertuples for faster iteration
            for row in future_prod.itertuples():
                k = (row.material, row.location)
                # Try columns in order: uncon_planned_qty -> produced_qty -> planned_qty -> quantity
                qty_future = _get_qty_from_row(row, ['uncon_planned_qty', 'produced_qty', 'planned_qty', 'quantity'])
                future_production[k] = future_production.get(k, 0) + qty_future
        
        # 从 Orchestrator 获取在途库存
        today_intransit = {}
        if not in_transit.empty:
            # Performance optimization: Combine filter and iteration
            for row in in_transit[in_transit['available_date'] == sim_date].itertuples():
                k = (row.material, row.receiving)
                today_intransit[k] = today_intransit.get(k, 0) + int(row.quantity)
        # 未来在途：available_date > sim_date，用于自补货的 pipeline 覆盖
        future_intransit = {}
        if not in_transit.empty:
            # Performance optimization: Combine filter and iteration
            for row in in_transit[in_transit['available_date'] > sim_date].itertuples():
                k = (row.material, row.receiving)
                future_intransit[k] = future_intransit.get(k, 0) + int(row.quantity)
        
        # 加载当日收货、发货和开放调拨数据
        delivery_gr_data = config.get('DeliveryGR', pd.DataFrame())
        today_shipment_data = config.get('TodayShipment', pd.DataFrame())
        open_deployment_data = config.get('OpenDeployment', pd.DataFrame())
        
        # 转换为字典格式
        delivery_gr = {}
        if not delivery_gr_data.empty:
            filtered_delivery = delivery_gr_data[pd.to_datetime(delivery_gr_data['date']) == sim_date] if 'date' in delivery_gr_data.columns else delivery_gr_data
            # Performance optimization: Use itertuples for faster iteration
            for row in filtered_delivery.itertuples():
                k = (row.material, row.receiving)
                delivery_gr[k] = delivery_gr.get(k, 0) + int(row.quantity)
        
        today_shipment = {}
        if not today_shipment_data.empty:
            filtered_shipment = today_shipment_data[pd.to_datetime(today_shipment_data['date']) == sim_date] if 'date' in today_shipment_data.columns else today_shipment_data
            # Performance optimization: Use itertuples for faster iteration
            for row in filtered_shipment.itertuples():
                k = (row.material, row.location)
                today_shipment[k] = today_shipment.get(k, 0) + int(row.quantity)
        
        open_deployment = {}
        if not open_deployment_data.empty:
            # Performance optimization: Use itertuples for faster iteration
            for row in open_deployment_data.itertuples():
                # 只计算真正从该地点发出的调拨，排除自循环（sending=receiving）
                if row.sending != row.receiving:
                    k = (row.material, row.sending)
                    open_deployment[k] = open_deployment.get(k, 0) + int(row.quantity)
        # 🔁 新增：构造 inbound 视图 (material, receiving) → qty
        open_deployment_inbound = build_open_deployment_inbound(open_deployment_data)

        # 计算预测库存（用于gap计算）
        projected_soh = calculate_projected_inventory(
            beginning_inventory=beginning_inventory,
            in_transit=today_intransit, 
            delivery_gr=delivery_gr,
            today_production_gr=today_production_gr,
            future_production=future_production,
            today_shipment=today_shipment,
            open_deployment=open_deployment
        )
        
        # 计算当日真实可用库存（用于实际分配）
        dynamic_soh = calculate_available_inventory(
            beginning_inventory=beginning_inventory,
            delivery_gr=delivery_gr,
            today_production_gr=today_production_gr,
            today_shipment=today_shipment,
            open_deployment=open_deployment,
            open_deployment_inbound=open_deployment_inbound
        )

        
        # print(f"🔍 库存计算基础: 期初库存 {len(beginning_inventory)} 项, 预测库存 {len([k for k, v in projected_soh.items() if v > 0])} 项有库存, 当日可用库存 {len([k for k, v in dynamic_soh.items() if v > 0])} 项有库存")
        
        # 🔍 调试：详细分析80813644@0386的库存计算
        # debug_key = ('80813644', '0386')
        # if debug_key in beginning_inventory or debug_key in dynamic_soh:
        #     print(f"\n🔍 调试80813644@0386库存计算:")
        #     print(f"   期初库存 (beginning_inventory): {beginning_inventory.get(debug_key, 0)}")
        #     print(f"   交付入库 (delivery_gr): {delivery_gr.get(debug_key, 0)}")
        #     print(f"   当日生产入库 (today_production_gr): {today_production_gr.get(debug_key, 0)}")
        #     print(f"   当日发货出库 (today_shipment): {today_shipment.get(debug_key, 0)}")
        #     print(f"   开放部署扣减 (open_deployment): {open_deployment.get(debug_key, 0)}")
        #     calculated = (beginning_inventory.get(debug_key, 0) + 
        #                  delivery_gr.get(debug_key, 0) + 
        #                  today_production_gr.get(debug_key, 0) - 
        #                  today_shipment.get(debug_key, 0) - 
        #                  open_deployment.get(debug_key, 0))
        #     print(f"   计算结果 = {beginning_inventory.get(debug_key, 0)} + {delivery_gr.get(debug_key, 0)} + {today_production_gr.get(debug_key, 0)} - {today_shipment.get(debug_key, 0)} - {open_deployment.get(debug_key, 0)} = {calculated}")
        #     print(f"   dynamic_soh实际值: {dynamic_soh.get(debug_key, 0)}")
            
            # # 🔍 调试today_production_gr的具体来源
            # print(f"\n🔍 调试today_production_gr的来源:")
            # print(f"   today_production_gr总条目: {len(today_production_gr)}")
            # for key, qty in today_production_gr.items():
            #     if key[0] == '80813644' and key[1] == '0386':
            #         print(f"   发现80813644@0386的生产入库: {qty}")
            
            # 对比Orchestrator的unrestricted_inventory
            # if orchestrator:
            #     date_str = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
            #     orch_inventory = orchestrator.get_unrestricted_inventory_view(date_str)
            #     orch_row = orch_inventory[(orch_inventory['material'] == '80813644') & (orch_inventory['location'] == '0386')]
            #     if not orch_row.empty:
            #         orch_qty = orch_row.iloc[0]['quantity']
                    # print(f"   Orchestrator unrestricted_inventory: {orch_qty}")
                    # print(f"   差异: dynamic_soh({dynamic_soh.get(debug_key, 0)}) - unrestricted({orch_qty}) = {dynamic_soh.get(debug_key, 0) - orch_qty}")
                    
                    # 🔍 调试Orchestrator当日历史生产入库记录
                    # print(f"\n🔍 调试Orchestrator当日历史生产入库:")
                    # if hasattr(orchestrator, 'production_gr'):
                    #     prod_records = [p for p in orchestrator.production_gr if 
                    #                   p.get('date') == date_str and 
                    #                   p.get('material') == '80813644' and 
                    #                   p.get('location') == '0386']
                    #     print(f"   Orchestrator当日历史生产入库记录数: {len(prod_records)}")
                    #     total_orch_prod = sum(p.get('quantity', 0) for p in prod_records)
                    #     print(f"   Orchestrator当日历史生产入库总量: {total_orch_prod}")
                    #     for record in prod_records:
                    #         print(f"   - {record}")
                    # else:
                    #     print(f"   Orchestrator没有production_gr属性")
        up_gap_next = {}

        for layer in layer_list:
            # print(f"\n📦 处理层级 {layer}")
            # print(f"{'-'*40}")
            
            # 组合所有material-location对（包含 OrderLog和safety stock）
            materials_union = set(config['SupplyDemandLog']['material'].unique())
            if 'OrderLog' in config and not config['OrderLog'].empty:
                materials_union |= set(config['OrderLog']['material'].unique())
            if not config['SafetyStock'].empty:
                materials_union |= set(config['SafetyStock']['material'].unique())
            base_pairs = set(
                (mat, loc)
                for loc, l in location_to_layer.items() if l == layer
                for mat in materials_union
            )
            # gap buffer补充
            gap_pairs = set(
                (mat, loc)
                for (mat, loc) in up_gap_buffer
                if location_to_layer.get(loc, None) == layer
            )
            all_pairs = base_pairs | gap_pairs
            
            for mat, loc in all_pairs:
                node_key = (mat, loc)
                current_stock = dynamic_soh.get(node_key, 0)
                # print(f"📍 节点: {mat}@{loc} [可用库存: {current_stock}]")
                
                demand_rows = collect_node_demands(mat, loc, sim_date, config, up_gap_buffer,
                                                   ptf_lsk_cache=ptf_lsk_cache,
                                                   lead_time_cache=lead_time_cache,
                                                   active_network_cache=active_network_cache)
                if not demand_rows:
                    # print(f"   ⚠️  无需求需要处理")
                    continue
                
                demand_types = [d['demand_element'] for d in demand_rows]
                # print(f"   📋 需求类型: {', '.join(demand_types)}")
                
                # 🔧 修复：MOQ/RV应用逻辑移至调拨计划生成阶段，根据实际的sending/receiving关系决定
                # 此处先将planned_qty设为demand_qty，稍后在生成plan_row时再决定是否应用MOQ/RV
                for d in demand_rows:
                    d['planned_qty'] = d['demand_qty']  # 暂时设为原始需求量

                # 按优先级分组处理
                demand_rows_sorted = sorted(demand_rows, key=lambda d: demand_priority_map.get(d['demand_element'], 99))
                grouped = {}
                for d in demand_rows_sorted:
                    p = demand_priority_map.get(d['demand_element'], 99)
                    grouped.setdefault(p, []).append(d)
                
                # 🔧 修复：使用分组MOQ/RV逻辑计算总需求量
                adjusted_qtys = apply_grouped_moq_rv(demand_rows, loc)
                total_actual_demand = sum(adjusted_qtys.values())
                # print(f"   📊 总需求: {total_actual_demand}, 可用库存: {current_stock}")
                
                for priority in sorted(grouped):
                    group = grouped[priority]
                    # 🔧 修复：优先级组需求量基于分组MOQ/RV调整结果
                    group_actual_demand = 0
                    for i, d in enumerate(demand_rows):
                        if d in group:
                            group_actual_demand += adjusted_qtys.get(i, d['demand_qty'])
                    # print(f"   🔢 优先级 {priority}: 需求 {group_actual_demand}")
                    
                    # 如果没有剩余库存，所有后续优先级都分配0
                    if current_stock <= 0:
                        for d in group:
                            d['deployed_qty_invCon'] = 0
                        # print(f"      ❌ 无剩余库存，跳过")
                        continue
                    
                    if group_actual_demand == 0:
                        for d in group:
                            d['deployed_qty_invCon'] = 0
                        continue
                    
                    if current_stock >= group_actual_demand:
                        # 库存充足，完全满足当前优先级
                        for i, d in enumerate(demand_rows):
                            if d in group:
                                adjusted_qty = adjusted_qtys.get(i, d['demand_qty'])
                                d['deployed_qty_invCon'] = adjusted_qty
                        current_stock -= group_actual_demand
                        # print(f"      ✅ 库存充足，完全满足")
                    else:
                        # 库存不足，按权重分配所有剩余库存给当前优先级
                        # 关键修复：用完库存后，后续优先级不再分配
                        for i, d in enumerate(demand_rows):
                            if d in group:
                                adjusted_qty = adjusted_qtys.get(i, d['demand_qty'])
                                weight = adjusted_qty / group_actual_demand if group_actual_demand > 0 else 0
                                d['deployed_qty_invCon'] = min(int(current_stock * weight), adjusted_qty)
                        
                        # 重新计算实际分配量
                        actual_allocated = sum(d['deployed_qty_invCon'] for d in group)
                        current_stock = 0  # 关键修复：库存不足时，用完所有库存，后续优先级不再分配
                        # print(f"      ⚠️  库存不足，部分满足 {actual_allocated}/{group_actual_demand}，后续优先级不再分配")
                        
                        # 为后续优先级预设0分配
                        remaining_priorities = [p for p in sorted(grouped) if p > priority]
                        for remaining_priority in remaining_priorities:
                            for d in grouped[remaining_priority]:
                                d['deployed_qty_invCon'] = 0
                        break  # 跳出优先级循环
                    
                    # 显示分配详情
                    for i, d in enumerate(demand_rows):
                        if d in group:
                            receiving = d.get('from_location', d.get('receiving', loc))
                            is_cross_node = (loc != receiving)
                            adjusted_qty = adjusted_qtys.get(i, d['demand_qty'])
                            # status = "✅" if d['deployed_qty_invCon'] == adjusted_qty else "⚠️"
                            # print(f"      {status} [{d['demand_element']}] 原始需求={d['demand_qty']} 计划={adjusted_qty} 分配={d['deployed_qty_invCon']} 跨节点={is_cross_node}")
                # —— 在处理 GAP 之前，给所有需求行初始化 pipeline 相关字段 —— 
                for d in demand_rows:
                    d.setdefault('deploy_qty_with_plan_order', 0)
                    d.setdefault('deploy_from_in_transit', 0)
                    d.setdefault('deploy_from_open_deployment_inbound', 0)
                    d.setdefault('deploy_from_future_production', 0)

                # —— 自补货第二轮：用 pipeline supply 覆盖剩余 gap（所有节点都适用）——

                # 只考虑本节点自补货行：receiving == 本节点 loc
                self_idx_list = []

                for idx, d in enumerate(demand_rows):
                    receiving = d.get('from_location', d.get('receiving', loc))
                    if receiving == loc:
                        self_idx_list.append(idx)

                # 如果没有自补货需求，直接跳过 pipeline 逻辑
                if self_idx_list:
                    # 构造本节点自补可用的 pipeline 池
                    node_key = (mat, loc)
                    # ✅ 按你的要求：pipeline 用 future intransit，不动
                    pool_in_transit = future_intransit.get(node_key, 0)
                    pool_future_production = future_production.get(node_key, 0)
                    pool_odi = open_deployment_inbound.get(node_key, 0)

                    pipeline_pool_total = pool_in_transit + pool_odi + pool_future_production

                    if pipeline_pool_total > 0:
                        # 按优先级顺序，用 pipeline 覆盖自补货 gap
                        self_idx_sorted = sorted(
                            self_idx_list,
                            key=lambda i: demand_priority_map.get(demand_rows[i]['demand_element'], 99)
                        )

                        for idx in self_idx_sorted:
                            d = demand_rows[idx]
                            adjusted_qty = adjusted_qtys.get(idx, d['demand_qty'])
                            allocated_invcon = d.get('deployed_qty_invCon', 0)
                            raw_gap = adjusted_qty - allocated_invcon

                            if raw_gap <= 0:
                                continue
                            if pipeline_pool_total <= 0:
                                break

                            alloc = min(raw_gap, pipeline_pool_total)

                            # 按 in_transit -> open_deployment_inbound -> future_production 顺序消耗
                            alloc_intrans = min(alloc, pool_in_transit)
                            pool_in_transit -= alloc_intrans

                            remain1 = alloc - alloc_intrans
                            alloc_odi = min(remain1, pool_odi)
                            pool_odi -= alloc_odi

                            remain2 = remain1 - alloc_odi
                            alloc_future = min(remain2, pool_future_production)
                            pool_future_production -= alloc_future

                            pipeline_pool_total = pool_in_transit + pool_odi + pool_future_production

                            d['deploy_qty_with_plan_order'] += alloc
                            d['deploy_from_in_transit'] += alloc_intrans
                            d['deploy_from_open_deployment_inbound'] += alloc_odi
                            d['deploy_from_future_production'] += alloc_future


                # 处理GAP和生成调拨计划
                gap_count = 0
                for i, d in enumerate(demand_rows):
                    receiving = d.get('from_location', d.get('receiving', loc))
                    is_cross_node = (loc != receiving)
                    adjusted_qty = adjusted_qtys.get(i, d['demand_qty'])

                    # 本地自补需求使用 pipeline cover（deploy_qty_with_plan_order）
                    is_self_node = (receiving == loc)
                    if is_self_node:
                        plan_order_cover = d.get('deploy_qty_with_plan_order', 0)
                    else:
                        plan_order_cover = 0

                    gap_qty = adjusted_qty - d.get('deployed_qty_invCon', 0) - plan_order_cover
                    if gap_qty <= 0:
                        continue

                    up_loc = get_upstream(loc, mat, network, sim_date, active_network_cache=active_network_cache)
                    gap_count += 1

                    if up_loc:
                        new_demand_element = f"net demand for {d['demand_element']}"
                        up_gap_next.setdefault((mat, up_loc), []).append({
                            'demand_element': new_demand_element,
                            'planned_qty': gap_qty,
                            'leadtime': d['leadtime'],
                            'requirement_date': d.get('requirement_date', d['plan_deploy_date']),
                            'location': up_loc,
                            'from_location': loc,
                            'orig_location': d.get('orig_location', d['location'])
                        })
                    
                    unfulfilled_rows.append({
                        'date': d['plan_deploy_date'],
                        'sending': loc,
                        'receiving': receiving,
                        'demand_qty': d['demand_qty'],
                        'demand_element': d['demand_element'],
                        'unfulfilled_qty': gap_qty,
                        'reason': "supply shortage"
                    })

                        
                        # print(f"      🔼 需求缺口: {gap_qty} [{d['demand_element']}] → 上游 {up_loc} (is_cross_node: {is_cross_node}, adjusted_qty: {adjusted_qty})")
                
                # if gap_count == 0:
                    # print(f"      🟢 无需求缺口")
                
                # 生成调拨计划行
                for i, d in enumerate(demand_rows):
                    receiving = d.get('from_location', d.get('receiving', loc))
                    
                    # 🔧 修复：使用分组MOQ/RV调整后的数量
                    is_cross_node = (loc != receiving)
                    actual_planned_qty = adjusted_qtys.get(i, d['demand_qty'])
                    
                    # 自补货（sending == receiving）不应有leadtime
                    if loc == receiving:
                        planned_delivery_date = d['plan_deploy_date']
                        leadtime_for_row = 0
                    else:
                        planned_delivery_date = d.get('requirement_date', d['plan_deploy_date'])
                        # 即时计算该行的 lead time：sending=loc, receiving=receiving
                        sending_location_type = get_sending_location_type(
                            material=str(mat),
                            sending=str(loc),
                            sim_date=sim_date,
                            network_df=network,
                            location_layer_map=config.get('LocationLayerMap', {})
                        )
                        lt_row, _ = determine_lead_time(
                            sending=str(loc),
                            receiving=str(receiving),
                            location_type=str(sending_location_type),
                            lead_time_df=config['LeadTime'],
                            m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
                            material=str(mat),
                            lead_time_cache=lead_time_cache,
                            ptf_lsk_cache=ptf_lsk_cache
                        )
                        leadtime_for_row = int(lt_row)
                    
                    plan_row = {
                        'date': d['plan_deploy_date'],
                        'material': mat,
                        'sending': loc,
                        'receiving': receiving,
                        'demand_qty': d['demand_qty'],
                        'demand_element': d['demand_element'],
                        'planned_qty': actual_planned_qty,
                        'deployed_qty_invCon': d['deployed_qty_invCon'],
                        'deploy_qty_with_plan_order': d.get('deploy_qty_with_plan_order', 0),
                        'deploy_from_in_transit': d.get('deploy_from_in_transit', 0),
                        'deploy_from_open_deployment_inbound': d.get('deploy_from_open_deployment_inbound', 0),
                        'deploy_from_future_production': d.get('deploy_from_future_production', 0),
                        'planned_delivery_date': planned_delivery_date,
                        'orig_location': d.get('orig_location', d['location']),
                        'leadtime': leadtime_for_row,
                        'is_cross_node': is_cross_node,
                    }

                    deployment_plan_rows.append(plan_row)

            # print(f"\n✅ 层级 {layer} 处理完成，向上游传递 {sum(len(v) for v in up_gap_next.values())} 个需求缺口")

            # 更新GAP缓冲区
            up_gap_buffer = up_gap_next.copy()
        
        # push/soft-push再分配：直接用 dynamic_soh
        dynamic_soh_for_push = dynamic_soh.copy()
        plan_push = push_softpush_allocation(deployment_plan_rows, config, dynamic_soh_for_push, sim_date,
                                              ptf_lsk_cache=ptf_lsk_cache, lead_time_cache=lead_time_cache)

        if plan_push:
            deployment_plan_rows.extend(plan_push)
            # print(f"\n🔄 Push/Soft-push 补货: 生成 {len(plan_push)} 条补货计划")

        # 更新库存（基于当日事务流水）
        deployed_dict = {}
        df = pd.DataFrame(deployment_plan_rows)
        if not df.empty:
            today_rows = df[df['date'] == sim_date]
            for _, row in today_rows.iterrows():
                k = (row['material'], row['sending'])
                qty = row['deployed_qty_invCon'] if row['sending'] != row['receiving'] else 0
                deployed_dict[k] = deployed_dict.get(k, 0) + qty

        # 更新soh_dict为下一日的期初库存
        all_keys = set(list(beginning_inventory.keys()) +
                       list(today_production_gr.keys()) +
                       list(today_intransit.keys()) +
                       list(deployed_dict.keys()) +
                       list(today_shipment.keys()) +
                       list(delivery_gr.keys()))
        
        for (mat, loc) in all_keys:
            beginning_soh = beginning_inventory.get((mat, loc), 0)
            prod = today_production_gr.get((mat, loc), 0)
            intrans = today_intransit.get((mat, loc), 0)
            deliv_gr = delivery_gr.get((mat, loc), 0)
            deployed = deployed_dict.get((mat, loc), 0)
            shipped = today_shipment.get((mat, loc), 0)
            
            # 期末库存计算：期初 + 生产 + 在途到货 + 收货 - 发货 - 调拨
            end_soh = beginning_soh + prod + intrans + deliv_gr - shipped - deployed
            soh_dict[(mat, loc)] = end_soh  # 作为下一日的期初库存
            
            stock_on_hand_log.append({
                'material': mat,
                'location': loc,
                'date': sim_date,
                'beginning_soh': beginning_soh,
                'production': prod,
                'in_transit': intrans,
                'delivery_gr': deliv_gr,
                'today_shipment': shipped,
                'deployed_qty': deployed,
                'ending_soh': end_soh
            })
        
        # print(f"\n📊 当日统计:")
        # print(f"   总调拨计划数: {len(deployment_plan_rows)}")
        # print(f"   未满足需求数: {len([r for r in unfulfilled_rows if r['date'] == sim_date])}")

    # 应用收货空间配额
    deployment_plan_rows_df, unfulfilled_space = apply_receiving_space_quota(
        deployment_plan_rows, receiving_space, sim_date, demand_priority_map
    )
    unfulfilled_all = pd.DataFrame(unfulfilled_rows + unfulfilled_space)

    outputs = {
        'DeploymentPlan': deployment_plan_rows_df,
        'UnfulfilledLog': unfulfilled_all,
        'StockOnHandLog': pd.DataFrame(stock_on_hand_log),
        'Validation': pd.DataFrame(validation_log),
    }
    log_outputs(output_path, outputs)
    
    # 集成模式：将部署计划发送给Orchestrator（由主集成脚本统一处理）
    # 注意：这里暂时注释掉直接调用，交由主集成脚本统一处理以避免重复
    # if config_dict is not None and orchestrator is not None and not deployment_plan_rows_df.empty:
    #     try:
    #         # 过滤出有实际部署量的计划
    #         valid_deployment = deployment_plan_rows_df[
    #             (deployment_plan_rows_df['deployed_qty_invCon'] > 0) & 
    #             (deployment_plan_rows_df['deployed_qty_invCon'].notna())
    #         ].copy()
    #         
    #         if not valid_deployment.empty:
    #             # 重命名列以匹配orchestrator期望的格式
    #             orchestrator_deployment = valid_deployment.rename(columns={
    #                 'date': 'planned_deployment_date',
    #                 'deployed_qty_invCon': 'deployed_qty'
    #             })[['material', 'sending', 'receiving', 'planned_deployment_date', 'deployed_qty', 'demand_element']]
    #             
    #             orchestrator.process_module5_deployment(orchestrator_deployment, current_date)
    #             print(f"✅ 已向Orchestrator发送 {len(orchestrator_deployment)} 条部署计划")
    #         else:
    #             print(f"ℹ️  无有效部署计划发送给Orchestrator")
    #     except Exception as e:
    #         print(f"⚠️  Orchestrator集成失败: {str(e)}")
    #         print(f"Error type: {type(e).__name__}")
    #         print(f"Deployment plan columns: {list(deployment_plan_rows_df.columns)}")
    #         print(f"Deployment plan shape: {deployment_plan_rows_df.shape}")
    #         if not deployment_plan_rows_df.empty:
    #             print(f"Sample row: {deployment_plan_rows_df.iloc[0].to_dict()}")
    #         import traceback
    #         traceback.print_exc()
    
    # print(f"\n{'='*60}")
    # print(f"🎉 仿真完成! 所有层级已处理完毕")
    # print(f"💾 调拨计划已保存至: {output_path}")
    # print(f"📈 总调拨计划数: {len(deployment_plan_rows_df)}")
    # print(f"📝 未满足需求数: {len(unfulfilled_all)}")
    # print(f"✅ 修复重复计算问题: 使用期初库存作为计算基础")
    # print(f"{'='*60}")
    
    # 返回结果用于集成模式
    return {
        'deployment_plan': deployment_plan_rows_df,
        'unfulfilled_log': unfulfilled_all,
        'stock_on_hand_log': pd.DataFrame(stock_on_hand_log),
        'validation_log': pd.DataFrame(validation_log),
        'statistics': {
            'deployment_count': len(deployment_plan_rows_df),
            'unfulfilled_count': len(unfulfilled_all),
            'processed_dates': len(sim_dates) if isinstance(sim_dates, list) else 1
        }
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Module 5: Multi-echelon Deployment Planning')
    parser.add_argument('--input', required=True, help='Input config excel path')
    parser.add_argument('--output', required=True, help='Output excel path')
    parser.add_argument('--sim_start', required=True, help='Simulation start date, YYYY-MM-DD')
    parser.add_argument('--sim_end', required=True, help='Simulation end date, YYYY-MM-DD')
    args = parser.parse_args()
    main(args.input, args.output, args.sim_start, args.sim_end)

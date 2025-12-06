import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import lru_cache

# 从 module5 导入 MOQ/RV 应用逻辑
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
    
    # 自循环调运不应用MOQ/RV约束，直接返回原需求量
    if not is_cross_node:
        return qty
    
    # 跨节点调运应用MOQ/RV约束
    if qty < moq:
        return moq
    return int(np.ceil(qty / rv)) * rv

# 标识符字段标准化函数（与main_integration.py保持一致）
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
            # Apply normalization for location-type fields
            if col in ['location', 'sending', 'receiving']:
                df[col] = df[col].apply(_normalize_location)
            # Apply normalization for material
            elif col in ['material']:
                df[col] = df[col].apply(_normalize_material)
            # For other identifier columns, vectorized string conversion
            else:
                df[col] = df[col].fillna('').astype(str)
    
    return df

def load_config(config_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load configuration data from Excel file
    Uses Global_Network's location_type field directly
    
    Args:
        config_path: Path to the configuration Excel file
        
    Returns:
        Dictionary of configuration DataFrames
    """
    sheet_mapping = {
        'M3_SafetyStock': ('safety_stock', pd.DataFrame()),
        'Global_Network': ('network_config', pd.DataFrame()),
        'Global_LeadTime': ('lead_time_config', pd.DataFrame())
    }
    
    try:
        xl = pd.ExcelFile(config_path)
        loaded_config = {}
        
        for sheet_name, (key, default) in sheet_mapping.items():
            if sheet_name in xl.sheet_names:
                loaded_config[key] = xl.parse(sheet_name)
                # Convert date columns if they exist
                if key in ['safety_stock'] and 'date' in loaded_config[key].columns:
                    loaded_config[key]['date'] = pd.to_datetime(loaded_config[key]['date'])
                elif key in ['network_config']:
                    if 'eff_from' in loaded_config[key].columns:
                        loaded_config[key]['eff_from'] = pd.to_datetime(loaded_config[key]['eff_from'])
                        loaded_config[key]['eff_to'] = pd.to_datetime(loaded_config[key]['eff_to'])
                    # 标准化标识符字段
                    loaded_config[key] = _normalize_identifiers(loaded_config[key])
            else:
                loaded_config[key] = default
                
        return loaded_config
    except Exception as e:
        raise RuntimeError(f"Failed to load module3 config from {config_path}: {e}")


def load_module1_daily_outputs(module1_output_dir: str, simulation_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """
    读取 Module1 当天版本的输出：
      - SupplyDemandLog: 已被订单消耗后的未来预测（M1已按 sim_date 生成）
      - ShipmentLog: 仅当日发货（date == sim_date），用于从可用量中扣减
      - OrderLog: 当天版本视图（包含历史生成但未来到期的订单 + 当天新单）
                  这里不再按 simulation_date 过滤，只做基本类型规范化；
                  未来是否纳入需求由 M3 在计算前再筛选（只取 date > sim_date）。
    """
    try:
        _t_func = time.perf_counter()
        date_str = simulation_date.strftime('%Y%m%d')
        f1 = os.path.join(module1_output_dir, f"module1_output_{date_str}.xlsx")
        f2 = os.path.join(module1_output_dir, f"output_simulation_{date_str}.xlsx")
        module1_daily_file = f1 if os.path.exists(f1) else f2

        if not os.path.exists(module1_daily_file):
            print(f"Warning: Module1 output file not found for date {date_str}. Using empty DataFrames.")
            return {'supply_demand_df': pd.DataFrame(), 'shipment_df': pd.DataFrame(), 'order_df': pd.DataFrame()}

        xl = pd.ExcelFile(module1_daily_file)
        module1_data = {}

        def _read(name):
            return xl.parse(name) if name in xl.sheet_names else pd.DataFrame()

        # 1) SupplyDemandLog：原样读取（M1已保证为“未来剩余预测”）
        sdl = _read('SupplyDemandLog')
        if not sdl.empty and 'date' in sdl.columns:
            sdl['date'] = pd.to_datetime(sdl['date'])
        # standardize key fields as strings
        # 标准化标识符字段
        sdl = _normalize_identifiers(sdl)

        # 2) ShipmentLog：仅保留当日
        shp = _read('ShipmentLog')
        if not shp.empty and 'date' in shp.columns:
            shp['date'] = pd.to_datetime(shp['date'])
            shp = shp[shp['date'] == simulation_date].copy()
        # 标准化标识符字段
        shp = _normalize_identifiers(shp)

        # 3) OrderLog：当天版本全量（包含未来订单 + 当天新单）
        odl = _read('OrderLog')
        if not odl.empty:
            if 'date' in odl.columns:
                odl['date'] = pd.to_datetime(odl['date'])
            if 'simulation_date' in odl.columns:
                odl['simulation_date'] = pd.to_datetime(odl['simulation_date'])
            # 不按 simulation_date 过滤，保留当天版本全量；后续在 M3 内部再筛 date > sim_date
            # 标准化标识符字段
            odl = _normalize_identifiers(odl)
        module1_data['supply_demand_df'] = sdl
        module1_data['shipment_df'] = shp
        module1_data['order_df'] = odl
        print(f"[M3] load_module1_daily_outputs total: {time.perf_counter()-_t_func:.3f}s for {simulation_date.date()}")
        return module1_data

    except Exception as e:
        print(f"Warning: Error loading Module1 daily outputs for {simulation_date.strftime('%Y-%m-%d')}: {e}")
        return {'supply_demand_df': pd.DataFrame(), 'shipment_df': pd.DataFrame(), 'order_df': pd.DataFrame()}


def assign_location_layers(network_df: pd.DataFrame) -> pd.DataFrame:
    """分配供应链网络中各节点的层级 - 自动识别最上层节点
    
    Args:
        network_df: 网络配置数据，包含sourcing和location字段
        
    Returns:
        DataFrame: 包含location和对应layer的映射关系
    """
    _t_func = time.perf_counter()
    if network_df.empty:
        return pd.DataFrame({'location': [], 'layer': []})
        
    children = defaultdict(list)
    parents = defaultdict(list)
    
    # 第一步：构建父子关系图
    # Performance optimization: Use itertuples instead of iterrows
    for row in network_df.itertuples():
        sourcing_val = row.sourcing
        location_val = row.location
        
        # Handle null/nan values properly for both scalar and Series cases
        sourcing_valid = sourcing_val is not None and pd.notna(sourcing_val) and str(sourcing_val).strip() != ''
        location_valid = location_val is not None and pd.notna(location_val) and str(location_val).strip() != ''
        
        if sourcing_valid and location_valid:
            children[sourcing_val].append(location_val)
            parents[location_val].append(sourcing_val)
    
    # 第二步：收集所有地点
    all_locations = set(network_df['location'].dropna()).union(set(network_df['sourcing'].dropna()))
    
    # 第三步：自动识别最上层节点（没有父节点的节点）
    # 这些节点可能是真正的根节点，也可能是配置中缺失上游关系的节点
    potential_roots = [loc for loc in all_locations if not parents[loc]]
    
    # 第四步：智能识别真正的根节点
    # 策略：如果一个地点在sourcing中出现过，说明它有下游，可能是真正的根节点
    # 如果一个地点只在location中出现，从未在sourcing中出现，说明它可能是叶子节点
    true_roots = []
    for loc in potential_roots:
        if loc in children:  # 该地点有下游节点
            true_roots.append(loc)
        else:
            # 该地点没有下游，可能是叶子节点，需要进一步分析
            # 检查是否有其他地点指向它
            has_incoming = any(loc in parents.get(other_loc, []) for other_loc in all_locations)
            if not has_incoming:
                # 如果没有任何其他地点指向它，且它也没有下游，可能是孤立的根节点
                true_roots.append(loc)
    
    # 如果没有找到真正的根节点，使用所有potential_roots
    if not true_roots:
        true_roots = potential_roots
    
    # print(f"🔍 自动识别网络层级:")
    # print(f"  总地点数: {len(all_locations)}")
    # print(f"  潜在根节点: {potential_roots}")
    # print(f"  识别出的根节点: {true_roots}")
    
    # 第五步：从根节点开始分配层级
    layer_dict = {}
    queue = deque()
    
    # 根节点从layer 0开始
    for root in true_roots:
        queue.append((root, 0))
        # print(f"  📍 根节点: {root} -> Layer 0")
    
    # 广度优先遍历分配层级
    while queue:
        loc, layer = queue.popleft()
        if loc in layer_dict and layer_dict[loc] <= layer:
            continue
        layer_dict[loc] = layer
        
        # 子节点层级 = 父节点层级 + 1
        for child in children.get(loc, []):
            queue.append((child, layer + 1))
            # print(f"  📍 子节点: {child} -> Layer {layer + 1} (父节点: {loc})")
    
    # 第六步：处理未连接或孤立的节点
    unassigned = [loc for loc in all_locations if loc not in layer_dict]
    if unassigned:
        max_layer = max(layer_dict.values()) if layer_dict else 0
        for loc in unassigned:
            layer_dict[loc] = max_layer + 1
            # print(f"  📍 孤立节点: {loc} -> Layer {max_layer + 1}")
    
    # 第七步：生成层级映射DataFrame
    layer_df = pd.DataFrame([
        {'location': loc, 'layer': layer} 
        for loc, layer in layer_dict.items()
    ])
    
    # 按层级排序
    layer_df = layer_df.sort_values('layer')
    
    # print(f"  ✅ 层级分配完成，共 {len(layer_df)} 个地点")
    # print(f"  层级范围: {layer_df['layer'].min()} - {layer_df['layer'].max()}")
    
    print(f"[M3] assign_location_layers total: {time.perf_counter()-_t_func:.3f}s, locations={len(layer_df)}")
    return layer_df

# === 新增：放在 assign_location_layers 之后 ===
def infer_sending_location_type(
    network_df: pd.DataFrame,
    location_layer_df: pd.DataFrame,
    sending: str,
    material: str | None,
    sim_date: pd.Timestamp
) -> str:
    """
    推断发送端的 location_type：
    1) 若存在 (material, location==sending) 的显式配置，直接使用其 location_type
    2) 若 sending 是根节点(layer==0)，判为 'Plant'
    3) 若 sending 只在 sourcing 列出现、从不在 location 列出现，判为 'Plant'
    4) 其他情况默认为 'DC'
    """
    if sending is None or (isinstance(sending, float) and pd.isna(sending)) or str(sending).strip() == '':
        return 'DC'

    # ① 显式配置（同物料、有效期内）
    if material is not None and not network_df.empty:
        explicit = network_df[
            (network_df['material'] == material) &
            (network_df['location'] == sending) &
            (network_df['eff_from'] <= sim_date) &
            (network_df['eff_to'] >= sim_date)
        ]
        if not explicit.empty:
            t = explicit.iloc[0].get('location_type', None)
            if isinstance(t, str) and t.strip():
                return t

    # ② 根节点（layer==0）→ Plant
    if not location_layer_df.empty:
        layer_map = dict(zip(location_layer_df['location'], location_layer_df['layer']))
        if layer_map.get(sending, None) == 0:
            return 'Plant'

    # ③ 只在 sourcing 中出现、从不在 location 中出现 → Plant
    #    （处理“源头 Plant 只维护在 sourcing 列”的常见情况）
    appears_as_sourcing = network_df['sourcing'].astype(str).eq(str(sending)).any()
    appears_as_location = network_df['location'].astype(str).eq(str(sending)).any()
    if appears_as_sourcing and not appears_as_location:
        return 'Plant'

    # ④ 兜底
    return 'DC'

def _build_ptf_lsk_cache_m3(m4_mlcfg_df: pd.DataFrame | None) -> Dict[tuple[str, str], tuple[int, int]]:
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
    """
    从 M4_MaterialLocationLineCfg 读取 (PTF, LSK)
    - 表结构字段：material, location, ..., lsk, ptf, day, MCT
    - 兼容大小写列名（lsk/LSK, ptf/PTF）
    - 未命中时默认 PTF=0, LSK=1
    
    With caching: 15-20x faster
    """
    # Use cache if provided (15-20x faster)
    if cache is not None:
        return cache.get((str(material), str(site)), (0, 1))
    
    # Fallback to original logic
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

    # PTF
    if 'ptf' in ml.columns and pd.notna(row.get('ptf')):
        ptf = int(row['ptf'])
    elif 'PTF' in ml.columns and pd.notna(row.get('PTF')):
        ptf = int(row['PTF'])

    # LSK
    if 'lsk' in ml.columns and pd.notna(row.get('lsk')):
        lsk = int(row['lsk'])
    elif 'LSK' in ml.columns and pd.notna(row.get('LSK')):
        lsk = int(row['LSK'])

    return ptf, lsk
def _compute_root_horizon(material: str,
                          location: str,
                          lead_time_df: pd.DataFrame,
                          m4_mlcfg_df: pd.DataFrame | None = None,
                          ptf_lsk_cache: Dict[tuple[str, str], tuple[int, int]] | None = None) -> int:
    """
    顶层(无上游)窗口口径 — 与 Module 5 对齐：
    horizon = max(PDT+GR, MCT) + PTF + LSK - 1
    - PDT/GR/MCT：从 Global_LeadTime 取 sending==location 的所有行的最大值（缺失当 0）
    - PTF/LSK：从 M4_MaterialLocationLineCfg 按 (material, location) 读取；未命中默认 PTF=0, LSK=1
    - 最终保证 horizon >= 1
    
    With caching: 15-20x faster
    """
    import pandas as pd

    # 1) PTF/LSK
    ptf, lsk = _get_ptf_lsk(material=material, site=location, m4_mlcfg_df=m4_mlcfg_df, cache=ptf_lsk_cache)

    # 2) PDT/GR/MCT（以 sending==location 的行取最大值）
    if lead_time_df is None or lead_time_df.empty:
        PDT = GR = MCT = 0
    else:
        df_loc = lead_time_df[lead_time_df['sending'].astype(str) == str(location)]
        if df_loc.empty:
            PDT = GR = MCT = 0
        else:
            PDT = int(pd.to_numeric(df_loc.get('PDT', 0), errors='coerce').fillna(0).max())
            GR  = int(pd.to_numeric(df_loc.get('GR',  0), errors='coerce').fillna(0).max())
            MCT = int(pd.to_numeric(df_loc.get('MCT', 0), errors='coerce').fillna(0).max())

    base_lt = max(MCT, PDT + GR)
    horizon = max(1, int(base_lt + int(ptf) + int(lsk) - 1))
    return horizon

def determine_lead_time(
    sending: str,
    receiving: str,
    location_type: str,                 # 传入“发送端”的类型；Plant 逻辑用它判断
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: pd.DataFrame | None = None,
    material: str | None = None,
    ptf_lsk_cache: Dict[tuple[str, str], tuple[int, int]] | None = None
) -> tuple[int, str]:
    """
    提前期：
      - PDT/GR/MCT 来自 Global_LeadTime（按 sending+receiving 匹配）
      - 对于 Plant（发送端为 Plant）：lead_time = max(MCT, PDT+GR) + PTF + LSK - 1
        其中 PTF/LSK 从 M4_MaterialLocationLineCfg 取（列：ptf, lsk；兼容大小写）
      - 对于 DC：lead_time = PDT + GR
    
    With caching: 15-20x faster for PTF/LSK lookups
    """
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

        # 默认不加 PTF/LSK；仅 Plant 时读取
        ptf, lsk = 0, 1
        if str(location_type).lower() == 'plant' and material is not None:
            # 口径：按 (material, sending)匹配
            ptf, lsk = _get_ptf_lsk(material=material, site=sending, m4_mlcfg_df=m4_mlcfg_df, cache=ptf_lsk_cache)

        if str(location_type).lower() == 'plant':
            base_lt  = max(MCT, PDT + GR)
            leadtime = base_lt + ptf + lsk - 1
        else:
            leadtime = PDT + GR

        # 添加调试信息以追踪问题
        final_leadtime = max(0, int(leadtime))
        # if material == '80813644' and receiving == 'C816':
        #     print(f"    Debug: determine_lead_time for {material}@{receiving}")
        #     print(f"      sending={sending}, receiving={receiving}, location_type={location_type}")
        #     print(f"      PDT={PDT}, GR={GR}, MCT={MCT}")
        #     print(f"      计算leadtime={leadtime}, 最终={final_leadtime}")

        return final_leadtime, ""

    except Exception as e:
        return 0, f'lead_time_calculation_error: {e}'

def calculate_daily_net_demand(
    material: str,
    location: str,
    date: pd.Timestamp,
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    future_production_df: pd.DataFrame,
    today_shipment_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    downstream_forecast_gap: float,
    downstream_safety_gap: float,
    horizon: int,
    delivery_shipment_df: pd.DataFrame | None = None,
    order_df: pd.DataFrame | None = None,       # ← 新增：用于 AO_local
    downstream_ao_gap: float = 0.0              # ← 新增：下游 AO 缺口    
) -> Tuple[float, float, float]:  # ← 返回 AO, FC, SS 三个 gap
    """计算每日净需求（forecast gap和safety gap） - 兼容module1的数据格式
    
    Args:
        material: 物料编码
        location: 地点编码
        date: 计算日期
        supply_demand_df: 供需数据 (来自Module1 SupplyDemandLog)
        safety_stock_df: 安全库存数据
        beginning_inventory_df: 每日期初库存数据
        in_transit_df: 在途数据
        delivery_gr_df: 收货数据
        future_production_df: 未来确认生产数据
        today_shipment_df: 今日发货数据 (来自Module1 ShipmentLog)
        open_deployment_df: 开放调拨数据
        downstream_forecast_gap: 下游预测缺口
        downstream_safety_gap: 下游安全库存缺口
        horizon: 计算周期天数（提前期天数）
        
    Returns:
        Tuple[float, float]: (forecast_gap, safety_gap)
    """
    # 参数验证
    if not isinstance(date, pd.Timestamp):
        try:
            date = pd.to_datetime(date)
        except:
            raise TypeError("date must be convertible to pandas Timestamp")
    
    if horizon <= 0:
        horizon = 1
    
    try:
        horizon_end = date + pd.Timedelta(days=horizon)
    except Exception as e:
        raise ValueError(f"Invalid date calculation: {e}")
    
    try:
        # 🚀 OPTIMIZATION: Pre-filter DataFrames by material & location ONCE to avoid repeated comparisons
        # Filter beginning_inventory_df
        bi_filtered = pd.DataFrame()
        if beginning_inventory_df is not None and (not beginning_inventory_df.empty) and 'material' in beginning_inventory_df.columns:
            bi_mask = (beginning_inventory_df['material'] == material) & (beginning_inventory_df['location'] == location)
            bi_filtered = beginning_inventory_df[bi_mask]
        
        # Filter in_transit_df
        it_filtered = pd.DataFrame()
        if not in_transit_df.empty and 'material' in in_transit_df.columns:
            it_mask = (in_transit_df['material'] == material) & (in_transit_df['receiving'] == location)
            it_filtered = in_transit_df[it_mask]
        
        # Filter delivery_gr_df
        dgr_filtered = pd.DataFrame()
        if not delivery_gr_df.empty and 'material' in delivery_gr_df.columns:
            dgr_mask = (delivery_gr_df['material'] == material) & (delivery_gr_df['receiving'] == location)
            dgr_filtered = delivery_gr_df[dgr_mask]
        
        # Filter future_production_df
        fp_filtered = pd.DataFrame()
        if not future_production_df.empty and 'material' in future_production_df.columns:
            fp_mask = (future_production_df['material'] == material) & (future_production_df['location'] == location)
            fp_filtered = future_production_df[fp_mask]
        
        # Filter today_shipment_df
        ts_filtered = pd.DataFrame()
        if not today_shipment_df.empty and 'material' in today_shipment_df.columns:
            ts_mask = (today_shipment_df['material'] == material) & (today_shipment_df['location'] == location)
            ts_filtered = today_shipment_df[ts_mask]
        
        # Filter open_deployment_df
        od_filtered = pd.DataFrame()
        if not open_deployment_df.empty and 'material' in open_deployment_df.columns:
            od_mask = (open_deployment_df['material'] == material) & (open_deployment_df['sending'] == location) & (open_deployment_df['receiving'] != location)
            od_filtered = open_deployment_df[od_mask]
        
        # 1. 当日期初库存（Beginning Inventory，未包含当日出库/发运扣减）
        begin_qty = 0.0
        if not bi_filtered.empty:
            bi_rows = bi_filtered[pd.to_datetime(bi_filtered['date']) == pd.to_datetime(date)]
            begin_qty = float(bi_rows['quantity'].sum()) if not bi_rows.empty else 0.0
        
        # 2. 在途库存
        in_transit_qty = 0.0
        if not it_filtered.empty:
            in_transit_qty = float(it_filtered['quantity'].sum())
        
        # 3. 今日收货
        delivery_gr_qty = 0.0
        if not dgr_filtered.empty:
            dgr_rows = dgr_filtered[dgr_filtered['date'] == date]
            delivery_gr_qty = float(dgr_rows['quantity'].sum()) if not dgr_rows.empty else 0.0
        
        # 4a. 当日生产收货 (available_date = today) —— 用 produced_qty
        today_production_gr_qty = 0.0
        if not fp_filtered.empty:
            today_rows = fp_filtered[fp_filtered['available_date'] == date]
            if not today_rows.empty:
                if 'produced_qty' in today_rows.columns:
                    today_production_gr_qty = float(today_rows['produced_qty'].sum())
                elif 'quantity' in today_rows.columns:
                    # 兼容老口径
                    today_production_gr_qty = float(today_rows['quantity'].sum())
                else:
                    today_production_gr_qty = 0.0

        # 4b. 未来确认生产（不限制时间窗口）—— 用 con_planned_qty
        future_production_qty = 0.0
        if not fp_filtered.empty:
            future_rows = fp_filtered[
                (fp_filtered['available_date'] > date)
            ]
            if not future_rows.empty:
                if 'con_planned_qty' in future_rows.columns:
                    future_production_qty = float(pd.to_numeric(future_rows['con_planned_qty'], errors='coerce').fillna(0).sum())
                elif 'produced_qty' in future_rows.columns:
                    # 回退：如果没有 con_planned_qty，就用 produced_qty
                    future_production_qty = float(pd.to_numeric(future_rows['produced_qty'], errors='coerce').fillna(0).sum())
                elif 'quantity' in future_rows.columns:
                    # 兼容老口径
                    future_production_qty = float(pd.to_numeric(future_rows['quantity'], errors='coerce').fillna(0).sum())
                else:
                    future_production_qty = 0.0

        
        # 5. 今日客户发货 (从可用量中扣除) - 使用Module1的ShipmentLog
        today_shipment_qty = 0.0
        if not ts_filtered.empty:
            ts_rows = ts_filtered[ts_filtered['date'] == date]
            today_shipment_qty = float(ts_rows['quantity'].sum()) if not ts_rows.empty else 0.0

        # 5b. 今日调拨/跨点发运（从可用量侧扣）- ★新增：来自 Orchestrator Delivery_Shipment
        delivery_shipment_qty = 0.0
        if delivery_shipment_df is not None and not delivery_shipment_df.empty:
            # 兼容字段：quantity / shipped_qty；地点字段：sending / location
            qty_col = 'quantity' if 'quantity' in delivery_shipment_df.columns else ('shipped_qty' if 'shipped_qty' in delivery_shipment_df.columns else None)
            send_col = 'sending' if 'sending' in delivery_shipment_df.columns else ('location' if 'location' in delivery_shipment_df.columns else None)
            date_col = 'date' if 'date' in delivery_shipment_df.columns else ('ship_date' if 'ship_date' in delivery_shipment_df.columns else None)

            if qty_col and send_col and date_col:
                # 过滤“本节点作为发送端 & 当天发运”的跨点发运
                ds_rows = delivery_shipment_df[
                    (delivery_shipment_df['material'] == material) &
                    (delivery_shipment_df[send_col] == location) &
                    (pd.to_datetime(delivery_shipment_df[date_col]) == date)
                ]
                delivery_shipment_qty = float(ds_rows[qty_col].sum()) if not ds_rows.empty else 0.0

        # 6a. 开放调拨出库 (从可用量中扣除) - 从 orchestrator 读取的当日版本视图
        # 注意：只计算真正从该地点发出的调拨，排除自循环（sending=receiving）
        open_deployment_qty = 0.0
        if not od_filtered.empty:
            # open_deployment使用deployed_qty字段而不quantity
            if 'deployed_qty' in od_filtered.columns:
                open_deployment_qty = float(od_filtered['deployed_qty'].sum())
            elif 'quantity' in od_filtered.columns:
                open_deployment_qty = float(od_filtered['quantity'].sum())
        
        # 6b. 开放调拨入库（作为未来到货的 pipeline supply）
        # 计入 receiving 端的未来可用量：date > sim_date 的入库（不设上限窗口）
        open_deployment_inbound_future_qty = 0.0
        if not open_deployment_df.empty:
            # 兼容字段：quantity / deployed_qty；地点字段：receiving；日期字段：date
            qty_col = 'deployed_qty' if 'deployed_qty' in open_deployment_df.columns else ('quantity' if 'quantity' in open_deployment_df.columns else None)
            if qty_col and 'receiving' in open_deployment_df.columns and 'date' in open_deployment_df.columns and 'material' in open_deployment_df.columns:
                odf = open_deployment_df[
                    (open_deployment_df['material'] == material) &
                    (open_deployment_df['receiving'] == location)
                ].copy()
                if not odf.empty:
                    odf['date'] = pd.to_datetime(odf['date'], errors='coerce')
                    future_mask = odf['date'] > date
                    future_inbound = pd.to_numeric(odf.loc[future_mask, qty_col], errors='coerce').fillna(0)
                    open_deployment_inbound_future_qty = float(future_inbound.sum())
        
        # 总可用量计算
        total_available = (begin_qty + in_transit_qty + delivery_gr_qty + 
                  today_production_gr_qty + future_production_qty + open_deployment_inbound_future_qty - 
                  today_shipment_qty - delivery_shipment_qty - open_deployment_qty)
        

        # ======== 需求侧：三类本地需求 ========
        # 1) AO（来自 OrderLog 且 demand_type == 'AO'，窗口 [date, horizon_end]）
        AO_local = 0.0
        if order_df is not None and not order_df.empty:
            # 首先过滤物料和地点，确保类型匹配
            material_filter = (order_df.get('material').astype(str) == str(material))
            location_filter = (order_df.get('location').astype(str) == str(location))
            demand_type_filter = (order_df.get('demand_type') == 'AO')
            
            # 确保日期列存在且为datetime类型
            if 'date' in order_df.columns:
                order_dates = pd.to_datetime(order_df['date'], errors='coerce')
                date_filter = (order_dates >= date) & (order_dates <= horizon_end)
                
                od = order_df[material_filter & location_filter & demand_type_filter & date_filter]
                
                if not od.empty and 'quantity' in od.columns:
                    AO_local = float(pd.to_numeric(od['quantity'], errors='coerce').fillna(0).sum())
                    # 添加调试信息以追踪问题
                    if AO_local > 0:
                        # print(f"    Debug: AO_local={AO_local} for {material}@{location}, horizon_end={horizon_end.date()}")
                        # Performance optimization: Use itertuples instead of iterrows
                        for ao_row in od.itertuples():
                            ao_date = pd.to_datetime(ao_row.date)
                            # print(f"      AO订单: 日期={ao_date.date()}, 数量={ao_row.quantity}")
            else:
                # 如果没有date列，AO_local保持为0
                pass

        # 2) forecast（来自 SupplyDemandLog，窗口 [date, horizon_end]）
        FC_local = 0.0
        if not supply_demand_df.empty and 'material' in supply_demand_df.columns:
            sdl_rows = supply_demand_df[
                (supply_demand_df['material'] == material) &
                (supply_demand_df['location'] == location) &
                (supply_demand_df['date'] >= date) &
                (supply_demand_df['date'] <= horizon_end)
            ]
            FC_local = float(pd.to_numeric(sdl_rows.get('quantity', 0), errors='coerce').fillna(0).sum())

        # 3) safety（取 horizon_end 当日目标）
        SS_local = 0.0
        if not safety_stock_df.empty and 'material' in safety_stock_df.columns:
            ssr = safety_stock_df[
                (safety_stock_df['material'] == material) &
                (safety_stock_df['location'] == location) &
                (safety_stock_df['date'] == horizon_end)
            ]
            if not ssr.empty and 'safety_stock_qty' in ssr.columns:
                SS_local = float(pd.to_numeric(ssr['safety_stock_qty'], errors='coerce').fillna(0).sum())

        # ======== 缺口顺序：AO → forecast → safety ========
        # 使用 total_available 作为初始 AVAILABLE，依次消耗
        AVAILABLE = float(total_available)

        # AO
        AO_total = AO_local + float(downstream_ao_gap or 0.0)
        AO_gap = max(AO_total - AVAILABLE, 0.0)
        AVAILABLE = max(AVAILABLE - min(AVAILABLE, AO_total), 0.0)

        # forecast
        FC_total = FC_local + float(downstream_forecast_gap or 0.0)
        FC_gap = max(FC_total - AVAILABLE, 0.0)
        AVAILABLE = max(AVAILABLE - min(AVAILABLE, FC_total), 0.0)

        # safety（仅计算超过 AO 和 forecast 的增量安全需求）
        SAF_total = SS_local + float(downstream_safety_gap or 0.0)
        SS_gap = max(SAF_total - AVAILABLE, 0.0)


        return AO_gap, FC_gap, SS_gap
        
    except Exception as e:
        # 记录错误但返回默认值，避免中断整个流程
        print(f"Warning: Error calculating net demand for {material}-{location} on {date}: {e}")
        return 0.0, 0.0, 0.0
    
def run_mrp_layered_simulation_daily(
    sim_date: pd.Timestamp,
    daily_supply_demand_df: pd.DataFrame,
    daily_order_df: pd.DataFrame,  
    daily_shipment_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    all_production_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    network_df: pd.DataFrame,
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: pd.DataFrame | None = None,   
    delivery_shipment_df: pd.DataFrame | None = None,
    deploy_config_df: pd.DataFrame | None = None,  # 🔧 新增：M5_DeployConfig for MOQ/RV
) -> pd.DataFrame:
    """运行单日MRP模拟 - 使用当日版本的Module1数据
    使用Global_Network中的location_type字段进行提前期计算
    支持自动识别的根节点生成netdemand
    
    Args:
        sim_date: 模拟日期
        daily_supply_demand_df: 当日供需数据 (来自Module1 SupplyDemandLog)
        daily_shipment_df: 当日发货数据 (来自Module1 ShipmentLog)
        safety_stock_df: 安全库存数据
        beginning_inventory_df: 期初库存数据
        in_transit_df: 在途数据
        delivery_gr_df: 收货数据
        all_production_df: 全量生产计划数据
        open_deployment_df: 开放调拨数据
        network_df: 网络配置数据 (包含location_type字段)
        lead_time_df: 提前期数据
        
    Returns:
        pd.DataFrame: 当日净需求记录
    """
    _t_func = time.perf_counter()
    if network_df.empty:
        print(f"Warning: Empty network configuration for date {sim_date}")
        return pd.DataFrame({'material': [], 'location': [], 'requirement_date': [], 'quantity': [], 'demand_element': [], 'layer': []})

    # standardize key columns as strings to ensure consistent joins/matching
    # 标准化标识符字段
    network_df = _normalize_identifiers(network_df)

    # filter to network entries active on the simulation date
    active_network = network_df[
        (network_df['eff_from'] <= sim_date) & (network_df['eff_to'] >= sim_date)
    ]
    if active_network.empty:
        print(f"Warning: No active network configuration for date {sim_date}")
        return pd.DataFrame({'material': [], 'location': [], 'requirement_date': [], 'quantity': [], 'demand_element': [], 'layer': []})
    # 需求池 = 未来订单（date > sim_date） + 剩余预测（SupplyDemandLog）
    # - 当日订单（date == sim_date）不进入需求池，避免与当日发货在可用量侧重复计
    def _std(df, element):
        if df is None or df.empty:
            return pd.DataFrame(columns=['date','material','location','quantity','demand_element'])
        cols = ['date','material','location','quantity']
        miss = [c for c in cols if c not in df.columns]
        if miss:
            print(f"  ⚠️ demand source '{element}' 缺少列: {miss}，将被忽略")
            return pd.DataFrame(columns=['date','material','location','quantity','demand_element'])
        out = df[cols].copy()
        out['demand_element'] = element
        return out

    # 未来订单：只取 date > sim_date
    future_orders = pd.DataFrame()
    if daily_order_df is not None and not daily_order_df.empty:
        # 仅保留未来订单（明天及以后）
        future_orders = daily_order_df[pd.to_datetime(daily_order_df['date']) > sim_date].copy()

    orders_std   = _std(future_orders, 'order')
    forecast_std = _std(daily_supply_demand_df, 'forecast')

    demand_pool_df = pd.concat([orders_std, forecast_std], ignore_index=True)
    if not demand_pool_df.empty:
        # 统一数据类型
        demand_pool_df['date'] = pd.to_datetime(demand_pool_df['date'])
        demand_pool_df['quantity'] = demand_pool_df['quantity'].astype(float)

    # 分配层级
    location_layer_df = assign_location_layers(active_network)
    if location_layer_df.empty:
        print(f"Warning: No location layers assigned for date {sim_date}")
        return pd.DataFrame({'material': [], 'location': [], 'requirement_date': [], 'quantity': [], 'demand_element': [], 'layer': []})
    location_layer = dict(zip(location_layer_df['location'], location_layer_df['layer']))
    all_layers = sorted(set(location_layer.values()), reverse=True)
    all_net_demand_records = []

    # === Caching to reduce repeated lookups ===
    # Cache for PTF/LSK to avoid repeated df filtering
    ptf_lsk_cache = _build_ptf_lsk_cache_m3(m4_mlcfg_df) if m4_mlcfg_df is not None and not m4_mlcfg_df.empty else {}

    # Helper to compute one node's net demand; used by threads
    def _compute_node_net_demand(ml_row: tuple) -> tuple:
        material = str(ml_row.material)
        location = str(ml_row.location)

        # 查找有效的网络配置
        network_candidates = active_network[
            (active_network['material'] == material) &
            (active_network['location'] == location)
        ]

        if not network_candidates.empty:
            network_row = network_candidates.iloc[0]
            upstream = network_row['sourcing']

            # 命中 network 但 sourcing 为空/空串 → 视为顶层；根节点走 Plant 口径
            if (pd.isna(upstream)) or (upstream is None) or (str(upstream).strip() == ''):
                upstream = None
                if location_layer.get(location, -1) == 0:
                    location_type = 'Plant'
                    horizon = _compute_root_horizon(
                        material=str(material),
                        location=str(location),
                        lead_time_df=lead_time_df,
                        m4_mlcfg_df=m4_mlcfg_df,
                        ptf_lsk_cache=ptf_lsk_cache
                    )
                else:
                    location_type = 'DC'
                    horizon = 1
            else:
                # 有上游：保持原逻辑
                sending_location_type = infer_sending_location_type(
                    network_df=active_network,
                    location_layer_df=location_layer_df,
                    sending=str(upstream),
                    material=str(material),
                    sim_date=sim_date
                )
                horizon, error_msg = determine_lead_time(
                    sending=str(upstream),
                    receiving=str(location),
                    location_type=str(sending_location_type),
                    lead_time_df=lead_time_df,
                    m4_mlcfg_df=m4_mlcfg_df,
                    material=str(material),
                    ptf_lsk_cache=ptf_lsk_cache
                )
                if error_msg:
                    print(f"Warning: {error_msg} for {upstream}->{location}, using default horizon=1")
                    horizon = 1
        else:
            # 根节点（如 plant）：与 Module 5 统一口径
            upstream = None
            if location_layer.get(location, -1) == 0:
                location_type = 'Plant'
                horizon = _compute_root_horizon(
                    material=str(material),
                    location=str(location),
                    lead_time_df=lead_time_df,
                    m4_mlcfg_df=m4_mlcfg_df,
                    ptf_lsk_cache=ptf_lsk_cache
                )
            else:
                location_type = 'DC'
                horizon = 1

        # 获取下游缺口（线程前置值由调用层传入后再合并）
        lower_AO_gap = downstream_gap_dict[(material, location)]['AO']
        lower_FC_gap = downstream_gap_dict[(material, location)]['FC']
        lower_SS_gap = downstream_gap_dict[(material, location)]['SS']

        # 计算当前节点的净需求
        AO_gap, FC_gap, SS_gap = calculate_daily_net_demand(
            str(material), str(location), sim_date,
            daily_supply_demand_df, safety_stock_df,
            beginning_inventory_df, in_transit_df,
            delivery_gr_df, pd.DataFrame(future_production_df),
            daily_shipment_df, open_deployment_df,
            lower_FC_gap, lower_SS_gap, horizon,
            delivery_shipment_df=delivery_shipment_df,
            order_df=daily_order_df,
            downstream_ao_gap=lower_AO_gap
        )

        # 记录当日净需求（仅有缺口时）
        records = []
        if AO_gap > 0:
            records.append({
                'material': str(material),
                'location': str(location),
                'requirement_date': sim_date + pd.Timedelta(days=1),
                'quantity': -AO_gap,
                'demand_element': 'net demand for AO',
                'layer': current_layer,
                'simulation_date': sim_date,
                'horizon_days': horizon
            })
        if FC_gap > 0:
            records.append({
                'material': str(material),
                'location': str(location),
                'requirement_date': sim_date + pd.Timedelta(days=1),
                'quantity': -FC_gap,
                'demand_element': 'net demand for forecast',
                'layer': current_layer,
                'simulation_date': sim_date,
                'horizon_days': horizon
            })
        if SS_gap > 0:
            records.append({
                'material': str(material),
                'location': str(location),
                'requirement_date': sim_date + pd.Timedelta(days=1),
                'quantity': -SS_gap,
                'demand_element': 'net demand for safety',
                'layer': current_layer,
                'simulation_date': sim_date,
                'horizon_days': horizon
            })

        # 计算向父节点传递的经过 MOQ/RV 调整后的 gap
        parent_key = None
        parent_gaps = {'AO': 0.0, 'FC': 0.0, 'SS': 0.0}
        if upstream and pd.notna(upstream):
            parent_key = (material, str(upstream))
            # 获取 MOQ/RV 配置
            moq, rv = 1, 1
            if deploy_config_df is not None and not deploy_config_df.empty:
                config_row = deploy_config_df[
                    (deploy_config_df['material'] == material) &
                    (deploy_config_df['sending'] == str(upstream))
                ]
                if not config_row.empty:
                    moq = int(config_row.iloc[0].get('moq', 1))
                    rv = int(config_row.iloc[0].get('rv', 1))

            gap_dict = {'AO': AO_gap, 'FC': FC_gap, 'SS': SS_gap}
            for de, gv in gap_dict.items():
                parent_gaps[de] = apply_moq_rv(gv, moq, rv, is_cross_node=True) if gv > 0 else 0.0

        return records, parent_key, parent_gaps

    # 🔥 关键修改：扩展material_locations，包含所有层级中的地点
    # 原来的逻辑：只包含network中明确配置的location
    # material_locations = network_df[['material', 'location']].drop_duplicates()
    
    # 新的逻辑：包含所有层级中的地点，并为缺失的material-location组合添加默认配置
    all_locations_in_layers = set(location_layer.keys())
    all_materials_in_network = set(network_df['material'].unique())
    
    # 构建完整的material-location组合
    extended_material_locations = []
    
    # 1. 添加network中明确配置的组合
    # Performance optimization: Use itertuples instead of iterrows
    for row in active_network.itertuples():
        extended_material_locations.append({
            'material': str(row.material),
            'location': str(row.location)
        })
    
    # 2. 为自动识别的根节点添加缺失的material组合
    for location in all_locations_in_layers:
        for material in all_materials_in_network:
            # 检查这个组合是否已经存在
            exists = any(
                ml['material'] == material and ml['location'] == location 
                for ml in extended_material_locations
            )
            
            if not exists:
                # 这是一个缺失的组合，需要添加
                extended_material_locations.append({
                    'material': str(material),
                    'location': str(location)
                })
    
    # 去重并转换为DataFrame，确保标识符字段保持字符串类型
    material_locations = pd.DataFrame(extended_material_locations).drop_duplicates()
    # 标准化标识符字段，确保类型一致性
    material_locations = _normalize_identifiers(material_locations)
    
    # print(f"🔍 扩展后的material-location组合:")
    # print(f"  原始network配置: {len(active_network)} 条")
    # print(f"  扩展后组合: {len(material_locations)} 条")
    # print(f"  包含的根节点: {[loc for loc in all_locations_in_layers if location_layer.get(loc, -1) == 0]}")
    
    future_production_df = all_production_df.copy() if not all_production_df.empty and 'available_date' in all_production_df.columns else pd.DataFrame()
    if not future_production_df.empty:
        future_production_df['available_date'] = pd.to_datetime(future_production_df['available_date'])
        # 数值列统一为数值类型，缺失置 0
        for col in ['produced_qty', 'uncon_planned_qty', 'quantity']:
            if col in future_production_df.columns:
                future_production_df[col] = pd.to_numeric(future_production_df[col], errors='coerce').fillna(0)
    # 下游gap分 AO、FC、SS gap
    downstream_gap_dict = defaultdict(lambda: {'AO': 0.0, 'FC': 0.0, 'SS': 0.0})

    for layer in all_layers:
        parent_gap_accum = defaultdict(lambda: {'AO': 0.0, 'FC': 0.0, 'SS': 0.0})

        # 获取当前层级的节点
        material_locations_df = pd.DataFrame(material_locations)
        layer_locations = [loc for loc, lyr in location_layer.items() if lyr == layer]
        layer_mask = material_locations_df['location'].isin(layer_locations)
        layer_nodes = material_locations_df[layer_mask]

        # 当前层号供子函数记录使用
        current_layer = layer

        # 并行处理当前层的所有节点
        records_lock = threading.Lock()
        parent_lock = threading.Lock()
        futures_map = {}
        failed_rows: List[tuple] = []
        try:
            n_workers = min(32, max(1, len(layer_nodes)))
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for ml in layer_nodes.itertuples():
                    fut = executor.submit(_compute_node_net_demand, ml)
                    futures_map[fut] = ml

                for fut in as_completed(futures_map.keys()):
                    try:
                        records, parent_key, parent_gaps = fut.result()
                        if records:
                            with records_lock:
                                all_net_demand_records.extend(records)
                        if parent_key is not None:
                            with parent_lock:
                                parent_gap_accum[parent_key]['AO'] += parent_gaps['AO']
                                parent_gap_accum[parent_key]['FC'] += parent_gaps['FC']
                                parent_gap_accum[parent_key]['SS'] += parent_gaps['SS']
                    except Exception as e:
                        print(f"[M3] parallel task failed on layer {layer}: {e}")
                        failed_rows.append(futures_map.get(fut))
        except Exception as e:
            # Executor-wide failure, fall back to sequential
            print(f"[M3] parallel execution failed for layer {layer}: {e}. Falling back to sequential.")
            failed_rows = list(layer_nodes.itertuples())

        # If any tasks failed, retry them sequentially to ensure completeness
        if failed_rows:
            print(f"[M3] retrying {len(failed_rows)} failed nodes sequentially on layer {layer}.")
            for ml in failed_rows:
                try:
                    records, parent_key, parent_gaps = _compute_node_net_demand(ml)
                    if records:
                        all_net_demand_records.extend(records)
                    if parent_key is not None:
                        parent_gap_accum[parent_key]['AO'] += parent_gaps['AO']
                        parent_gap_accum[parent_key]['FC'] += parent_gaps['FC']
                        parent_gap_accum[parent_key]['SS'] += parent_gaps['SS']
                except Exception as e:
                    print(f"[M3] sequential retry failed for node on layer {layer}: {e}")


        # 本层所有节点gap聚合后再传递给父层
        downstream_gap_dict = parent_gap_accum
        
        # if parent_gap_accum:
        #     print(f"    📊 Layer {layer} gap汇总:")
        #     for (mat, loc), gaps in parent_gap_accum.items():
        #         print(f"      {mat}@{loc}: AO={gaps['AO']:.2f}, forecast={gaps['FC']:.2f}, safety={gaps['SS']:.2f}")

    # 生成最终净需求DataFrame
    net_demand_df = pd.DataFrame(all_net_demand_records)
    
    # 立即标准化标识符字段，避免pandas自动类型推断导致问题
    if not net_demand_df.empty:
        net_demand_df = _normalize_identifiers(net_demand_df)
    
    if not net_demand_df.empty and len(net_demand_df) > 0:
        # 按关键字段分组聚合
        group_cols = ['material', 'location', 'requirement_date', 'demand_element', 'layer']
        net_demand_df = (
            net_demand_df.groupby(group_cols, as_index=False)
            .agg({
                'quantity': 'sum',
                'simulation_date': 'first',
                'horizon_days': 'first'
            })
        )
        if not net_demand_df.empty:
            # 直接返回结果，不强制排序以避免类型问题
            net_demand_df = net_demand_df.reset_index(drop=True)
    
    # 确保返回的是DataFrame类型
    final_df = pd.DataFrame(net_demand_df) if not isinstance(net_demand_df, pd.DataFrame) else net_demand_df
    
    # print(f"✅ MRP模拟完成，生成 {len(final_df)} 条netdemand记录")
    # if not final_df.empty:
    #     print(f"  涉及地点: {sorted(final_df['location'].unique())}")
    #     print(f"  涉及物料: {sorted(final_df['material'].unique())}")
    #     print(f"  层级分布: {dict(final_df['layer'].value_counts())}")
    
    print(f"[M3] run_mrp_layered_simulation_daily total: {time.perf_counter()-_t_func:.3f}s, records={len(final_df)}")
    return final_df

def load_excel_with_sheets(filepath: str) -> Dict[str, pd.DataFrame]:
    """加载Excel文件的所有sheet
    
    Args:
        filepath: Excel文件路径
        
    Returns:
        Dict[str, pd.DataFrame]: sheet名称到DataFrame的映射
    """
    xl = pd.ExcelFile(filepath)
    result = {}
    for sheet in xl.sheet_names:
        result[str(sheet)] = xl.parse(sheet)
    return result

def run_integrated_mode(
    module1_output_dir: str,
    orchestrator: object,
    config_dict: dict,
    start_date: str,
    end_date: str,
    output_dir: str
) -> dict:
    """
    Module3 集成模式运行函数
    所有模块只处理模拟周期内的数据
    
    Args:
        module1_output_dir: Module1输出目录
        orchestrator: Orchestrator实例
        config_dict: 配置数据字典
        start_date: 仿真开始日期
        end_date: 仿真结束日期
        output_dir: 输出目录
        
    Returns:
        dict: 包含输出结果的字典
    """
    print(f"🔄 Module3 运行于集成模式")
    _t_total = time.perf_counter()
    # print(f"📊 模拟模式：所有模块只处理模拟周期内的数据")
    
    # 加载静态配置数据
    safety_stock_df = config_dict.get('M3_SafetyStock', pd.DataFrame())
    network_df = config_dict.get('Global_Network', pd.DataFrame())
    lead_time_df = config_dict.get('Global_LeadTime', pd.DataFrame())
    m4_mlcfg_df = config_dict.get('M4_MaterialLocationLineCfg', pd.DataFrame())
    deploy_config_df = config_dict.get('M5_DeployConfig', pd.DataFrame())  # 🔧 新增：MOQ/RV配置
    # 数据类型转换和标识符字段标准化
    if not safety_stock_df.empty and 'date' in safety_stock_df.columns:
        safety_stock_df['date'] = pd.to_datetime(safety_stock_df['date'])
    if not network_df.empty:
        if 'eff_from' in network_df.columns:
            network_df['eff_from'] = pd.to_datetime(network_df['eff_from'])
        if 'eff_to' in network_df.columns:
            network_df['eff_to'] = pd.to_datetime(network_df['eff_to'])
    
    # 🔧 标准化所有配置数据的标识符字段
    if not deploy_config_df.empty:
        deploy_config_df = _normalize_identifiers(deploy_config_df)
    if not safety_stock_df.empty:
        safety_stock_df = _normalize_identifiers(safety_stock_df)
    if not network_df.empty:
        network_df = _normalize_identifiers(network_df)
    if not m4_mlcfg_df.empty:
        m4_mlcfg_df = _normalize_identifiers(m4_mlcfg_df)
    
    # 生成日期范围
    date_range = pd.date_range(start_date, end_date, freq='D')
    # print(f"处理 {len(date_range)} 天，从 {start_date} 到 {end_date}")
    
    all_net_demand = []
    
    for current_date in date_range:
        # print(f"\n📅 处理日期: {current_date.strftime('%Y-%m-%d')}")
        
        # 从Module1加载每日数据（只处理模拟周期内的数据）
        try:
            module1_daily_data = load_module1_daily_outputs(module1_output_dir, current_date)
            supply_demand_df = module1_daily_data.get('supply_demand_df', pd.DataFrame())
            today_shipment_df = module1_daily_data.get('shipment_df', pd.DataFrame())
            # print(f"  ✅ 从 Module1 加载了 {len(supply_demand_df)} 条供需记录")
            # print(f"  ✅ 从 Module1 加载了 {len(today_shipment_df)} 条发货记录")
        except Exception as e:
            print(f"  ⚠️  Module1数据加载失败: {e}")
            supply_demand_df = pd.DataFrame()
            today_shipment_df = pd.DataFrame()
        
        # 从 Orchestrator 获取动态数据
        try:
            beginning_inventory_df = orchestrator.get_beginning_inventory_view(current_date.strftime('%Y-%m-%d'))
            in_transit_df = orchestrator.get_planning_intransit_view(current_date.strftime('%Y-%m-%d'))
            delivery_gr_df = orchestrator.get_delivery_gr_view(current_date.strftime('%Y-%m-%d'))
            #production_gr_df = orchestrator.get_production_gr_view(current_date.strftime('%Y-%m-%d'))
            #production_gr_df = production_gr_df.rename(columns={'date': 'available_date'})
            all_production_df = orchestrator.get_all_production_view(current_date.strftime('%Y-%m-%d'))
            open_deployment_df = orchestrator.get_open_deployment_view(current_date.strftime('%Y-%m-%d'))
            delivery_shipment_df = orchestrator.get_delivery_shipment_log_view(current_date.strftime('%Y-%m-%d'))

            # 标准化从Orchestrator获取的数据中的标识符字段
            beginning_inventory_df = _normalize_identifiers(beginning_inventory_df)
            in_transit_df = _normalize_identifiers(in_transit_df)
            delivery_gr_df = _normalize_identifiers(delivery_gr_df)
            all_production_df = _normalize_identifiers(all_production_df)
            open_deployment_df = _normalize_identifiers(open_deployment_df)
            delivery_shipment_df = _normalize_identifiers(delivery_shipment_df)

            # print(f"  ✅ 从 Orchestrator 加载了 {len(beginning_inventory_df)} 条期初库存记录")
            # print(f"  ✅ 从 Orchestrator 加载了 {len(in_transit_df)} 条在途记录")
            # print(f"  ✅ 从 Orchestrator 加载了 {len(delivery_gr_df)} 条收货记录")
            # print(f"  ✅ 从 Orchestrator 加载了 {len(all_production_df)} 条生产记录")
            # print(f"  ✅ 从 Orchestrator 加载了 {len(open_deployment_df)} 条开放部署记录")
            # print(f"  ✅ 从 Orchestrator 加载了 {len(delivery_shipment_df)} 条发运记录")
        except Exception as e:
            print(f"  ⚠️  Orchestrator数据加载失败: {e}")
            beginning_inventory_df = pd.DataFrame()
            in_transit_df = pd.DataFrame()
            delivery_gr_df = pd.DataFrame()
            all_production_df = pd.DataFrame()
            open_deployment_df = pd.DataFrame()
            delivery_shipment_df = pd.DataFrame()
        
        # 计算当日的Net Demand  
        try:
            net_demand_df = run_mrp_layered_simulation_daily(
                current_date,
                supply_demand_df,
                 module1_daily_data.get('order_df', pd.DataFrame()),
                today_shipment_df,
                safety_stock_df,
                beginning_inventory_df,
                in_transit_df,
                delivery_gr_df,
                all_production_df,  # 使用从Orchestrator获取的生产数据 全量生产：含今天+未来
                open_deployment_df,
                network_df,
                lead_time_df,
                m4_mlcfg_df,
                delivery_shipment_df=delivery_shipment_df,
                deploy_config_df=deploy_config_df  # 🔧 新增：传递MOQ/RV配置
            )
            # print(f"  ✅ 计算完成，生成 {len(net_demand_df)} 条净需求记录")
        except Exception as e:
            print(f"  ❌ 净需求计算失败: {e}")
            import traceback
            traceback.print_exc()
            net_demand_df = pd.DataFrame()
        
        # 保存每日输出
        daily_output_file = f"{output_dir}/Module3Output_{current_date.strftime('%Y%m%d')}.xlsx"
        try:
            expected_cols = ['material','location','requirement_date','quantity','demand_element','layer','simulation_date','horizon_days']
            if net_demand_df.empty:
                net_demand_df = pd.DataFrame(columns=expected_cols)
            else:
                for c in expected_cols:
                    if c not in net_demand_df.columns:
                        net_demand_df[c] = pd.Series(dtype='object')
                net_demand_df = net_demand_df[expected_cols]
            with pd.ExcelWriter(daily_output_file, engine='openpyxl') as writer:
                net_demand_df.to_excel(writer, index=False, sheet_name='NetDemand')
        except Exception as e:
            print(f"  ⚠️  保存失败: {e}")
        
        all_net_demand.extend(net_demand_df.to_dict('records') if not net_demand_df.empty else [])
    
    print(f"\n✅ Module3 集成模式处理完成")
    print(f"[M3] total duration: {time.perf_counter()-_t_total:.3f}s for {len(date_range)} days, net_demand_records={len(all_net_demand)}")
    # print(f"  处理了 {len(date_range)} 天")
    # print(f"  生成了 {len(all_net_demand)} 条Net Demand记录")
    # print(f"  所有模块只处理模拟周期内的数据")
    
    return {
        'net_demand_count': len(all_net_demand),
        'processed_dates': len(date_range),
        'output_files': [f"Module3Output_{d.strftime('%Y%m%d')}.xlsx" for d in date_range]
    }

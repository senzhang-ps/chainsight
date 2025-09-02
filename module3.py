import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, List
from collections import defaultdict, deque
from datetime import datetime, timedelta

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
                elif key in ['network_config'] and 'eff_from' in loaded_config[key].columns:
                    loaded_config[key]['eff_from'] = pd.to_datetime(loaded_config[key]['eff_from'])
                    loaded_config[key]['eff_to'] = pd.to_datetime(loaded_config[key]['eff_to'])
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

        # 2) ShipmentLog：仅保留当日
        shp = _read('ShipmentLog')
        if not shp.empty and 'date' in shp.columns:
            shp['date'] = pd.to_datetime(shp['date'])
            shp = shp[shp['date'] == simulation_date].copy()

        # 3) OrderLog：当天版本全量（包含未来订单 + 当天新单）
        odl = _read('OrderLog')
        if not odl.empty:
            if 'date' in odl.columns:
                odl['date'] = pd.to_datetime(odl['date'])
            if 'simulation_date' in odl.columns:
                odl['simulation_date'] = pd.to_datetime(odl['simulation_date'])
            # 不按 simulation_date 过滤，保留当天版本全量；后续在 M3 内部再筛 date > sim_date

        module1_data['supply_demand_df'] = sdl
        module1_data['shipment_df'] = shp
        module1_data['order_df'] = odl
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
    if network_df.empty:
        return pd.DataFrame({'location': [], 'layer': []})
        
    children = defaultdict(list)
    parents = defaultdict(list)
    
    # 第一步：构建父子关系图
    for _, row in network_df.iterrows():
        sourcing_val = row['sourcing']
        location_val = row['location']
        
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
    
    print(f"🔍 自动识别网络层级:")
    print(f"  总地点数: {len(all_locations)}")
    print(f"  潜在根节点: {potential_roots}")
    print(f"  识别出的根节点: {true_roots}")
    
    # 第五步：从根节点开始分配层级
    layer_dict = {}
    queue = deque()
    
    # 根节点从layer 0开始
    for root in true_roots:
        queue.append((root, 0))
        print(f"  📍 根节点: {root} -> Layer 0")
    
    # 广度优先遍历分配层级
    while queue:
        loc, layer = queue.popleft()
        if loc in layer_dict and layer_dict[loc] <= layer:
            continue
        layer_dict[loc] = layer
        
        # 子节点层级 = 父节点层级 + 1
        for child in children.get(loc, []):
            queue.append((child, layer + 1))
            print(f"  📍 子节点: {child} -> Layer {layer + 1} (父节点: {loc})")
    
    # 第六步：处理未连接或孤立的节点
    unassigned = [loc for loc in all_locations if loc not in layer_dict]
    if unassigned:
        max_layer = max(layer_dict.values()) if layer_dict else 0
        for loc in unassigned:
            layer_dict[loc] = max_layer + 1
            print(f"  📍 孤立节点: {loc} -> Layer {max_layer + 1}")
    
    # 第七步：生成层级映射DataFrame
    layer_df = pd.DataFrame([
        {'location': loc, 'layer': layer} 
        for loc, layer in layer_dict.items()
    ])
    
    # 按层级排序
    layer_df = layer_df.sort_values('layer')
    
    print(f"  ✅ 层级分配完成，共 {len(layer_df)} 个地点")
    print(f"  层级范围: {layer_df['layer'].min()} - {layer_df['layer'].max()}")
    
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

def _get_ptf_lsk(material: str, site: str, m4_mlcfg_df: pd.DataFrame | None) -> tuple[int, int]:
    """
    从 M4_MaterialLocationLineCfg 读取 (PTF, LSK)
    - 表结构字段：material, location, ..., lsk, ptf, day, MCT
    - 兼容大小写列名（lsk/LSK, ptf/PTF）
    - 未命中时默认 PTF=0, LSK=1
    """
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

def determine_lead_time(
    sending: str,
    receiving: str,
    location_type: str,                 # 传入“发送端”的类型；Plant 逻辑用它判断
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: pd.DataFrame | None = None,
    material: str | None = None,
) -> tuple[int, str]:
    """
    提前期：
      - PDT/GR/MCT 来自 Global_LeadTime（按 sending+receiving 匹配）
      - 对于 Plant（发送端为 Plant）：lead_time = max(MCT, PDT+GR) + PTF + LSK - 1
        其中 PTF/LSK 从 M4_MaterialLocationLineCfg 取（列：ptf, lsk；兼容大小写）
      - 对于 DC：lead_time = PDT + GR
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
            ptf, lsk = _get_ptf_lsk(material=material, site=sending, m4_mlcfg_df=m4_mlcfg_df)

        if str(location_type).lower() == 'plant':
            base_lt  = max(MCT, PDT + GR)
            leadtime = base_lt + ptf + lsk - 1
        else:
            leadtime = PDT + GR

        return max(0, int(leadtime)), ""

    except Exception as e:
        return 0, f'lead_time_calculation_error: {e}'

def calculate_daily_net_demand(
    material: str,
    location: str,
    date: pd.Timestamp,
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    unrestricted_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    future_production_df: pd.DataFrame,
    today_shipment_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    downstream_forecast_gap: float,
    downstream_safety_gap: float,
    horizon: int,
    delivery_shipment_df: pd.DataFrame | None = None
) -> Tuple[float, float]:
    """计算每日净需求（forecast gap和safety gap） - 兼容module1的数据格式
    
    Args:
        material: 物料编码
        location: 地点编码
        date: 计算日期
        supply_demand_df: 供需数据 (来自Module1 SupplyDemandLog)
        safety_stock_df: 安全库存数据
        unrestricted_inventory_df: 无限制库存数据
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
        # 1. 当前无限制库存
        unrestricted_qty = 0.0
        if not unrestricted_inventory_df.empty and 'material' in unrestricted_inventory_df.columns:
            inv_row = unrestricted_inventory_df[
                (unrestricted_inventory_df['material'] == material) &
                (unrestricted_inventory_df['location'] == location) &
                (unrestricted_inventory_df['date'] == date)
            ]
            unrestricted_qty = float(inv_row['quantity'].sum()) if not inv_row.empty else 0.0
        
        # 2. 在途库存
        in_transit_qty = 0.0
        if not in_transit_df.empty and 'material' in in_transit_df.columns:
            in_transit_rows = in_transit_df[
                (in_transit_df['material'] == material) &
                (in_transit_df['receiving'] == location)
            ]
            in_transit_qty = float(in_transit_rows['quantity'].sum()) if not in_transit_rows.empty else 0.0
        
        # 3. 今日收货
        delivery_gr_qty = 0.0
        if not delivery_gr_df.empty and 'material' in delivery_gr_df.columns:
            delivery_gr_rows = delivery_gr_df[
                (delivery_gr_df['material'] == material) &
                (delivery_gr_df['receiving'] == location) &
                (delivery_gr_df['date'] == date)
            ]
            delivery_gr_qty = float(delivery_gr_rows['quantity'].sum()) if not delivery_gr_rows.empty else 0.0
        
        # 4a. 当日生产收货 (available_date = today)
        today_production_gr_qty = 0.0
        if not future_production_df.empty and 'material' in future_production_df.columns:
            today_production_rows = future_production_df[
                (future_production_df['material'] == material) &
                (future_production_df['location'] == location) &
                (future_production_df['available_date'] == date)
            ]
            today_production_gr_qty = float(today_production_rows['quantity'].sum()) if not today_production_rows.empty else 0.0
        
        # 4b. 未来确认生产 (available_date > simulation_date)
        future_production_qty = 0.0
        if not future_production_df.empty and 'material' in future_production_df.columns:
            future_production_rows = future_production_df[
                (future_production_df['material'] == material) &
                (future_production_df['location'] == location) &
                (future_production_df['available_date'] > date) &
                (future_production_df['available_date'] <= horizon_end)
            ]
            future_production_qty = float(future_production_rows['quantity'].sum()) if not future_production_rows.empty else 0.0
        
        # 5. 今日客户发货 (从可用量中扣除) - 使用Module1的ShipmentLog
        today_shipment_qty = 0.0
        if not today_shipment_df.empty and 'material' in today_shipment_df.columns:
            today_shipment_rows = today_shipment_df[
                (today_shipment_df['material'] == material) &
                (today_shipment_df['location'] == location) &
                (today_shipment_df['date'] == date)
            ]
            today_shipment_qty = float(today_shipment_rows['quantity'].sum()) if not today_shipment_rows.empty else 0.0

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

        # 6. 开放调拨 (从可用量中扣除) - 从 orchestrator 读取的已经是当日版本的视图
        open_deployment_qty = 0.0
        if not open_deployment_df.empty and 'material' in open_deployment_df.columns:
            open_deployment_rows = open_deployment_df[
                (open_deployment_df['material'] == material) &
                (open_deployment_df['sending'] == location)
            ]
            # open_deployment使用deployed_qty字段而不quantity
            if not open_deployment_rows.empty and 'deployed_qty' in open_deployment_rows.columns:
                open_deployment_qty = float(open_deployment_rows['deployed_qty'].sum())
            elif not open_deployment_rows.empty and 'quantity' in open_deployment_rows.columns:
                open_deployment_qty = float(open_deployment_rows['quantity'].sum())
        
        # 总可用量计算
        total_available = (unrestricted_qty + in_transit_qty + delivery_gr_qty + 
                          today_production_gr_qty + future_production_qty - 
                          today_shipment_qty - delivery_shipment_qty - open_deployment_qty)

        # 计算总预测需求 = 本节点需求 + 下游预测缺口
        # 使用Module1的SupplyDemandLog数据
        supply_demand_qty = 0.0
        if not supply_demand_df.empty and 'material' in supply_demand_df.columns:
            supply_demand_rows = supply_demand_df[
                (supply_demand_df['material'] == material) &
                (supply_demand_df['location'] == location) &
                (supply_demand_df['date'] >= date) &
                (supply_demand_df['date'] <= horizon_end)
            ]
            # 根据Module1的数据结构，使用quantity字段
            supply_demand_qty = float(supply_demand_rows['quantity'].sum()) if not supply_demand_rows.empty else 0.0
        
        total_forecast_demand = supply_demand_qty + downstream_forecast_gap
        forecast_gap = max(total_forecast_demand - total_available, 0.0)
        
        # 计算安全库存需求缺口
        safety_stock_qty = 0.0
        if not safety_stock_df.empty and 'material' in safety_stock_df.columns:
            safety_row = safety_stock_df[
                (safety_stock_df['material'] == material) &
                (safety_stock_df['location'] == location) &
                (safety_stock_df['date'] == horizon_end)
            ]
            safety_stock_qty = float(safety_row['safety_stock_qty'].sum()) if not safety_row.empty else 0.0
        
        # 总安全需求 = 预测需求 + 下游安全缺口 + 本地安全库存
        total_safety_demand = total_forecast_demand + safety_stock_qty + downstream_safety_gap
        safety_gap = max(total_safety_demand - total_available, 0.0) - forecast_gap
        
        return forecast_gap, safety_gap
        
    except Exception as e:
        # 记录错误但返回默认值，避免中断整个流程
        print(f"Warning: Error calculating net demand for {material}-{location} on {date}: {e}")
        return 0.0, 0.0
    
def run_mrp_layered_simulation_daily(
    sim_date: pd.Timestamp,
    daily_supply_demand_df: pd.DataFrame,
    daily_order_df: pd.DataFrame,  
    daily_shipment_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    unrestricted_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    all_production_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    network_df: pd.DataFrame,
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: pd.DataFrame | None = None,   
    delivery_shipment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """运行单日MRP模拟 - 使用当日版本的Module1数据
    使用Global_Network中的location_type字段进行提前期计算
    支持自动识别的根节点生成netdemand
    
    Args:
        sim_date: 模拟日期
        daily_supply_demand_df: 当日供需数据 (来自Module1 SupplyDemandLog)
        daily_shipment_df: 当日发货数据 (来自Module1 ShipmentLog)
        safety_stock_df: 安全库存数据
        unrestricted_inventory_df: 无限制库存数据
        in_transit_df: 在途数据
        delivery_gr_df: 收货数据
        all_production_df: 全量生产计划数据
        open_deployment_df: 开放调拨数据
        network_df: 网络配置数据 (包含location_type字段)
        lead_time_df: 提前期数据
        
    Returns:
        pd.DataFrame: 当日净需求记录
    """
    if network_df.empty:
        print(f"Warning: Empty network configuration for date {sim_date}")
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
    location_layer_df = assign_location_layers(network_df)
    if location_layer_df.empty:
        print(f"Warning: No location layers assigned for date {sim_date}")
        return pd.DataFrame({'material': [], 'location': [], 'requirement_date': [], 'quantity': [], 'demand_element': [], 'layer': []})
    
    location_layer = dict(zip(location_layer_df['location'], location_layer_df['layer']))
    all_layers = sorted(set(location_layer.values()), reverse=True)
    all_net_demand_records = []

    # 🔥 关键修改：扩展material_locations，包含所有层级中的地点
    # 原来的逻辑：只包含network中明确配置的location
    # material_locations = network_df[['material', 'location']].drop_duplicates()
    
    # 新的逻辑：包含所有层级中的地点，并为缺失的material-location组合添加默认配置
    all_locations_in_layers = set(location_layer.keys())
    all_materials_in_network = set(network_df['material'].unique())
    
    # 构建完整的material-location组合
    extended_material_locations = []
    
    # 1. 添加network中明确配置的组合
    for _, row in network_df.iterrows():
        extended_material_locations.append({
            'material': str(row['material']),
            'location': str(row['location'])
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
    
    # 去重并转换为DataFrame
    material_locations = pd.DataFrame(extended_material_locations).drop_duplicates()
    
    print(f"🔍 扩展后的material-location组合:")
    print(f"  原始network配置: {len(network_df)} 条")
    print(f"  扩展后组合: {len(material_locations)} 条")
    print(f"  包含的根节点: {[loc for loc in all_locations_in_layers if location_layer.get(loc, -1) == 0]}")
    
    future_production_df = all_production_df.copy() if not all_production_df.empty and 'available_date' in all_production_df.columns else pd.DataFrame()
    
    # 下游gap分 forecast_gap、safety_gap
    downstream_gap_dict = defaultdict(lambda: {'forecast_gap': 0.0, 'safety_gap': 0.0})

    for layer in all_layers:
        parent_gap_accum = defaultdict(lambda: {'forecast_gap': 0.0, 'safety_gap': 0.0})
        
        # 获取当前层级的节点
        material_locations_df = pd.DataFrame(material_locations)
        layer_mask = material_locations_df['location'].apply(lambda loc: location_layer.get(loc, -1) == layer)
        layer_nodes = material_locations_df[layer_mask]
        
        print(f"   处理Layer {layer}: {len(layer_nodes)} 个节点")
        
        for _, ml in layer_nodes.iterrows():
            material = str(ml['material'])
            location = str(ml['location'])

            # 查找有效的网络配置
            network_candidates = network_df[
                (network_df['material'] == material) &
                (network_df['location'] == location) &
                (network_df['eff_from'] <= sim_date) &
                (network_df['eff_to'] >= sim_date)
            ]

            if not network_candidates.empty:
                network_row = network_candidates.iloc[0]
                upstream = network_row['sourcing']
                
                # 处理upstream为nan或None的情况
                if pd.isna(upstream) or upstream is None:
                    upstream = None
                    location_type = 'DC'
                    horizon = 1
                else:
                    # MCT是微生物检测时间，与sending site相关
                    # 需要查找sending location的location_type
                    sending_location_type = infer_sending_location_type(
                        network_df=network_df,
                        location_layer_df=location_layer_df,
                        sending=str(upstream),
                        material=str(material),
                        sim_date=sim_date
                    )

                    horizon, error_msg = determine_lead_time(
                        sending=str(upstream),
                        receiving=str(location),
                        location_type=str(sending_location_type),   # ← 现在能正确识别 Plant
                        lead_time_df=lead_time_df,
                        m4_mlcfg_df=m4_mlcfg_df,
                        material=str(material)
                    )
                    
                    if error_msg:
                        print(f"Warning: {error_msg} for {upstream}->{location}, using default horizon=1")
                        horizon = 1
            else:
                # 🔥 新增：处理自动识别的根节点（如plant）
                # 这些节点在network中没有明确配置，但通过层级分析被识别为根节点
                upstream = None
                if location_layer.get(location, -1) == 0:
                    # 这是根节点（如plant），设置默认值
                    location_type = 'Plant'
                    horizon = 1
                    print(f"     自动识别根节点: {material}@{location} (Layer 0)")
                else:
                    # 其他未配置的节点
                    location_type = 'DC'
                    horizon = 1

            # 获取下游缺口
            lower_forecast_gap = downstream_gap_dict[(material, location)]['forecast_gap']
            lower_safety_gap = downstream_gap_dict[(material, location)]['safety_gap']

            # 计算当前节点的净需求
            forecast_gap, safety_gap = calculate_daily_net_demand(
                str(material), str(location), sim_date,
                demand_pool_df, safety_stock_df,
                unrestricted_inventory_df, in_transit_df,
                delivery_gr_df, pd.DataFrame(future_production_df),
                daily_shipment_df, open_deployment_df,
                lower_forecast_gap, lower_safety_gap, horizon,
                delivery_shipment_df=delivery_shipment_df
            )

            # gap分别加给父节点
            if upstream and pd.notna(upstream):
                parent_gap_accum[(material, upstream)]['forecast_gap'] += forecast_gap
                parent_gap_accum[(material, upstream)]['safety_gap'] += safety_gap
                print(f"    📤 传递gap到上游: {material}@{upstream} += forecast:{forecast_gap:.2f}, safety:{safety_gap:.2f}")

            # 记录当日净需求
            if forecast_gap > 0:
                all_net_demand_records.append({
                    'material': str(material),
                    'location': str(location),
                    'requirement_date': sim_date + pd.Timedelta(days=1),  # +1天，给第二天的Module4使用
                    'quantity': -forecast_gap,  # 负值表示需求
                    'demand_element': 'Distribution Demand - Forecast',
                    'layer': layer,
                    'simulation_date': sim_date,
                    'horizon_days': horizon
                })
                
            if safety_gap > 0:
                all_net_demand_records.append({
                    'material': str(material),
                    'location': str(location),
                    'requirement_date': sim_date + pd.Timedelta(days=1),  # +1天，给第二天的Module4使用
                    'quantity': -safety_gap,  # 负值表示需求
                    'demand_element': 'Distribution Demand - Safety Stock',
                    'layer': layer,
                    'simulation_date': sim_date,
                    'horizon_days': horizon
                })

        # ★关键：本层所有节点gap聚合后再传递给父层
        downstream_gap_dict = parent_gap_accum
        
        if parent_gap_accum:
            print(f"    📊 Layer {layer} gap汇总:")
            for (mat, loc), gaps in parent_gap_accum.items():
                print(f"      {mat}@{loc}: forecast={gaps['forecast_gap']:.2f}, safety={gaps['safety_gap']:.2f}")

    # 生成最终净需求DataFrame
    net_demand_df = pd.DataFrame(all_net_demand_records)
    
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
    
    print(f"✅ MRP模拟完成，生成 {len(final_df)} 条netdemand记录")
    if not final_df.empty:
        print(f"  涉及地点: {sorted(final_df['location'].unique())}")
        print(f"  涉及物料: {sorted(final_df['material'].unique())}")
        print(f"  层级分布: {dict(final_df['layer'].value_counts())}")
    
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
    print(f"📊 模拟模式：所有模块只处理模拟周期内的数据")
    
    # 加载静态配置数据
    safety_stock_df = config_dict.get('M3_SafetyStock', pd.DataFrame())
    network_df = config_dict.get('Global_Network', pd.DataFrame())
    lead_time_df = config_dict.get('Global_LeadTime', pd.DataFrame())
    m4_mlcfg_df = config_dict.get('M4_MaterialLocationLineCfg', pd.DataFrame())
    # 数据类型转换
    if not safety_stock_df.empty and 'date' in safety_stock_df.columns:
        safety_stock_df['date'] = pd.to_datetime(safety_stock_df['date'])
    if not network_df.empty:
        if 'eff_from' in network_df.columns:
            network_df['eff_from'] = pd.to_datetime(network_df['eff_from'])
        if 'eff_to' in network_df.columns:
            network_df['eff_to'] = pd.to_datetime(network_df['eff_to'])
    
    # 生成日期范围
    date_range = pd.date_range(start_date, end_date, freq='D')
    print(f"处理 {len(date_range)} 天，从 {start_date} 到 {end_date}")
    
    all_net_demand = []
    
    for current_date in date_range:
        print(f"\n📅 处理日期: {current_date.strftime('%Y-%m-%d')}")
        
        # 从Module1加载每日数据（只处理模拟周期内的数据）
        try:
            module1_daily_data = load_module1_daily_outputs(module1_output_dir, current_date)
            supply_demand_df = module1_daily_data.get('supply_demand_df', pd.DataFrame())
            today_shipment_df = module1_daily_data.get('shipment_df', pd.DataFrame())
            print(f"  ✅ 从 Module1 加载了 {len(supply_demand_df)} 条供需记录")
            print(f"  ✅ 从 Module1 加载了 {len(today_shipment_df)} 条发货记录")
        except Exception as e:
            print(f"  ⚠️  Module1数据加载失败: {e}")
            supply_demand_df = pd.DataFrame()
            today_shipment_df = pd.DataFrame()
        
        # 从 Orchestrator 获取动态数据
        try:
            unrestricted_inventory_df = orchestrator.get_unrestricted_inventory_view(current_date.strftime('%Y-%m-%d'))
            in_transit_df = orchestrator.get_planning_intransit_view(current_date.strftime('%Y-%m-%d'))
            delivery_gr_df = orchestrator.get_delivery_gr_view(current_date.strftime('%Y-%m-%d'))
            production_gr_df = orchestrator.get_production_gr_view(current_date.strftime('%Y-%m-%d'))
            production_gr_df = production_gr_df.rename(columns={'date': 'available_date'})
            open_deployment_df = orchestrator.get_open_deployment_view(current_date.strftime('%Y-%m-%d'))
            delivery_shipment_df = orchestrator.get_delivery_shipment_log_view(current_date.strftime('%Y-%m-%d'))

            print(f"  ✅ 从 Orchestrator 加载了 {len(unrestricted_inventory_df)} 条库存记录")
            print(f"  ✅ 从 Orchestrator 加载了 {len(in_transit_df)} 条在途记录")
            print(f"  ✅ 从 Orchestrator 加载了 {len(delivery_gr_df)} 条收货记录")
            print(f"  ✅ 从 Orchestrator 加载了 {len(production_gr_df)} 条生产记录")
            print(f"  ✅ 从 Orchestrator 加载了 {len(open_deployment_df)} 条开放部署记录")
            print(f"  ✅ 从 Orchestrator 加载了 {len(delivery_shipment_df)} 条发运记录")
        except Exception as e:
            print(f"  ⚠️  Orchestrator数据加载失败: {e}")
            unrestricted_inventory_df = pd.DataFrame()
            in_transit_df = pd.DataFrame()
            delivery_gr_df = pd.DataFrame()
            production_gr_df = pd.DataFrame()
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
                unrestricted_inventory_df,
                in_transit_df,
                delivery_gr_df,
                production_gr_df,  # 使用从Orchestrator获取的生产数据
                open_deployment_df,
                network_df,
                lead_time_df,
                m4_mlcfg_df,
                delivery_shipment_df=delivery_shipment_df
            )
            print(f"  ✅ 计算完成，生成 {len(net_demand_df)} 条净需求记录")
        except Exception as e:
            print(f"  ❌ 净需求计算失败: {e}")
            import traceback
            traceback.print_exc()
            net_demand_df = pd.DataFrame()
        
        # 保存每日输出
        daily_output_file = f"{output_dir}/Module3Output_{current_date.strftime('%Y%m%d')}.xlsx"
        try:
            with pd.ExcelWriter(daily_output_file, engine='openpyxl') as writer:
                net_demand_df.to_excel(writer, index=False, sheet_name='NetDemand')
            print(f"  ✅ 已保存每日输出: {daily_output_file}")
        except Exception as e:
            print(f"  ⚠️  保存失败: {e}")
        
        all_net_demand.extend(net_demand_df.to_dict('records') if not net_demand_df.empty else [])
    
    print(f"\n✅ Module3 集成模式处理完成")
    print(f"  处理了 {len(date_range)} 天")
    print(f"  生成了 {len(all_net_demand)} 条Net Demand记录")
    print(f"  所有模块只处理模拟周期内的数据")
    
    return {
        'net_demand_count': len(all_net_demand),
        'processed_dates': len(date_range),
        'output_files': [f"Module3Output_{d.strftime('%Y%m%d')}.xlsx" for d in date_range]
    }

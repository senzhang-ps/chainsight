#module 5
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from typing import Dict, List

# ========= 集成数据加载函数 (新增) =========

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
                    return shipment_df[required_cols].copy()
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
                return renamed_df[required_cols].copy()
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
        pd.DataFrame: 开放调拨数据 [material, sending, quantity]
    """
    try:
        date_str = current_date.strftime('%Y-%m-%d')
        open_deployment_view = orchestrator.get_open_deployment_view(date_str)
        
        if isinstance(open_deployment_view, pd.DataFrame) and not open_deployment_view.empty:
            # 确保包含需要的列
            required_cols = ['material', 'sending', 'quantity']
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
                return renamed_df[required_cols].copy()
            else:
                print(f"⚠️  Orchestrator open_deployment_view缺少字段: {missing_cols}")
        else:
            print(f"⚠️  Orchestrator返回空的open_deployment_view")
    except Exception as e:
        print(f"⚠️  从Orchestrator加载开放调拨数据失败: {e}")
    
    # 返回空DataFrame
    return pd.DataFrame(columns=['material', 'sending', 'quantity'])

def calculate_available_inventory(
    unrestricted_inventory: dict,
    in_transit: dict, 
    delivery_gr: dict,
    today_production_gr: dict,
    future_production: dict,
    today_shipment: dict,
    open_deployment: dict
) -> dict:
    """
    计算可用库存，与Module3逻辑完全一致
    
    Formula: available_inventory = unrestricted + in_transit + delivery_gr + 
             today_production + future_production - today_shipment - open_deployment
    
    Args:
        各个库存维度的字典，键为(material, location)，值为数量
        
    Returns:
        dict: 可用库存字典 {(material, location): quantity}
    """
    all_keys = set()
    for d in [unrestricted_inventory, in_transit, delivery_gr, today_production_gr, 
              future_production, today_shipment, open_deployment]:
        all_keys.update(d.keys())
    
    available_inventory = {}
    for key in all_keys:
        available_inventory[key] = (
            unrestricted_inventory.get(key, 0) +
            in_transit.get(key, 0) +
            delivery_gr.get(key, 0) +
            today_production_gr.get(key, 0) +
            future_production.get(key, 0) -
            today_shipment.get(key, 0) -
            open_deployment.get(key, 0)
        )
    
    return available_inventory

# ========= 1. 通用辅助 =========

def get_upstream(location, material, network_df, sim_date):
    row = get_active_network(network_df, material, location, sim_date)
    if not row.empty:
        return row.iloc[0]['sourcing']
    return None

def apply_moq_rv(qty, moq, rv):
    """补货量小于moq补moq，否则向上取整到rv的倍数"""
    if qty <= 0:
        return 0
    if qty < moq:
        return moq
    return int(np.ceil(qty / rv)) * rv

def determine_lead_time(
    sending: str,
    receiving: str,
    location_type: str,
    lead_time_df: pd.DataFrame
) -> tuple[int, str]:
    """
    确定两地之间的提前期 - 与Module3保持一致
    使用Global_Network中的location_type字段进行计算
    
    Args:
        sending: 发送地点
        receiving: 接收地点
        location_type: 地点类型（来自Global_Network）
        lead_time_df: 提前期配置数据
        
    Returns:
        tuple[int, str]: (提前期天数, 错误信息)
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
        PDT = int(row.iloc[0]['PDT']) if pd.notna(row.iloc[0]['PDT']) else 0
        GR = int(row.iloc[0]['GR']) if pd.notna(row.iloc[0]['GR']) else 0
        MCT = int(row.iloc[0]['MCT']) if pd.notna(row.iloc[0]['MCT']) else 0
        
        if str(location_type).lower() == 'plant':
            lead_time = max(MCT, PDT + GR)
        else:  # DC (Distribution Center)
            lead_time = PDT + GR
            
        return max(1, lead_time), ""
        
    except Exception as e:
        return 1, f'lead_time_calculation_error: {str(e)}'

def assign_network_layers(network_df: pd.DataFrame) -> pd.DataFrame:
    from collections import defaultdict, deque

    # 1. 原始网络数据处理
    net = network_df.copy()
    
    # 2. 构建父子关系图
    children = defaultdict(list)
    parents = defaultdict(list)
    for _, row in net.iterrows():
        if pd.notna(row['sourcing']):  # 只处理有效的sourcing关系
            children[row['sourcing']].append(row['location'])
            parents[row['location']].append(row['sourcing'])

    # 3. 收集所有涉及的节点
    all_locations = set()
    # 添加所有location节点
    all_locations.update(network_df['location'].dropna().unique())
    # 添加所有sourcing节点（可能不在location列中）
    all_locations.update(network_df['sourcing'].dropna().unique())

    # 4. 自动识别根节点（没有父节点的节点）
    roots = []
    for loc in all_locations:
        if not parents.get(loc):  # 没有父节点或父节点为空
            roots.append(loc)
    
    # 5. 如果没有明确的根节点，使用启发式方法
    if not roots:
        # 启发式：寻找在网络中作为sourcing出现但不作为location出现的节点
        sourcing_only = set(network_df['sourcing'].dropna().unique()) - set(network_df['location'].dropna().unique())
        if sourcing_only:
            roots = list(sourcing_only)
        else:
            # 最后手段：使用所有节点中最上游的节点
            all_sourcing = set(network_df['sourcing'].dropna().unique())
            all_locations_set = set(network_df['location'].dropna().unique())
            potential_roots = all_sourcing - all_locations_set
            roots = list(potential_roots) if potential_roots else list(all_locations)[:1] if all_locations else []

    # 6. 层级分配（BFS）
    layer_dict = {}
    queue = deque()
    for root in roots:
        queue.append((root, 0))
    
    while queue:
        loc, layer = queue.popleft()
        if loc in layer_dict and layer_dict[loc] <= layer:
            continue
        layer_dict[loc] = layer
        for child in children.get(loc, []):
            queue.append((child, layer + 1))
    
    # 7. 孤立点处理
    for loc in all_locations:
        if loc not in layer_dict:
            layer_dict[loc] = max(layer_dict.values()) + 1 if layer_dict else 0
    
    # 8. 反转层级（让消费者层为0，供应商层递增）
    layer_df = pd.DataFrame([{'location': loc, 'layer': layer} for loc, layer in layer_dict.items()])
    max_layer = layer_df['layer'].max()
    layer_df['layer'] = max_layer - layer_df['layer']
    
    return layer_df

def get_active_network(network_df, material, location, sim_date):
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


def allocate_by_priority_and_weight(demand_rows, available_stock, demand_priority_map):
    demand_rows_sorted = sorted(demand_rows, key=lambda d: demand_priority_map.get(d['demand_type'], 99))
    grouped = {}
    for d in demand_rows_sorted:
        p = demand_priority_map.get(d['demand_type'], 99)
        grouped.setdefault(p, []).append(d)
    stock_left = available_stock
    for priority in sorted(grouped):
        group = grouped[priority]
        total = sum(d['planned_qty'] for d in group)
        if total == 0:
            for d in group:
                d['deployed_qty_invCon'] = 0
            continue
        if stock_left >= total:
            for d in group:
                d['deployed_qty_invCon'] = d['planned_qty']
            stock_left -= total
        else:
            allocated = 0
            for d in group:
                weight = d['planned_qty'] / total
                d['deployed_qty_invCon'] = int(stock_left * weight)
                allocated += d['deployed_qty_invCon']
            stock_left -= allocated
            for d in group:
                d['deployed_qty_invCon'] = min(d['deployed_qty_invCon'], d['planned_qty'])
    return stock_left

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
    
    # 2. 从Module1加载当日数据
    config['SupplyDemandLog'] = config_dict.get('M5_SupplyDemandLog', pd.DataFrame())  # 从测试配置加载
    
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
                        print(f"  ✅ 从 Module1 加载了 {len(m1_supply_demand)} 条 SupplyDemandLog 数据")
        except Exception as e:
            print(f"  ⚠️  无法从 Module1 加载数据: {e}")
    
    # 从Module1加载当日发货数据
    if module1_output_dir and current_date:
        config['TodayShipment'] = load_module1_daily_shipment(module1_output_dir, current_date)
    else:
        config['TodayShipment'] = pd.DataFrame()
    
    # 3. 从Module4加载生产计划
    config['ProductionPlan'] = config_dict.get('M5_ProductionPlan', pd.DataFrame())   # 从测试配置加载
    
    # 实际从Module4输出加载数据
    if module4_output_path and os.path.exists(module4_output_path):
        try:
            xl = pd.ExcelFile(module4_output_path)
            if 'ProductionPlan' in xl.sheet_names:
                m4_production = xl.parse('ProductionPlan')
                if not m4_production.empty:
                    config['ProductionPlan'] = m4_production
                    print(f"  ✅ 从 Module4 加载了 {len(m4_production)} 条生产计划数据")
        except Exception as e:
            print(f"  ⚠️  无法从 Module4 加载数据: {e}")
    
    # 4. 从Orchestrator加载动态数据
    if orchestrator and current_date:
        date_str = current_date.strftime('%Y-%m-%d')
        try:
            config['InventoryLog'] = orchestrator.get_unrestricted_inventory_view(date_str)
            config['InTransit'] = orchestrator.get_planning_intransit_view(date_str)
            config['DeliveryGR'] = load_orchestrator_delivery_gr(orchestrator, current_date)
            config['OpenDeployment'] = load_orchestrator_open_deployment(orchestrator, current_date)
            config['ReceivingSpace'] = orchestrator.get_space_quota_view(date_str)
            print(f"  ✅ 从 Orchestrator 加载了动态数据")
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
    }
    
    for sheet, fields in date_fields.items():
        if sheet in config and not config[sheet].empty:
            for f in fields:
                if f in config[sheet].columns:
                    config[sheet][f] = pd.to_datetime(config[sheet][f])
    
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
    }
    for sheet, fields in date_fields.items():
        if sheet in config and not config[sheet].empty:
            for f in fields:
                if f in config[sheet].columns:
                    config[sheet][f] = pd.to_datetime(config[sheet][f])
    config['ValidationLog'] = validation_log
    return config

def validate_config_before_run(config, validation_log):
    # 检查leadtime缺失，pushpull缺失，demand_priority缺失
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
    # 校验demand_priority
    demand_types = set(config['SupplyDemandLog']['demand_element'].unique()) if not config['SupplyDemandLog'].empty else set()
    for dt in demand_types:
        if demand_priority[demand_priority['demand_element'] == dt].empty:
            validation_log.append({'No': len(validation_log)+1,
                                   'Issue': f"DemandPriority not defined for {dt}"})
    return validation_log

def collect_node_demands(material, location, sim_date, config, up_gap_buffer):
    supply_demand_log = config['SupplyDemandLog']
    safety_stock = config['SafetyStock']
    deploy_cfg = config['DeployConfig']
    network = config['Network']
    leadtime_df = config['LeadTime']

    # 参数
    param_row = deploy_cfg[
        (deploy_cfg['material'] == material) & (deploy_cfg['sending'] == location)
    ]
    if not param_row.empty:
        moq = int(param_row.iloc[0]['moq'])
        rv = int(param_row.iloc[0]['rv'])
        lsk = param_row.iloc[0]['lsk']
        day = int(param_row.iloc[0]['day'])
    else:
        moq, rv, lsk, day = 1, 1, 1, 1

    network_row = get_active_network(network, material, location, sim_date)
    if not network_row.empty:
        upstream = network_row.iloc[0]['sourcing']
        
        # MCT是微生物检测时间，与sending site相关
        # 需要查找sending location的location_type
        if upstream:
            sending_network_row = get_active_network(network, material, upstream, sim_date)
            if not sending_network_row.empty:
                sending_location_type = sending_network_row.iloc[0].get('location_type', 'DC')
            else:
                sending_location_type = 'DC'
        else:
            sending_location_type = 'DC'
    else:
        upstream = None
        sending_location_type = 'DC'

    # 使用与Module3一致的提前期计算逻辑
    if upstream and pd.notna(upstream) and str(upstream).strip():
        leadtime, error_msg = determine_lead_time(
            sending=str(upstream),
            receiving=str(location),
            location_type=str(sending_location_type),  # 使用sending location的location_type
            lead_time_df=leadtime_df
        )
        if error_msg:
            print(f"Warning: {error_msg} for {upstream}->{location}, using default leadtime=1")
            leadtime = 1
    else:
        # 顶层节点（无upstream）不需要计算提前期
        leadtime = 0

    # 使用统一的planned_deploy_date筛选逻辑: [simulation_date, simulation_date + lsk - 1]
    filter_start = sim_date
    filter_end = sim_date + pd.Timedelta(days=int(lsk) - 1)

    demand_rows = []

    # SupplyDemandLog（需求原始行）
    sdl = supply_demand_log[
        (supply_demand_log['material'] == material) & (supply_demand_log['location'] == location)
    ].copy()
    if not sdl.empty:
        # date字段代表requirement_date（需求需要的日期）
        sdl['requirement_date'] = pd.to_datetime(sdl['date'])
        # 计算planned_deploy_date并筛选
        sdl['planned_deploy_date'] = sdl['requirement_date'] - pd.Timedelta(days=leadtime)
        sdl['planned_deploy_date'] = sdl[['planned_deploy_date']].apply(lambda x: max(x['planned_deploy_date'], sim_date), axis=1)
        # 使用planned_deploy_date窗口筛选
        mask = (sdl['planned_deploy_date'] >= filter_start) & (sdl['planned_deploy_date'] <= filter_end)
        sdl = sdl[mask]
    for _, row in sdl.iterrows():
        requirement_date = row['requirement_date']
        planned_deploy_date = row['planned_deploy_date']
        
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
            'leadtime': leadtime,
            'requirement_date': requirement_date,
            'plan_deploy_date': planned_deploy_date,
        })

    # SafetyStock同理
    ss = safety_stock[
        (safety_stock['material'] == material) & (safety_stock['location'] == location)
    ].copy()
    if not ss.empty:
        # safety stock的date字段也代表requirement_date
        ss['requirement_date'] = pd.to_datetime(ss['date'])
        # 计算planned_deploy_date并筛选
        ss['planned_deploy_date'] = ss['requirement_date'] - pd.Timedelta(days=leadtime)
        ss['planned_deploy_date'] = ss[['planned_deploy_date']].apply(lambda x: max(x['planned_deploy_date'], sim_date), axis=1)
        # 使用planned_deploy_date窗口筛选
        mask = (ss['planned_deploy_date'] >= filter_start) & (ss['planned_deploy_date'] <= filter_end)
        ss = ss[mask]
    for _, row in ss.iterrows():
        requirement_date = row['requirement_date']
        planned_deploy_date = row['planned_deploy_date']
        
        demand_rows.append({
            'material': material,
            'location': location,
            'sending': upstream,
            'receiving': location,
            'demand_element': 'safety',
            'demand_qty': int(row['safety_stock_qty']),
            'planned_qty': int(row['safety_stock_qty']),
            'moq': moq,
            'rv': rv,
            'leadtime': leadtime,
            'requirement_date': requirement_date,
            'plan_deploy_date': planned_deploy_date,
        })

    # gap行，这部分需要按requirement_date重新计算planned_deploy_date并筛选
    if up_gap_buffer is not None and (material, location) in up_gap_buffer:
        for gap in up_gap_buffer[(material, location)]:
            requirement_date = gap.get('requirement_date', None)
            if requirement_date is None:
                # 如果没有requirement_date，默认使用当前日期
                requirement_date = sim_date
                planned_deploy_date = sim_date
            else:
                requirement_date = pd.to_datetime(requirement_date)
                # 基于上游节点的leadtime重新计算planned_deploy_date
                planned_deploy_date = requirement_date - pd.Timedelta(days=leadtime)
                planned_deploy_date = max(planned_deploy_date, sim_date)
            
            # 检查planned_deploy_date是否在筛选窗口内
            if planned_deploy_date >= filter_start and planned_deploy_date <= filter_end:
                demand_rows.append({
                    'material': material,
                    'location': gap.get('location', location),
                    'receiving': gap.get('receiving', gap.get('location', location)),
                    'orig_location': gap.get('orig_location', gap.get('location', location)),
                    'sending': upstream,
                    'demand_element': gap['demand_element'],
                    'demand_qty': gap['planned_qty'],
                    'planned_qty': gap['planned_qty'],
                    'moq': moq,
                    'rv': rv,
                    'leadtime': leadtime,
                    'requirement_date': requirement_date,
                    'plan_deploy_date': planned_deploy_date,
                    'from_location': gap.get('from_location', None),
                })
    return demand_rows

def push_softpush_allocation(
    deployment_plan_rows, config, dynamic_soh, sim_date
):
    """
    对push/soft-push模式节点，分配剩余库存到下游receiving, 输出push补货计划行
    修正：planned_delivery_date = date + leadtime (按LeadTime表查)
    """
    pushpull = config['PushPullModel']
    safety_stock = config['SafetyStock']
    leadtime_df = config['LeadTime']
    deploy_cfg = config['DeployConfig']
    net = config['Network']
    plan_rows_push = []
    group_keys = {(row['material'], row['sending']) for row in deployment_plan_rows}
    for mat, sending in group_keys:
        row_pp = pushpull[
            (pushpull['material'] == mat) & (pushpull['sending'] == sending)
        ]
        if row_pp.empty:
            continue
        model = row_pp.iloc[0]['model']
        if model not in ['push', 'soft push']:
            continue
        soh = dynamic_soh.get((mat, sending), 0)
        recs = net[(net['material']==mat) & (net['sourcing']==sending)]['location'].unique()
        param_row = deploy_cfg[
            (deploy_cfg['material'] == mat) & (deploy_cfg['sending'] == sending)
        ]
        if not param_row.empty:
            lsk = int(param_row.iloc[0]['lsk'])  # 确保LSK为整数
            day = int(param_row.iloc[0]['day'])
        else:
            lsk, day = 1, 1
        
        # 使用统一的筛选逻辑计算filter_end
        filter_end = sim_date + pd.Timedelta(days=lsk - 1)
        ss = safety_stock[
            (safety_stock['material'] == mat) & (safety_stock['location'].isin(recs))
        ]
        ss = ss[pd.to_datetime(ss['date']) == filter_end]
        total_ss = ss['safety_stock_qty'].sum()
        for _, row in ss.iterrows():
            loc = row['location']
            ss_val = row['safety_stock_qty']
            if model == 'push':
                if total_ss > 0:
                    qty = soh * ss_val / total_ss
                else:
                    qty = 0
            else:  # soft push
                # 计算本层site的safety
                own_ss = 0
                ss_self = safety_stock[
                    (safety_stock['material'] == mat) & (safety_stock['location'] == sending)
                ]
                if not ss_self.empty:
                    own_ss = ss_self['safety_stock_qty'].sum()
                qty_avail = max(0, soh - own_ss)
                qty = qty_avail * ss_val / total_ss if total_ss > 0 else 0
            qty = int(np.floor(qty))
            # 关键：查leadtime，使用与Module3一致的逻辑
            # MCT是微生物检测时间，与sending site相关
            # 获取sending location的location_type
            sending_network_row = net[
                (net['material'] == mat) & (net['location'] == sending)
            ]
            if not sending_network_row.empty:
                sending_location_type = sending_network_row.iloc[0].get('location_type', 'DC')
            else:
                sending_location_type = 'DC'
            
            leadtime, error_msg = determine_lead_time(
                sending=str(sending),
                receiving=str(loc),
                location_type=str(sending_location_type),  # 使用sending location的location_type
                lead_time_df=leadtime_df
            )
            if error_msg:
                print(f"Warning: push/soft push {error_msg} for {sending}->{loc}, using default leadtime=1")
                leadtime = 1
                print(f"Warning: {error_msg} for {sending}->{loc}, using default leadtime=1")
                leadtime = 1
            planned_delivery_date = sim_date + timedelta(days=leadtime)
            plan = {
                'date': sim_date,
                'material': mat,
                'sending': sending,
                'receiving': loc,
                'demand_qty': 0,
                'demand_element': 'push replenishment' if model=='push' else 'soft push replenishment',
                'planned_qty': qty,
                'deployed_qty_invCon_push': qty,
                'planned_delivery_date': planned_delivery_date,
            }
            plan['deployed_qty_invCon'] = plan['deployed_qty_invCon_push']  # 兼容后续空间分配和库存统计
            plan_rows_push.append(plan)
    return plan_rows_push


def apply_receiving_space_quota(deployment_plan_rows, receiving_space, sim_date, demand_priority_map):
    """
    在所有调运计划明细生成后，按receiving space quota再分配，更新deployed_qty，unfulfilled log
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
        if grp['deployed_qty_invCon'].sum() <= quota:
            df.loc[grp.index, 'deployed_qty'] = grp['deployed_qty_invCon']
            df.loc[grp.index, 'quota'] = quota
            continue
        # 空间不足，按优先级+权重分配
        rows = grp.to_dict(orient='records')
        total = sum(r['deployed_qty_invCon'] for r in rows)
        # 按优先级
        rows_sorted = sorted(rows, key=lambda r: demand_priority_map.get(r['demand_element'], 99))
        grouped = {}
        for r in rows_sorted:
            p = demand_priority_map.get(r['demand_element'], 99)
            grouped.setdefault(p, []).append(r)
        left = quota
        deploy_qtys = {i: 0 for i in range(len(rows))}
        for priority in sorted(grouped):
            group = grouped[priority]
            group_total = sum(r['deployed_qty_invCon'] for r in group)
            if left >= group_total:
                for r in group:
                    idx = rows.index(r)
                    deploy_qtys[idx] = r['deployed_qty_invCon']
                left -= group_total
            else:
                allocated = 0
                for r in group:
                    idx = rows.index(r)
                    weight = r['deployed_qty_invCon'] / group_total if group_total > 0 else 0
                    q = int(left * weight)
                    deploy_qtys[idx] = min(q, r['deployed_qty_invCon'])
                    allocated += deploy_qtys[idx]
                left -= allocated
                # 不再分配
        # 更新实际分配
        for idx, qty in deploy_qtys.items():
            i = grp.index[idx]
            df.at[i, 'deployed_qty'] = qty
            df.at[i, 'quota'] = quota
            gap = rows[idx]['deployed_qty_invCon'] - qty
            if gap > 0:
                unfulfilled.append({
                    'date': date,
                    'sending': rows[idx]['sending'],
                    'receiving': rows[idx]['receiving'],
                    'demand_qty': rows[idx]['demand_qty'],
                    'demand_element': rows[idx]['demand_element'],
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
                df.to_excel(writer, sheet_name=sheet, index=False)

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
        print("\n🔄 Module5 运行于集成模式")
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
        print("\n📜 Module5 运行于独立模式") 
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

    network_layers = assign_network_layers(network)
    location_to_layer = dict(zip(network_layers['location'], network_layers['layer']))
    layer_list = sorted(network_layers['layer'].unique())

    demand_priority_map = {row['demand_element']: row['priority'] for _, row in demand_priority.iterrows()}

    # ========== 初始化库存 soh_dict ==========

    # 1. 全收集所有material/location
    all_mats = set(config['SupplyDemandLog']['material'].unique()) | \
            set(config['SafetyStock']['material'].unique())
    all_locs = set(config['SupplyDemandLog']['location'].unique()) | \
            set(config['SafetyStock']['location'].unique())

    # 2. 确定仿真开始日期并获取当天的库存
    # 集成模式下使用第一个仿真日期，独立模式下使用sim_start参数
    actual_sim_start = sim_dates[0] if isinstance(sim_dates, list) and len(sim_dates) > 0 else sim_start
    
    inv_df = inventory_log[inventory_log['date'] == actual_sim_start]
    if inv_df.empty:
        print(f"[WARN] No inventory records found for sim_start: {actual_sim_start}")

    # 3. 检查是否有重复记录
    duplicates = inv_df.duplicated(subset=['material', 'location'], keep=False)
    if duplicates.any():
        dup_rows = inv_df[duplicates]
        raise ValueError(f"InventoryLog contains duplicate (material, location) on sim_start {sim_start}:\n{dup_rows[['material', 'location', 'date']]}")

    # 4. 初始化soh_dict，默认0
    soh_dict = {}
    for mat in all_mats:
        for loc in all_locs:
            soh_dict[(mat, loc)] = 0  # 默认0

    for _, row in inv_df.iterrows():
        soh_dict[(row['material'], row['location'])] = int(row['quantity'])


    deployment_plan_rows = []
    unfulfilled_rows = []
    stock_on_hand_log = []

    up_gap_buffer = {}

    for sim_date in sim_dates:
        # === 优化的日志输出 ===
        print(f"\n{'='*60}")
        print(f"📅 仿真日期: {sim_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        # ===== 库存计算逻辑重构 (与Module3保持一致) =====
        # 目标公式: available_inventory = 
        #   unrestricted_inventory +        # 从orchestrator获取当日无限制库存
        #   in_transit +                   # 从orchestrator获取当日在途库存
        #   delivery_gr +                  # 从orchestrator获取当日收货数据
        #   today_production +             # 从Module4获取当日生产 (available_date = today)
        #   future_production +            # 从Module4获取未来生产 (available_date > today)  
        #   - today_shipment -             # 从Module1获取当日发货数据
        #   - open_deployment              # 从orchestrator获取开放调拨数据
        
        start_soh_dict = soh_dict.copy()
        
        # 从 Module4 获取当日和未来生产
        today_production_gr = {}
        future_production = {}
        if not production_plan.empty:
            # 当日生产 (available_date = today)
            today_prod = production_plan[production_plan['available_date'] == sim_date]
            for _, row in today_prod.iterrows():
                k = (row['material'], row['location'])
                today_production_gr[k] = today_production_gr.get(k, 0) + int(row.get('produced_qty', row.get('planned_qty', 0)))
            
            # 未来生产 (available_date > today)
            future_prod = production_plan[production_plan['available_date'] > sim_date]
            for _, row in future_prod.iterrows():
                k = (row['material'], row['location'])
                future_production[k] = future_production.get(k, 0) + int(row.get('produced_qty', row.get('planned_qty', 0)))
        
        # 从 Orchestrator 获取在途库存
        today_intransit = {}
        if not in_transit.empty:
            for _, row in in_transit[in_transit['available_date'] == sim_date].iterrows():
                k = (row['material'], row['receiving'])
                today_intransit[k] = today_intransit.get(k, 0) + int(row['quantity'])
        
        # 加载当日收货、发货和开放调拨数据
        delivery_gr_data = config.get('DeliveryGR', pd.DataFrame())
        today_shipment_data = config.get('TodayShipment', pd.DataFrame())
        open_deployment_data = config.get('OpenDeployment', pd.DataFrame())
        
        # 转换为字典格式
        delivery_gr = {}
        if not delivery_gr_data.empty:
            filtered_delivery = delivery_gr_data[pd.to_datetime(delivery_gr_data['date']) == sim_date] if 'date' in delivery_gr_data.columns else delivery_gr_data
            for _, row in filtered_delivery.iterrows():
                k = (row['material'], row['receiving'])
                delivery_gr[k] = delivery_gr.get(k, 0) + int(row['quantity'])
        
        today_shipment = {}
        if not today_shipment_data.empty:
            filtered_shipment = today_shipment_data[pd.to_datetime(today_shipment_data['date']) == sim_date] if 'date' in today_shipment_data.columns else today_shipment_data
            for _, row in filtered_shipment.iterrows():
                k = (row['material'], row['location'])
                today_shipment[k] = today_shipment.get(k, 0) + int(row['quantity'])
        
        open_deployment = {}
        if not open_deployment_data.empty:
            for _, row in open_deployment_data.iterrows():
                k = (row['material'], row['sending'])
                open_deployment[k] = open_deployment.get(k, 0) + int(row['quantity'])
        
        # 使用统一的多维度库存计算公式
        unrestricted_inventory = start_soh_dict  # 基础库存
        
        dynamic_soh = calculate_available_inventory(
            unrestricted_inventory=unrestricted_inventory,
            in_transit=today_intransit, 
            delivery_gr=delivery_gr,
            today_production_gr=today_production_gr,
            future_production=future_production,
            today_shipment=today_shipment,
            open_deployment=open_deployment
        )
        up_gap_next = {}

        for layer in layer_list:
            print(f"\n📦 处理层级 {layer}")
            print(f"{'-'*40}")
            
            # 组合所有material-location对
            base_pairs = set(
                (mat, loc)
                for loc, l in location_to_layer.items() if l == layer
                for mat in config['SupplyDemandLog']['material'].unique()
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
                print(f"📍 节点: {mat}@{loc} [当前库存: {current_stock}]")
                
                demand_rows = collect_node_demands(mat, loc, sim_date, config, up_gap_buffer)
                if not demand_rows:
                    print(f"   ⚠️  无需求需要处理")
                    continue
                
                demand_types = [d['demand_element'] for d in demand_rows]
                print(f"   📋 需求类型: {', '.join(demand_types)}")
                
                # 应用MOQ/RV规则
                for d in demand_rows:
                    d['planned_qty'] = apply_moq_rv(d['demand_qty'], d['moq'], d['rv'])

                # 按优先级分组处理
                demand_rows_sorted = sorted(demand_rows, key=lambda d: demand_priority_map.get(d['demand_element'], 99))
                grouped = {}
                for d in demand_rows_sorted:
                    p = demand_priority_map.get(d['demand_element'], 99)
                    grouped.setdefault(p, []).append(d)
                
                total_demand = sum(d['planned_qty'] for d in demand_rows)
                print(f"   📊 总需求: {total_demand}, 可用库存: {current_stock}")
                
                for priority in sorted(grouped):
                    group = grouped[priority]
                    group_demand = sum(d['planned_qty'] for d in group)
                    print(f"   🔢 优先级 {priority}: 需求 {group_demand}")
                    
                    # 如果没有剩余库存，所有后续优先级都分配0
                    if current_stock <= 0:
                        for d in group:
                            d['deployed_qty_invCon'] = 0
                        print(f"      ❌ 无剩余库存，跳过")
                        continue
                    
                    if group_demand == 0:
                        for d in group:
                            d['deployed_qty_invCon'] = 0
                        continue
                    
                    if current_stock >= group_demand:
                        # 库存充足，完全满足当前优先级
                        for d in group:
                            d['deployed_qty_invCon'] = d['planned_qty']
                        current_stock -= group_demand
                        print(f"      ✅ 库存充足，完全满足")
                    else:
                        # 库存不足，按权重分配所有剩余库存给当前优先级
                        # 关键修复：用完库存后，后续优先级不再分配
                        allocated = 0
                        for d in group:
                            weight = d['planned_qty'] / group_demand if group_demand > 0 else 0
                            d['deployed_qty_invCon'] = int(current_stock * weight)
                            allocated += d['deployed_qty_invCon']
                        # 确保分配不超过计划量
                        for d in group:
                            d['deployed_qty_invCon'] = min(d['deployed_qty_invCon'], d['planned_qty'])
                        
                        # 重新计算实际分配量
                        actual_allocated = sum(d['deployed_qty_invCon'] for d in group)
                        current_stock = 0  # 关键修复：库存不足时，用完所有库存，后续优先级不再分配
                        print(f"      ⚠️  库存不足，部分满足 {actual_allocated}/{group_demand}，后续优先级不再分配")
                        
                        # 为后续优先级预设0分配
                        remaining_priorities = [p for p in sorted(grouped) if p > priority]
                        for remaining_priority in remaining_priorities:
                            for d in grouped[remaining_priority]:
                                d['deployed_qty_invCon'] = 0
                        break  # 跳出优先级循环
                    
                    # 显示分配详情
                    for d in group:
                        status = "✅" if d['deployed_qty_invCon'] == d['planned_qty'] else "⚠️"
                        print(f"      {status} [{d['demand_element']}] 计划={d['planned_qty']} 分配={d['deployed_qty_invCon']} 原始位置={d.get('orig_location', loc)}")

                # 处理GAP和生成调拨计划
                gap_count = 0
                for d in demand_rows:
                    gap_qty = d['planned_qty'] - d['deployed_qty_invCon']
                    if gap_qty > 0:
                        up_loc = get_upstream(loc, mat, network, sim_date)
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
                            'receiving': d.get('from_location', d.get('receiving', loc)),
                            'demand_qty': d['demand_qty'],
                            'demand_element': d['demand_element'],
                            'unfulfilled_qty': gap_qty,
                            'reason': "supply shortage"
                        })
                        
                        print(f"      🔼 需求缺口: {gap_qty} [{d['demand_element']}] → 上游 {up_loc}")
                
                if gap_count == 0:
                    print(f"      🟢 无需求缺口")
                
                # 生成调拨计划行
                for d in demand_rows:
                    receiving = d.get('from_location', d.get('receiving', loc))
                    
                    # 自补货（sending == receiving）不应有leadtime
                    if loc == receiving:
                        planned_delivery_date = d['plan_deploy_date']  # 本地分配无需leadtime
                    else:
                        planned_delivery_date = d.get('requirement_date', d['plan_deploy_date'])  # 跨层级调拨使用requirement_date
                    
                    plan_row = {
                        'date': d['plan_deploy_date'],
                        'material': mat,
                        'sending': loc,
                        'receiving': receiving,
                        'demand_qty': d['demand_qty'],
                        'demand_element': d['demand_element'],
                        'planned_qty': d['planned_qty'],
                        'deployed_qty_invCon': d['deployed_qty_invCon'],
                        'planned_delivery_date': planned_delivery_date,
                        'orig_location': d.get('orig_location', d['location'])
                    }
                    deployment_plan_rows.append(plan_row)

            print(f"\n✅ 层级 {layer} 处理完成，向上游传递 {sum(len(v) for v in up_gap_next.values())} 个需求缺口")

            # 更新GAP缓冲区
            up_gap_buffer = up_gap_next.copy()
        
        # push/soft-push再分配
        plan_push = push_softpush_allocation(deployment_plan_rows, config, dynamic_soh, sim_date)
        if plan_push:
            for plan in plan_push:
                plan['planned_delivery_date'] = plan['date']
            deployment_plan_rows.extend(plan_push)
            print(f"\n🔄 Push/Soft-push 补货: 生成 {len(plan_push)} 条补货计划")

        # 更新库存
        deployed_dict = {}
        df = pd.DataFrame(deployment_plan_rows)
        if not df.empty:
            today_rows = df[df['date'] == sim_date]
            for _, row in today_rows.iterrows():
                k = (row['material'], row['sending'])
                qty = row['deployed_qty_invCon'] if row['sending'] != row['receiving'] else 0
                deployed_dict[k] = deployed_dict.get(k, 0) + qty

        all_keys = set(list(start_soh_dict.keys()) +
                       list(today_production_gr.keys()) +
                       list(today_intransit.keys()) +
                       list(deployed_dict.keys()))
        
        for (mat, loc) in all_keys:
            start_soh = start_soh_dict.get((mat, loc), 0)
            prod = today_production_gr.get((mat, loc), 0)
            intrans = today_intransit.get((mat, loc), 0)
            deployed = deployed_dict.get((mat, loc), 0)
            end_soh = start_soh + prod + intrans - deployed
            soh_dict[(mat, loc)] = end_soh
            stock_on_hand_log.append({
                'material': mat,
                'location': loc,
                'date': sim_date,
                'start_soh': start_soh,
                'production': prod,
                'in_transit': intrans,
                'deployed_qty': deployed,
                'stock_on_hand': end_soh
            })
        
        print(f"\n📊 当日统计:")
        print(f"   总调拨计划数: {len(deployment_plan_rows)}")
        print(f"   未满足需求数: {len([r for r in unfulfilled_rows if r['date'] == sim_date])}")

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
    
    print(f"\n{'='*60}")
    print(f"🎉 仿真完成! 所有层级已处理完毕")
    print(f"💾 调拨计划已保存至: {output_path}")
    print(f"📈 总调拨计划数: {len(deployment_plan_rows_df)}")
    print(f"📝 未满足需求数: {len(unfulfilled_all)}")
    print(f"{'='*60}")
    
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

# 辅助函数（如assign_network_layers、collect_node_demands等）请按你当前最新版粘贴在同一个文件
# get_upstream需要sim_date参数
def get_upstream(location, material, network_df, sim_date):
    row = get_active_network(network_df, material, location, sim_date)
    if not row.empty:
        return row.iloc[0]['sourcing']
    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Module 5: Multi-echelon Deployment Planning')
    parser.add_argument('--input', required=True, help='Input config excel path')
    parser.add_argument('--output', required=True, help='Output excel path')
    parser.add_argument('--sim_start', required=True, help='Simulation start date, YYYY-MM-DD')
    parser.add_argument('--sim_end', required=True, help='Simulation end date, YYYY-MM-DD')
    args = parser.parse_args()
    main(args.input, args.output, args.sim_start, args.sim_end)

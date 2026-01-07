# =============================================================
# Module 5 多层级部署规划（说明与配置指引）
#
# 用途：在给定网络、需求、库存与产运数据下，按优先级与约束生成跨节点调拨计划。
# 运行模式：
# - 独立模式：通过 `--input/--output/--sim_start/--sim_end` 读写Excel。
# - 集成模式：通过 `config_dict + orchestrator + current_date` 直接读取各模块输出。
#
# 关键配置表（列名以实际表为准）：
# - `DeployConfig`：按 (material, sending) 维护 `moq`(最小订货量) 与 `rv`(重订量)、`lsk`/`day`（回顾节奏元数据）。
#     影响：跨节点调运的需求量会按路径分组应用 MOQ/RV；自循环（sending=receiving）不应用MOQ/RV。
# - `PushPullModel`：按 (material, sending) 维护 `model` ∈ {push, soft push, pull}。
#     影响：当日非push需求完全满足且存在剩余库存时，push/soft-push会将剩余库存按下游安全库存权重下推；soft-push会先保留本节点当日安全库存。
# - `LeadTime`：按 (sending, receiving) 维护 `PDT/GR/MCT`。
#     影响：决定窗口与到货日；Plant口径为 `max(MCT, PDT+GR)+PTF+LSK-1`，DC为 `PDT+GR`。
# - `M4_MaterialLocationLineCfg`：按 (material, location) 维护 `PTF/LSK`。
#     影响：仅Plant计算口径需要；与Module3保持一致（以 sending 为 site）。
# - `SafetyStock`：按 (material, location, date) 维护安全库存目标量。
#     影响：在窗口末日(horizon_end)作为需求；push/soft-push按目标日的安全库存做权重分配。
# - `DemandPriority`：维护 `demand_element -> priority`。
#     影响：分配顺序（数值越小优先级越高）；若缺失会自动补齐 AO=1、normal=2、其他=9。
# - `ReceivingSpace`：按 (receiving, date) 维护 `max_qty`（收货空间上限）。
#     影响：对跨节点调运的当日到货按优先级+权重做二次限额；自我满足不受限额约束。
# - `Network`：按 (material, location) 维护 `sourcing`（上游）。
#     影响：用于窗口、路径与向上游传递缺口；不允许同一 (material, location) 存在多个 `sourcing`。
# - `OrderLog`（集成模式自动从Module1日输出提取）：
#     影响：AO/normal订单在 (sim_date, horizon_end] 作为需求参与分配。
#
# 使用建议：
# - 保证 `Network/LeadTime/DeployConfig/PushPullModel/SafetyStock/ReceivingSpace` 一致且完备，避免缺口传递异常。
# - 若开启 `push/soft-push`，请确保下游安全库存与lead time配置正确，否则可能导致分配偏差。
# - 收货空间限额仅影响跨节点到货；如空间不足，将在 `UnfulfilledLog` 记录原因为 `space constraint`。
# =============================================================
#module 5
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from typing import Dict, List
from functools import lru_cache

# ========= 集成数据加载函数 (新增) =========

def _normalize_location(location_str) -> str:
    """
    规范化地点编码：补齐为4位数字字符串。
    作用：统一 `location/sending/receiving/sourcing` 字段格式，避免匹配失败。
    """
    # Handle None and pandas NA
    if location_str is None or pd.isna(location_str):
        return ""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)

def _normalize_material(material_str) -> str:
    """
    规范化物料编码为字符串。
    作用：统一 `material` 字段格式，避免数值/字符串混用导致的合并分组问题。
    """
    # Handle None and pandas NA
    if material_str is None or pd.isna(material_str):
        return ""
    return str(material_str)

def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    标识字段统一为字符串（并格式化地点字段）。
    作用：确保配置与日志中的 `material/location/sending/receiving/sourcing` 可一致匹配。
    """
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
    加载 Module1 当日发货数据（ShipmentLog）。
    - 输入：目录与 `current_date`，自动读取 `module1_output_YYYYMMDD.xlsx`。
    - 输出：`[date, material, location, quantity]`。
    影响：用于当日可用库存与预测库存的扣减（对客发货）。
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
    加载 Module1 当日订单池（OrderLog）。
    - 选择：`requirement_date >= current_date`，不按 `simulation_date` 过滤。
    - 输出列：`date, material, location, demand_type(AO/normal), quantity, simulation_date`。
    影响：AO/normal订单参与当天窗口的需求分配与缺口传递。
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
    从 Orchestrator 加载当日收货（GR）视图。
    - 列名映射：`location→receiving`, `gr_qty/received_qty→quantity`。
    - 输出列：`date, material, receiving, quantity`。
    影响：用于当日可用库存与预测库存的增加（收货入库）。
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
                print(f"⚠️Orchestrator delivery_gr_view缺少字段: {missing_cols}", flush=True)
        else:
            print("⚠️Orchestrator返回空的delivery_gr_view", flush=True)
    except Exception as e:
        print(f"⚠️从Orchestrator加载收货数据失败: {e}", flush=True)
    
    # 返回空DataFrame
    return pd.DataFrame(columns=['date', 'material', 'receiving', 'quantity'])

def load_orchestrator_open_deployment(orchestrator: object, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    从 Orchestrator 加载开放调拨（Open Deployment）视图。
    - 列名映射：`location→sending`, `deployed_qty/planned_qty→quantity`。
    - 输出列：`material, sending, receiving, quantity`。
    影响：作为当日可用库存扣减（发送端）与 inbound 管道供给（接收端）。
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
                print(f"⚠️Orchestrator open_deployment_view缺少字段: {missing_cols}", flush=True)
        else:
            print("⚠️Orchestrator返回空的open_deployment_view", flush=True)
    except Exception as e:
        print(f"⚠️从Orchestrator加载开放调拨数据失败: {e}", flush=True)
    
    # 返回空DataFrame
    return pd.DataFrame(columns=['material', 'sending', 'receiving', 'quantity'])

def build_open_deployment_inbound(open_deployment_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """
    构造开放调拨的接收端视图（pipeline inbound）。
    - 过滤：`sending != receiving` 且数量>0。
    - 维度与汇总：`(material, receiving) -> sum(quantity)`。
    影响：自补货的管道供给覆盖缺口时使用（不计入当日现货）。
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
    计算预测库存（用于缺口判断与规划）：
    `beginning + in_transit + delivery_gr + today_production + future_production - today_shipment - open_deployment`。
    影响：决定是否存在可用于满足需求的总供给能力（含未来/在途）。
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
    计算当日真实可用库存 `dynamic_soh`（用于实际分配）：
    `beginning + delivery_gr + today_production_gr - open_deployment`。
    说明：`open_deployment_inbound` 不计入当日现货，仅在自补的管道覆盖中使用。
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
    """
    查找 (material, location) 在 `Network` 中的上游 `sourcing`（按有效期筛选）。
    影响：用于确定窗口口径与向上游传递缺口。
    """
    row = get_active_network(network_df, material, location, sim_date, cache=active_network_cache)
    if not row.empty:
        return row.iloc[0]['sourcing']
    return None

def apply_moq_rv(qty, moq, rv, is_cross_node=True):
    """
    应用最小订货量/重订量（来自 `DeployConfig`）：
    - 若自循环（非跨节点），直接返回原需求量。
    - 若跨节点：`qty < moq → moq`，否则向上取整到 `rv` 的倍数。
    影响：影响有效需求量与分配结果。
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

def _lookup_moq_rv(deploy_cfg: pd.DataFrame, material: str, sending: str, receiving: str | None) -> tuple[int, int]:
    """
    从 DeployConfig 查找 (material, sending, receiving) 的 MOQ/RV。
    - 优先按三键 (material, sending, receiving) 精确匹配；
    - 若无 receiving 列或未命中，则回退按 (material, sending)；
    - 仍未命中则回退 (moq=1, rv=1)。
    """
    try:
        if 'receiving' in deploy_cfg.columns and receiving is not None:
            rows = deploy_cfg[
                (deploy_cfg['material'] == str(material)) &
                (deploy_cfg['sending'] == str(sending)) &
                (deploy_cfg['receiving'] == str(receiving))
            ]
            if not rows.empty:
                moq = int(pd.to_numeric(rows.iloc[0].get('moq', 1), errors='coerce') or 1)
                rv  = int(pd.to_numeric(rows.iloc[0].get('rv', 1), errors='coerce') or 1)
                return max(0, moq), max(0, rv)
        # fallback: (material, sending)
        rows2 = deploy_cfg[
            (deploy_cfg['material'] == str(material)) &
            (deploy_cfg['sending'] == str(sending))
        ]
        if not rows2.empty:
            moq = int(pd.to_numeric(rows2.iloc[0].get('moq', 1), errors='coerce') or 1)
            rv  = int(pd.to_numeric(rows2.iloc[0].get('rv', 1), errors='coerce') or 1)
            return max(0, moq), max(0, rv)
    except Exception:
        pass
    return 1, 1

def apply_grouped_moq_rv(demand_rows, location):
    """
    按调运路径分组应用 MOQ/RV（来自 `DeployConfig`）：
    - 分组维度：仅 (material, sending, receiving)
    - 仅跨节点（sending != receiving）应用 MOQ/RV，自循环不应用。
    - 组内数量回分使用“最大余数法”，保证组内合计等于目标调整量。
    """
    # 路径级分组（不按 demand_element 拆分）
    route_groups = {}
    for i, d in enumerate(demand_rows):
        receiving = d.get('from_location', d.get('receiving', location))
        is_cross_node = (location != receiving)
        route_key = (d['material'], location, receiving)
        if route_key not in route_groups:
            route_groups[route_key] = {
                'items': [],
                'total_qty': 0,
                'is_cross_node': is_cross_node,
                'moq': int(d.get('moq', 0) or 0),
                'rv': int(d.get('rv', 0) or 0)
            }
        # 组内聚合：取保守的最大 MOQ/RV
        group = route_groups[route_key]
        group['items'].append((i, d))
        group['total_qty'] += max(0, int(d.get('demand_qty', 0) or 0))
        group['moq'] = max(group['moq'], int(d.get('moq', 0) or 0))
        group['rv']  = max(group['rv'],  int(d.get('rv', 0)  or 0))

    adjusted_qtys = {}
    for route_key, group in route_groups.items():
        material, sending, receiving = route_key
        total_qty = int(group['total_qty'] or 0)
        is_cross_node = group['is_cross_node']
        moq = int(group['moq'] or 0)
        rv = int(group['rv'] or 0)

        # 组合后的总量应用 MOQ/RV
        adjusted_total = apply_moq_rv(total_qty, moq, rv, is_cross_node=is_cross_node)

        # 组内“最大余数法”保和回分
        if total_qty <= 0:
            # 若目标调整量为正且组内原量为零，最小改动：全部给第一行
            if adjusted_total > 0 and group['items']:
                first_idx, _ = group['items'][0]
                adjusted_qtys[first_idx] = int(adjusted_total)
                for item_idx, _ in group['items'][1:]:
                    adjusted_qtys[item_idx] = 0
            else:
                for item_idx, _ in group['items']:
                    adjusted_qtys[item_idx] = 0
            continue

        r = adjusted_total / float(total_qty)
        floors = []  # (idx, floor_share, remainder, original_qty, position)
        for pos, (item_idx, item) in enumerate(group['items']):
            original_qty = max(0, int(item.get('demand_qty', 0) or 0))
            exact = original_qty * r
            floor_val = int(np.floor(exact))
            remainder = float(exact - floor_val)
            floors.append((item_idx, floor_val, remainder, original_qty, pos))

        P = int(sum(x[1] for x in floors))
        R = int(max(0, adjusted_total - P))

        # 按 remainder 降序；并列用 original_qty 降序；再用原始顺序稳定
        floors.sort(key=lambda x: (-x[2], -x[3], x[4]))
        # 先写入 floor 分配
        for item_idx, floor_val, _, _, _ in floors:
            adjusted_qtys[item_idx] = int(floor_val)
        # 把剩余的 R 逐个 +1 分配给排名靠前的行
        for k in range(min(R, len(floors))):
            idx = floors[k][0]
            adjusted_qtys[idx] += 1

    return adjusted_qtys

def apply_priority_allocation_vectorized(demand_rows, adjusted_qtys, current_stock, demand_priority_map):
    """
        目的（Purpose）：
        - 在同一节点内，按需求优先级对库存进行向量化分配；高优先级需求先满足，最后一个被部分满足的优先级按比例分配。

        输入（Input）：
        - demand_rows：list[dict]，每条需求行包含至少以下字段：
            - 'demand_element'：需求类型（用于映射优先级）
            - 'demand_qty'：原始需求量（整数）
            - 其他上下文字段（如 'location' 等），不会在此函数中被修改
        - adjusted_qtys：dict[int, int]，按索引（与 demand_rows 对应）提供分组MOQ/RV调整后的需求量；若缺失则回退为原始 'demand_qty'
        - current_stock：int，该节点当前可用于分配的库存量（已扣除更高层的消耗）
        - demand_priority_map：dict[str, int]，需求类型到优先级的映射（数值越小优先级越高）；缺失映射时回退为 99

        输出（Output）：
        - 返回剩余库存（int），同时就地更新 demand_rows 中每行的 'deployed_qty_invCon' 字段：
            - 对完全满足的优先级组：'deployed_qty_invCon' = 'adjusted_qty'
            - 对部分满足的优先级组：按该组内 'adjusted_qty' 比例分配，取整（floor）且不超过各自需求量
            - 对未处理到的更低优先级：'deployed_qty_invCon' = 0

        逻辑（Logic）：
        1) 将 demand_rows 转为 DataFrame，计算每行：索引 idx、优先级 priority、调整后需求 adjusted_qty，初始化部署量为 0
        2) 若 current_stock ≤ 0，直接写回 0 并返回
        3) 按优先级升序遍历分组：
             - 若当前组总需求 ≤ current_stock：整组完全满足，扣减库存
             - 否则：按 adjusted_qty 比例切分 current_stock（向下取整），写回并终止循环
        4) 将每行的分配结果写回 demand_rows['deployed_qty_invCon']，返回剩余库存
    """
    if not demand_rows:
        return current_stock
    n = len(demand_rows)
    # Build DataFrame with indices - 不使用set_index以避免pandas版本兼容问题
    df = pd.DataFrame(demand_rows).copy()
    df['idx'] = np.arange(n)
    df['priority'] = df['demand_element'].map(lambda x: demand_priority_map.get(x, 99))
    
    # 计算 adjusted_qty - 使用更安全的方式避免KeyError
    adjusted_qty_list = []
    for i in range(n):
        if i in adjusted_qtys:
            adjusted_qty_list.append(int(adjusted_qtys[i]))
        else:
            adjusted_qty_list.append(int(df.iloc[i]['demand_qty']))
    df['adjusted_qty'] = adjusted_qty_list
    df['deployed_qty_invCon'] = 0

    # Early exit
    if current_stock <= 0:
        # leave zeros
        for i in range(n):
            demand_rows[i]['deployed_qty_invCon'] = 0
        return 0

    # Process priorities in ascending order; stop when stock depleted
    # 使用unique()获取优先级列表，避免groupby的潜在问题
    priorities = sorted(df['priority'].unique())
    for p in priorities:
        block = df[df['priority'] == p].copy()
        if block.empty:
            continue
        group_total = int(block['adjusted_qty'].sum())
        if group_total <= 0:
            continue
        idxs = block['idx'].to_numpy()
        adj = block['adjusted_qty'].to_numpy()
        if current_stock >= group_total:
            # fully satisfy
            df.loc[df['idx'].isin(idxs), 'deployed_qty_invCon'] = adj
            current_stock -= group_total
            continue
        # partial: proportional by adjusted_qty, integer floors
        weights = adj.astype(float)
        shares = (current_stock * (weights / float(group_total))) if group_total > 0 else np.zeros_like(weights)
        alloc = np.minimum(np.floor(shares).astype(np.int64), adj)
        for j, idx in enumerate(idxs):
            df.loc[df['idx'] == idx, 'deployed_qty_invCon'] = alloc[j]
        current_stock = 0
        # zero all remaining priorities implicitly
        break

    # Write back
    for i, val in df[['idx','deployed_qty_invCon']].itertuples(index=False):
        demand_rows[int(i)]['deployed_qty_invCon'] = int(val)
    df = df.reset_index(drop=True)
    return current_stock

def _build_ptf_lsk_cache(m4_mlcfg_df: pd.DataFrame | None) -> Dict[tuple[str, str], tuple[int, int]]:
    """
    PTF/LSK 缓存构建（性能优化）
    用途：为 Plant 口径的 lead time 计算提供 `PTF/LSK`，来源表 `M4_MaterialLocationLineCfg`。
    影响：若未维护对应 (material, location) 的 PTF/LSK，则默认 PTF=0、LSK=1，窗口可能缩短。
    """
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
    获取指定 (material, site=location) 的 `PTF/LSK`（支持缓存）。
    来源：`M4_MaterialLocationLineCfg`。
    影响：用于 Plant 口径的 lead time 计算，影响窗口与目标日。
    """
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
    """
    构建 `LeadTime` 基础参数缓存（PDT/GR/MCT）。
    影响：显著提升多次查询性能；内容缺失将导致窗口计算回退。
    """
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
    计算 (sending→receiving) 的到货提前期：
    - Plant：`lead_time = max(MCT, PDT+GR) + PTF + LSK - 1`（PTF/LSK 来自 `M4_MaterialLocationLineCfg`，按 (material, sending)）
    - DC：`lead_time = PDT + GR`
    影响：窗口与到货日、push/soft-push安全库存目标日均依赖此计算；缺失时回退为 1 天。
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
    识别发送端类型（与 Module3 一致）：
    1) `Network` 有活动行则使用其 `location_type`
    2) 若为根层（`layer=0`），视为 Plant
    3) 否则默认 DC
    影响：决定 lead time 计算口径（Plant/DC）。
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
    """
    根据 `Network` 的 `sourcing→location` 关系，计算每个 `location` 的层级（layer）。
    影响：主流程按层级从下游到上游推进分配与缺口传递。
    """
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
    """
    构建 `Network` 活动行缓存（按 `eff_from/eff_to` 有效期）。
    影响：显著提升查找上游速度；缺失时回退至DataFrame过滤。
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
        # Store the row as a dictionary for easy access
        cache[key] = row
    
    return cache

def get_active_network(network_df, material, location, sim_date, 
                      cache: Dict[tuple[str, str, pd.Timestamp, pd.Timestamp], pd.Series] | None = None):
    """
    获取 (material, location) 在 `sim_date` 的活动 `Network` 行（支持缓存）。
    影响：用于确定上游与路径，驱动窗口与缺口传递。
    """
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
    """
    判断是否为回顾日（daily/weekly/monthly）。
    配置：`lsk/day` 来源 `DeployConfig`；仅在需要时参考，不直接控制窗口。
    """
    if lsk == 'daily':
        return True
    if lsk == 'weekly':
        return dt.weekday() == (int(day) - 1)
    if lsk == 'monthly':
        return dt.day == int(day)
    raise ValueError(f"Unknown LSK: {lsk}")

def compute_horizon(dt, lsk, day):
    """
    计算从回顾日至下次回顾日之间的窗口结束日。
    说明：本模块窗口统一由 `determine_lead_time` 控制，此函数仅保留兼容用途。
    """
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
    current_date: pd.Timestamp,
    module1_result: dict = None  # 新增：直接从内存获取Module1输出
) -> dict:
    """
    加载集成配置数据（替代 `load_config`）：
    - 静态表：`SafetyStock/Network/LeadTime/DemandPriority/PushPullModel/DeployConfig`
    - 当日数据：优先从module1_result内存获取，否则从 Module1 日输出文件读取 `SupplyDemandLog/OrderLog/TodayShipment`
    - 动态数据：从 orchestrator 读取 `BeginningInventory/InTransit/DeliveryGR/OpenDeployment/ReceivingSpace`
    - 生产：优先读取 orchestrator 当日历史生产GR（避免重复），若无则回退至 Module4 `ProductionPlan`
    影响：决定 `Module5` 当日库存基线与需求池、在途与开放调拨，以及后续分配与缺口传递的依据。
    """
    import time
    t0 = time.perf_counter()
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
    
    # 2. 从Module1加载当日数据 - 优先使用内存数据
    config['SupplyDemandLog'] = config_dict.get('M5_SupplyDemandLog', pd.DataFrame())
    
    # 🔧 优先从内存获取Module1数据（数据库模式）
    if module1_result is not None:
        # 直接从内存获取Module1输出
        orders_df = module1_result.get('orders_df', pd.DataFrame())
        supply_demand_df = module1_result.get('supply_demand_df', pd.DataFrame())
        shipment_df = module1_result.get('shipment_df', pd.DataFrame())
        
        if not orders_df.empty:
            config['OrderLog'] = _normalize_identifiers(orders_df)
        else:
            config['OrderLog'] = pd.DataFrame()
            
        if not supply_demand_df.empty:
            config['SupplyDemandLog'] = _normalize_identifiers(supply_demand_df)
            
        if not shipment_df.empty:
            config['TodayShipment'] = _normalize_identifiers(shipment_df)
        else:
            config['TodayShipment'] = pd.DataFrame()
    elif module1_output_dir and current_date:
        try:
            from concurrent.futures import ThreadPoolExecutor

            date_str = current_date.strftime('%Y%m%d')
            module1_file = f"{module1_output_dir}/module1_output_{date_str}.xlsx"

            def _load_orderlog():
                try:
                    return load_module1_daily_orders(module1_output_dir, current_date)
                except Exception as e:
                    print(f"  ⚠️  加载OrderLog失败: {e}")
                    return pd.DataFrame()

            def _load_supplydemand():
                try:
                    if os.path.exists(module1_file):
                        xl = pd.ExcelFile(module1_file)
                        if 'SupplyDemandLog' in xl.sheet_names:
                            df = xl.parse('SupplyDemandLog')
                            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
                    return pd.DataFrame()
                except Exception as e:
                    print(f"  ⚠️  加载SupplyDemandLog失败: {e}")
                    return pd.DataFrame()

            def _load_todayshipment():
                try:
                    return load_module1_daily_shipment(module1_output_dir, current_date)
                except Exception as e:
                    print(f"  ⚠️  加载TodayShipment失败: {e}")
                    return pd.DataFrame()

            with ThreadPoolExecutor(max_workers=3) as ex:
                f_order = ex.submit(_load_orderlog)
                f_sdl   = ex.submit(_load_supplydemand)
                f_ship  = ex.submit(_load_todayshipment)
                order_df = f_order.result()
                if not isinstance(order_df, pd.DataFrame):
                    print("  ⚠️  并行加载 OrderLog 返回非DataFrame，回退为空表")
                    config['OrderLog'] = pd.DataFrame()
                else:
                    config['OrderLog'] = order_df
                sdl_df = f_sdl.result()
                if isinstance(sdl_df, pd.DataFrame):
                    if not sdl_df.empty:
                        config['SupplyDemandLog'] = sdl_df
                    else:
                        print("  ⚠️  并行加载 SupplyDemandLog 结果为空，保持原配置字典中的值")
                else:
                    print("  ⚠️  并行加载 SupplyDemandLog 返回非DataFrame，保持原配置字典中的值")
                ship_df = f_ship.result()
                if not isinstance(ship_df, pd.DataFrame):
                    print("⚠️并行加载 TodayShipment 返回非DataFrame，回退为空表")
                    config['TodayShipment'] = pd.DataFrame()
                else:
                    config['TodayShipment'] = ship_df
        except Exception as e:
            print(f"⚠️并行加载 Module1 数据失败，回退串行: {e}")
            # 回退串行
            config['OrderLog'] = load_module1_daily_orders(module1_output_dir, current_date)
            try:
                if os.path.exists(module1_file):
                    xl = pd.ExcelFile(module1_file)
                    if 'SupplyDemandLog' in xl.sheet_names:
                        m1_supply_demand = xl.parse('SupplyDemandLog')
                        if not m1_supply_demand.empty:
                            config['SupplyDemandLog'] = m1_supply_demand
            except Exception as e2:
                print(f"⚠️无法从 Module1 加载数据: {e2}")
            config['TodayShipment'] = load_module1_daily_shipment(module1_output_dir, current_date)
    else:
        config['OrderLog'] = pd.DataFrame()
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
                print("⚠️Orchestrator当日无历史生产GR数据", flush=True)
        except Exception as e:
            print(f"⚠️从 Orchestrator 加载生产计划失败: {e}", flush=True)

    # === 回退：若 orchestrator 无数据，再尝试从 module4 文件读取 ProductionPlan ===
    if (config['ProductionPlan'].empty) and module4_output_path and os.path.exists(module4_output_path):
        try:
            xl = pd.ExcelFile(module4_output_path)
            if 'ProductionPlan' in xl.sheet_names:
                m4_production = xl.parse('ProductionPlan')
                if not m4_production.empty:
                    # Ensure 'available_date' column exists for ProductionPlan
                    if 'available_date' not in m4_production.columns:
                        if 'date' in m4_production.columns:
                            m4_production = m4_production.rename(columns={'date': 'available_date'})
                        # Do NOT map 'production_plan_date' to 'available_date' — distinct semantics
                    # Cast types
                    if 'available_date' in m4_production.columns:
                        m4_production['available_date'] = pd.to_datetime(m4_production['available_date'], errors='coerce')
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

    # 4. 并行从 Orchestrator 加载动态数据
    if orchestrator and current_date:
        date_str = current_date.strftime('%Y-%m-%d')
        try:
            from concurrent.futures import ThreadPoolExecutor

            def _get_beginning():
                try:
                    return orchestrator.get_beginning_inventory_view(date_str)
                except Exception as e:
                    print(f"  ⚠️  加载BeginningInventory失败: {e}")
                    return pd.DataFrame()

            def _get_intransit():
                try:
                    return orchestrator.get_planning_intransit_view(date_str)
                except Exception as e:
                    print(f"  ⚠️  加载InTransit失败: {e}")
                    return pd.DataFrame()

            def _get_delivery_gr():
                try:
                    return load_orchestrator_delivery_gr(orchestrator, current_date)
                except Exception as e:
                    print(f"  ⚠️  加载DeliveryGR失败: {e}")
                    return pd.DataFrame()

            def _get_open_deployment():
                try:
                    return load_orchestrator_open_deployment(orchestrator, current_date)
                except Exception as e:
                    print(f"  ⚠️  加载OpenDeployment失败: {e}")
                    return pd.DataFrame()

            def _get_space_quota():
                try:
                    return orchestrator.get_space_quota_view(date_str)
                except Exception as e:
                    print(f"  ⚠️  加载ReceivingSpace失败: {e}")
                    return pd.DataFrame()

            with ThreadPoolExecutor(max_workers=5) as ex:
                f_inv   = ex.submit(_get_beginning)
                f_it    = ex.submit(_get_intransit)
                f_gr    = ex.submit(_get_delivery_gr)
                f_open  = ex.submit(_get_open_deployment)
                f_space = ex.submit(_get_space_quota)
                inv_df = f_inv.result(); it_df = f_it.result(); gr_df = f_gr.result(); open_df = f_open.result(); space_df = f_space.result()
                if not isinstance(inv_df, pd.DataFrame):
                    print("  ⚠️  并行加载 BeginningInventory 返回非DataFrame，回退为空表")
                    config['InventoryLog'] = pd.DataFrame()
                else:
                    config['InventoryLog'] = inv_df
                if not isinstance(it_df, pd.DataFrame):
                    print("  ⚠️  并行加载 InTransit 返回非DataFrame，回退为空表")
                    config['InTransit'] = pd.DataFrame()
                else:
                    config['InTransit'] = it_df
                if not isinstance(gr_df, pd.DataFrame):
                    print("  ⚠️  并行加载 DeliveryGR 返回非DataFrame，回退为空表")
                    config['DeliveryGR'] = pd.DataFrame()
                else:
                    config['DeliveryGR'] = gr_df
                if not isinstance(open_df, pd.DataFrame):
                    print("  ⚠️  并行加载 OpenDeployment 返回非DataFrame，回退为空表")
                    config['OpenDeployment'] = pd.DataFrame()
                else:
                    config['OpenDeployment'] = open_df
                if not isinstance(space_df, pd.DataFrame):
                    print("  ⚠️  并行加载 ReceivingSpace 返回非DataFrame，回退为空表")
                    config['ReceivingSpace'] = pd.DataFrame()
                else:
                    config['ReceivingSpace'] = space_df
        except Exception as e:
            print(f"  ⚠️  并行加载 Orchestrator 数据失败，回退串行: {e}")
            try:
                config['InventoryLog'] = orchestrator.get_beginning_inventory_view(date_str)
                config['InTransit'] = orchestrator.get_planning_intransit_view(date_str)
                config['DeliveryGR'] = load_orchestrator_delivery_gr(orchestrator, current_date)
                config['OpenDeployment'] = load_orchestrator_open_deployment(orchestrator, current_date)
                config['ReceivingSpace'] = orchestrator.get_space_quota_view(date_str)
            except Exception as e2:
                print(f"  ⚠️  从 Orchestrator 加载动态数据失败: {e2}")

    # 5. 统一最终的 ProductionPlan 列规范（兜底）：确保存在 'available_date'
    if 'ProductionPlan' in config and isinstance(config['ProductionPlan'], pd.DataFrame):
        pp = config['ProductionPlan']
        if not pp.empty and 'available_date' not in pp.columns:
            # common legacy names
            if 'date' in pp.columns:
                pp = pp.rename(columns={'date': 'available_date'})
            # Do NOT map 'production_plan_date' → 'available_date' — distinct semantics
        if 'available_date' in pp.columns:
            pp['available_date'] = pd.to_datetime(pp['available_date'], errors='coerce')
        config['ProductionPlan'] = pp
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
    print(f"[M5] load_integrated_config 用时: {time.perf_counter()-t0:.3f}s")
    return config

def load_config(input_path: str):
    """
    独立模式读取Excel配置：
    - 必需工作表：`SupplyDemandLog/ProductionPlan/InventoryLog/InTransit/SafetyStock/Network/PushPullModel/ReceivingSpace/LeadTime/DemandPriority/DeployConfig`
    - 自动补充与日期类型转换，并统一标识字符串格式。
    影响：作为独立模式的数据来源与校验基础。
    """
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
    """
    运行前配置校验与自动补充：
    - `Network` 不允许同一 (material, location) 多个 `sourcing`
    - 缺失 `LeadTime/PushPullModel` 的必要行会记录校验项
    - `DemandPriority` 自动补齐 AO=1、normal=2、其他=9
    影响：保证主流程分配顺序与路径、窗口可用。
    """
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
    new_rows = []
    existing_elements = set(dp['demand_element'].unique()) if not dp.empty and 'demand_element' in dp.columns else set()

    for elem in needed:
        if elem not in existing_elements:
            default_p = 9
            if elem == 'AO':
                default_p = 1
            elif elem == 'normal':
                default_p = 2
            
            new_rows.append({'demand_element': elem, 'priority': default_p})
            validation_log.append({
                'No': len(validation_log)+1,
                'Issue': f'Auto add DemandPriority for {elem}={default_p}'
            })
            existing_elements.add(elem) # 防止重复添加

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # 修正 FutureWarning: 使用 pd.concat 代替 .loc[len(df)] 并在合并前确保类型一致
        dp = pd.concat([dp, new_df], ignore_index=True)

    # 回写
    config['DemandPriority'] = dp

    return validation_log

def collect_node_demands(material, location, sim_date, config, up_gap_buffer,
                         ptf_lsk_cache=None, lead_time_cache=None, active_network_cache=None):
    """
    收集节点在当天窗口内的需求：
    - 窗口：统一由 `determine_lead_time` 决定；无上游按 Plant 公式。
    - 需求来源：`SupplyDemandLog`（forecast/others）、`SafetyStock`（仅 horizon_end 当天）、`OrderLog`（AO/normal）、上游 `up_gap_buffer`（净需求）。
    - 行级 `leadtime`：跨节点使用统一窗口值，自补货为 0。
    影响：决定后续库存分配、缺口生成与向上游传递。
    """
    # Timing moved to caller-level aggregation to avoid per-ML spam
    import pandas as pd
    from datetime import timedelta

    supply_demand_log = config['SupplyDemandLog']
    safety_stock      = config['SafetyStock']
    deploy_cfg        = config['DeployConfig']
    network           = config['Network']
    leadtime_df       = config['LeadTime']

    # 读取 LSK/Day（保留原有，用于元数据）；MOQ/RV 将在逐行按 (material,sending,receiving) 获取
    param_row = deploy_cfg[(deploy_cfg['material'] == material) & (deploy_cfg['sending'] == location)]
    if not param_row.empty:
        lsk = param_row.iloc[0].get('lsk', 1)
        day = int(param_row.iloc[0].get('day', 1) or 1)
    else:
        lsk, day = 1, 1

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
    # 🚀 OPTIMIZATION: Use mask to avoid repeated comparisons
    sdl_mask = (supply_demand_log['material'] == material) & (supply_demand_log['location'] == location)
    sdl = supply_demand_log[sdl_mask].copy()
    if not sdl.empty:
        sdl['requirement_date'] = pd.to_datetime(sdl['date'])

        # 识别 forecast 行（完全匹配 'forecast'，忽略大小写）
        is_fc = sdl['demand_element'].astype(str).str.lower() == 'forecast'

        # forecast: [sim_date, horizon_end]
        sdl_fc = sdl[is_fc & (sdl['requirement_date'] >= sim_date) & (sdl['requirement_date'] <= horizon_end)]

        # 其他（含 net demand for xx 等）：[sim_date, horizon_end]
        sdl_others = sdl[~is_fc & (sdl['requirement_date'] >= sim_date) & (sdl['requirement_date'] <= horizon_end)]

        # 🚀 OPTIMIZATION: Use itertuples instead of iterrows (20-27x faster)
        for row in pd.concat([sdl_fc, sdl_others], ignore_index=True).itertuples():
            # 自补（接收端=本地）；业务规则：自补货不应用 MOQ/RV，默认 moq=1, rv=1
            row_moq, row_rv = 1, 1
            demand_rows.append({
                'material': material,
                'location': location,
                'sending': upstream,
                'receiving': location,
                'demand_element': row.demand_element,
                'demand_qty': int(row.quantity),
                'planned_qty': int(row.quantity),
                'moq': int(row_moq),
                'rv': int(row_rv),
                'leadtime': leadtime_for_row if upstream else 0,  # 顶层自补 0，跨节点=统一 horizon
                'requirement_date': row.requirement_date,
                'plan_deploy_date': sim_date,  # 计划触发日在窗口内分配环节使用，这里先放 sim_date
                'orig_location': location
            })

    # ========= 2) 安全库存：只取 horizon_end 当天 =========
    # 🚀 OPTIMIZATION: Use mask to avoid repeated comparisons
    ss_mask = (safety_stock['material'] == material) & (safety_stock['location'] == location)
    ss = safety_stock[ss_mask].copy()
    ss_qty = 0
    if not ss.empty:
        ss['date'] = pd.to_datetime(ss['date'])
        ss_end = ss[ss['date'] == horizon_end]
        if not ss_end.empty:
            ss_qty = int(pd.to_numeric(ss_end['safety_stock_qty'], errors='coerce').fillna(0).sum())

    if ss_qty > 0:
        # 自补安全库存：业务规则默认 moq=1, rv=1
        row_moq, row_rv = 1, 1
        demand_rows.append({
            'material': material,
            'location': location,
            'sending': upstream,
            'receiving': location,
            'demand_element': 'safety',
            'demand_qty': ss_qty,
            'planned_qty': ss_qty,
            'moq': int(row_moq),
            'rv': int(row_rv),
            'leadtime': leadtime_for_row if upstream else 0,
            'requirement_date': horizon_end,     # 目标日 = horizon_end
            'plan_deploy_date': sim_date,
            'orig_location': location
        })

    # ========= 3) 订单池（AO/normal）：[sim_date, horizon_end] =========
    order_df = config.get('OrderLog', pd.DataFrame())
    if not order_df.empty:
        # 🚀 OPTIMIZATION: Use mask to avoid repeated comparisons
        orders_mask = (order_df['material'] == material) & (order_df['location'] == location)
        orders = order_df[orders_mask].copy()
        if not orders.empty:
            orders['requirement_date'] = pd.to_datetime(orders['date'])
            orders['demand_element']  = orders['demand_type']

            # AO / normal 统一用 (sim_date, horizon_end]
            mask = (orders['requirement_date'] >= sim_date) & (orders['requirement_date'] <= horizon_end)
            orders = orders[mask]

            # 🚀 OPTIMIZATION: Use itertuples instead of iterrows (20-27x faster)
            for row in orders.itertuples():
                qty = int(row.quantity)
                # 自补订单：业务规则默认 moq=1, rv=1
                row_moq, row_rv = 1, 1
                demand_rows.append({
                    'material': material,
                    'location': location,
                    'sending': upstream,
                    'receiving': location,
                    'demand_element': str(row.demand_element),
                    'demand_qty': qty,
                    'planned_qty': qty,
                    'moq': int(row_moq),
                    'rv': int(row_rv),
                    'leadtime': leadtime_for_row if upstream else 0,
                    'requirement_date': row.requirement_date,
                    'plan_deploy_date': sim_date,
                    'orig_location': location
                })

    # ========= 4) GAP 传递（上游下发的净需求）：[sim_date, horizon_end] =========
    if up_gap_buffer is not None and (material, location) in up_gap_buffer:
        for gap in up_gap_buffer[(material, location)]:
            req_dt = pd.to_datetime(gap.get('requirement_date', sim_date))
            if (req_dt >= sim_date) and (req_dt <= horizon_end):
                # 对 GAP 行，接收端为本节点（self），但后续生成计划时会用 from_location 作为真正的 receiving
                # 因此 MOQ/RV 需按 (material, sending=location, receiving=from_location) 获取
                recv_for_cfg = str(gap.get('from_location', gap.get('location', location)))
                row_moq, row_rv = _lookup_moq_rv(deploy_cfg, material=str(material), sending=str(location), receiving=recv_for_cfg)
                demand_rows.append({
                    'material': material,
                    'location': gap.get('location', location),
                    'receiving': gap.get('receiving', gap.get('location', location)),
                    'orig_location': gap.get('orig_location', gap.get('location', location)),
                    'sending': upstream,
                    'demand_element': gap['demand_element'],
                    'demand_qty': int(gap['planned_qty']),
                    'planned_qty': int(gap['planned_qty']),
                    'moq': int(row_moq),
                    'rv': int(row_rv),
                    'leadtime': leadtime_for_row if upstream else 0,
                    'requirement_date': req_dt,
                    'plan_deploy_date': sim_date,
                    'from_location': gap.get('from_location', None),
                })

    return demand_rows

def push_softpush_allocation(
    deployment_plan_rows, config, dynamic_soh, sim_date,
    ptf_lsk_cache=None, lead_time_cache=None, projected_soh=None,
    node_demands_map: Dict[tuple[str, str], List] | None = None
):
    """
    目的（Purpose）：
    - 在当日所有非 push 需求已满足的前提下，使用剩余可用库存执行 push/soft-push 的补货分配。
    - 采用“挡位（bucket）+ 比例兜底”的方法，尽量使同一物料的各下游在可行挡位下对齐到安全库存倍数。

    输入（Input）：
    - deployment_plan_rows：list[dict]，当日已生成的分配计划（用于排除 push 自身并统计已分配库存）。
    - config：dict，包含静态与运行时配置（PushPullModel/SafetyStock/LeadTime/Network/DeployConfig/ReceivingSpace 等）。
    - dynamic_soh：dict[(material, location)->int]，当日真实可用库存（用于计算发送端可下推量及接收端基线的回退）。
    - sim_date：datetime，当日仿真日期。
    - ptf_lsk_cache / lead_time_cache：缓存，加速 lead time 计算。
    - projected_soh：dict[(material, location)->int]，接收端当日预测库存；若提供则作为倍数对齐的库存基线，未提供则回退使用 dynamic_soh。

    输出（Output）：
    - 返回 push/soft-push 计划行列表（list[dict]），不直接修改传入的 deployment_plan_rows；由调用方自行扩展。
    - 每条计划行包含：date/material/sending/receiving/demand_element/planned_qty/deployed_qty_invCon/planned_delivery_date/leadtime/is_cross_node 等。

    逻辑（Logic）：
    1) 统计当日非 push 的已分配库存，计算发送端真实剩余可用库存 available_soh；soft-push 先保留发送端当日安全库存。
    2) 获取下游接收端列表与其到货日（sim_date + leadtime）的安全库存。
    3) 选择最高可行挡位 L（默认 [1.2,1.5,2.0,2.5,3.0]，可由 config['M5_PushLevels'] 覆盖）：
       - 计算 need_r = max(0, L*SS_r - PI_r)，其中 PI_r 优先取 projected_soh，否则 dynamic_soh。
       - 若总需求 sum(need_r) ≤ available_soh，则该挡位可行；选取最高可行挡位，若无可行则回退最低挡位。
    4) 比例兜底分配：qty_r = floor(available_soh * need_r / sum_need)，后续由接收空间配额逻辑进行限额裁剪。
    5) 生成计划行（仅 qty_r>0），带上到货日与 lead time，demand_element 标注为 push/soft push replenishment。
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta

    pushpull   = config['PushPullModel']
    safety     = config['SafetyStock']
    lt_df      = config['LeadTime']
    deploy_cfg = config['DeployConfig']
    net        = config['Network']

    import time
    t0 = time.perf_counter()
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

        # H) 挡位 + 比例兜底分配（最小化改动；使用接收端当日目标安全库存与当前库存近似）
        # 读取挡位配置（无则默认）
        push_levels = config.get('M5_PushLevels', [1.2, 1.5, 2.0, 2.5, 3.0])
        try:
            push_levels = sorted([float(l) for l in push_levels])
        except Exception:
            push_levels = [1.2, 1.5, 2.0, 2.5, 3.0]

        # 当前接收端库存基线：优先使用 projected_soh，否则回退 dynamic_soh（最小改动）
        if projected_soh is not None:
            pi_map = {x['receiving']: float(projected_soh.get((mat, x['receiving']), 0) or 0) for x in receiving_ss_data}
        else:
            pi_map = {x['receiving']: float(dynamic_soh.get((mat, x['receiving']), 0) or 0) for x in receiving_ss_data}

        # 额外扣减：未来 leadtime 窗口内的 AO/normal 与 forecast，以及在窗口末日的 safety stock
        # 目的：使接收端基线库存考虑到到货前的承诺消耗与目标安全库存，从而更贴近现实
        commitments_map: dict[str, float] = {}
        for x in receiving_ss_data:
            rec = str(x['receiving'])
            target_date = pd.to_datetime(x['planned_delivery_date'])
            # 优先使用主流程已经收集好的节点需求，避免重复计算
            if node_demands_map is not None:
                rows = node_demands_map.get((mat, rec), [])
            else:
                try:
                    # 复用已实现的窗口需求收集逻辑，最小化改动
                    rows = collect_node_demands(
                        material=mat,
                        location=rec,
                        sim_date=sim_date,
                        config=config,
                        up_gap_buffer=None,
                        ptf_lsk_cache=ptf_lsk_cache,
                        lead_time_cache=lead_time_cache,
                        active_network_cache=None
                    )
                except Exception:
                    rows = []

            commit_qty = 0.0
            for r in rows:
                de = str(r.get('demand_element', '')).lower()
                rq = int(r.get('planned_qty', r.get('demand_qty', 0)) or 0)
                req_dt = pd.to_datetime(r.get('requirement_date', sim_date))

                # 仅计【未来】窗口内的 AO/normal/forecast，避免与 projected_soh 中的当日对客发货重复
                if de in ['ao', 'normal', 'forecast']:
                    if (req_dt > sim_date) and (req_dt <= target_date):
                        commit_qty += rq
                # 计入窗口末日的安全库存目标
                elif de == 'safety':
                    if req_dt == target_date:
                        commit_qty += rq

            commitments_map[rec] = commit_qty

        # 将承诺消耗从接收端库存基线中扣减，避免负数
        for rec, commit in commitments_map.items():
            pi_map[rec] = float(max(0.0, (pi_map.get(rec, 0.0) or 0.0) - float(commit or 0.0)))

        # 选择最高可行挡位
        feasible_level = None
        for L in push_levels:
            need_sum = 0.0
            for x in receiving_ss_data:
                ssq = float(x['ss_qty'] or 0)
                if ssq <= 0:
                    continue
                pi = pi_map.get(x['receiving'], 0.0)
                need_sum += max(0.0, L * ssq - pi)
            if need_sum <= float(available_soh) + 1e-9:
                feasible_level = L
            else:
                break
        if feasible_level is None:
            feasible_level = push_levels[0]

        # 比例兜底分配到接收端
        needs = []
        for x in receiving_ss_data:
            ssq = float(x['ss_qty'] or 0)
            if ssq <= 0:
                needs.append((x, 0.0))
                continue
            pi = pi_map.get(x['receiving'], 0.0)
            needs.append((x, max(0.0, feasible_level * ssq - pi)))
        total_need = sum(n for _, n in needs)
        allocated = []
        if total_need > 0:
            for x, need in needs:
                share = (available_soh * need / total_need) if total_need > 0 else 0.0
                q = int(np.floor(share))
                allocated.append((x, q))

        # I) 生成 push 计划行（只落 qty>0）
        for x, qty in allocated:
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

    print(f"[M5] push_softpush_allocation 用时: {time.perf_counter()-t0:.3f}s，生成行数: {len(plan_rows_push)}")
    return plan_rows_push


def apply_receiving_space_quota(deployment_plan_rows, receiving_space, sim_date, demand_priority_map):
    """
        目的（Purpose）：
        - 向量化应用接收端空间/能力配额（ReceivingSpace）到当日的跨节点调拨计划，控制每个接收地在当日的最大可接收量。

        输入（Input）：
        - deployment_plan_rows：list[dict]，已生成的当日计划行（包含 push/soft-push 与常规调拨），字段至少包含：
            - 'date'、'material'、'sending'、'receiving'、'deployed_qty_invCon'、'demand_element'、'leadtime' 等
        - receiving_space：pd.DataFrame，接收端空间设置表，常见字段：
            - 'date'（可选）、'material'（可选）、'location'/'receiving'、'quota'/'space'（当日最大可接收量）
        - sim_date：datetime，当日仿真日期，仅处理当日的计划行
        - demand_priority_map：dict[str,int]，需求类型优先级映射，用于在空间不足时分配优先顺序

        输出（Output）：
        - 返回 (df, logs)：
            - df：pd.DataFrame，为每条计划行计算并写入 'deployed_qty'（实际执行量）与 'quota'（接收端可用配额），其余字段保持不变
            - logs：list[str]，可选的限额应用记录（默认简洁；调用方可用于诊断）

        逻辑（Logic）：
        1) 仅对跨节点计划行（sending != receiving）应用接收空间限额；自补货（sending == receiving）不受此处配额影响
        2) 将当日计划行按接收端分组，计算每个接收端在 'quota'（或 'space'）范围内的可用配额
        3) 组内按需求优先级（demand_priority_map）排序；若配额充足则全量执行，否则在最后一个被部分满足的优先级内按权重比例分配（向下取整）
        4) 写回每行的 'deployed_qty'（不超过原 'deployed_qty_invCon' 且受接收端剩余配额约束），返回结果
    
        性能（Performance）：
        - 大部分操作在 DataFrame 上向量化完成，相比逐行循环通常加速 2-5 倍
    """
    import time
    t0 = time.perf_counter()
    df = pd.DataFrame(deployment_plan_rows)
    if df.empty:
        df['deployed_qty'] = []
        df['quota'] = []
        return df, []

    # Fast path: no receiving space
    if receiving_space.empty:
        df['deployed_qty'] = df['deployed_qty_invCon']
        df['quota'] = np.inf
        return df, []

    # Ensure date types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'date' in receiving_space.columns:
        receiving_space['date'] = pd.to_datetime(receiving_space['date'])

    # Precompute priority for rows (missing → 99)
    df['priority'] = df['demand_element'].map(lambda x: demand_priority_map.get(x, 99))

    # Split self-fulfillment vs cross-node
    is_cross = df['sending'] != df['receiving']
    df_self = df[~is_cross].copy()
    df_cross = df[is_cross].copy()

    # Self rows pass-through
    if not df_self.empty:
        df.loc[df_self.index, 'deployed_qty'] = df_self['deployed_qty_invCon']
        df.loc[df_self.index, 'quota'] = np.inf

    # If no cross-node rows, return
    if df_cross.empty:
        print(f"[M5] Receiving Space Quota 用时: {time.perf_counter()-t0:.3f}s，受限条目: 0")
        return df, []

    # Join quota to cross-node by (receiving,date)
    space = receiving_space[['receiving','date','max_qty']].copy()
    space = space.rename(columns={'max_qty':'quota'})
    df_cross = df_cross.merge(space, on=['receiving','date'], how='left')
    df_cross['quota'] = df_cross['quota'].fillna(np.inf)

    # Group by (receiving,date) for allocation
    unfulfilled = []
    results = []

    def _alloc_group(g: pd.DataFrame) -> pd.DataFrame:
        quota = g['quota'].iloc[0]
        total = g['deployed_qty_invCon'].sum()
        if total <= quota:
            g['deployed_qty'] = g['deployed_qty_invCon']
            g['quota'] = quota
            return g
        # Sort by priority then proportional weights within each priority
        g = g.sort_values(['priority'])
        left = float(quota)
        deployed = np.zeros(len(g), dtype=np.int64)

        # Process each priority block vectorized
        for p, block in g.groupby('priority', sort=True):
            block_total = block['deployed_qty_invCon'].sum()
            idx = block.index.to_numpy()
            if left >= block_total:
                deployed_idx_vals = block['deployed_qty_invCon'].to_numpy()
            else:
                # proportional weights; integer floor
                weights = block['deployed_qty_invCon'].to_numpy().astype(float)
                if block_total > 0:
                    shares = (left * (weights / float(block_total)))
                else:
                    shares = np.zeros_like(weights)
                deployed_idx_vals = np.minimum(np.floor(shares).astype(np.int64), block['deployed_qty_invCon'].to_numpy())
            deployed[g.index.get_indexer(idx)] = deployed_idx_vals
            left -= deployed_idx_vals.sum()
            if left <= 0:
                break

        g['deployed_qty'] = deployed
        g['quota'] = quota
        return g

    allocated = (
        df_cross.groupby(['receiving','date'], sort=False, group_keys=False)
        .apply(_alloc_group)
    )

    # Write back allocated values
    df.loc[allocated.index, 'deployed_qty'] = allocated['deployed_qty']
    df.loc[allocated.index, 'quota'] = allocated['quota']

    # Build unfulfilled log vectorized where gap > 0
    gaps = allocated[allocated['deployed_qty_invCon'] > allocated['deployed_qty']]
    if not gaps.empty:
        unfulfilled = [
            {
                'date': row.date,
                'sending': row.sending,
                'receiving': row.receiving,
                'material': row.material,
                'demand_qty': row.demand_qty,
                'demand_element': row.demand_element,
                'unfulfilled_qty': int(row.deployed_qty_invCon - row.deployed_qty),
                'reason': 'space constraint'
            }
            for row in gaps.itertuples(index=False)
        ]

    # Fill pass-through for any untouched rows
    df['deployed_qty'] = df['deployed_qty'].fillna(df['deployed_qty_invCon'])
    df['quota'] = df['quota'].fillna(np.nan)
    print(f"[M5] Receiving Space Quota 用时: {time.perf_counter()-t0:.3f}s，受限条目: {len(unfulfilled)}")
    return df, unfulfilled

def log_outputs(output_path: str, outputs: Dict[str, pd.DataFrame]):
    """
    将结果表写入Excel：`DeploymentPlan/UnfulfilledLog/StockOnHandLog/Validation`。
    说明：输出前统一标识字段格式，确保后续分析一致性。
    """
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
    current_date: str = None,
    skip_file_output: bool = False,
    module1_result: dict = None  # 新增：直接从内存获取Module1输出
):
    """
    Module 5 主入口：多层级部署规划。
    - 独立模式：使用Excel；集成模式：使用各模块/Orchestrator视图。
    - 配置影响：见文件头部说明，各表直接影响窗口、分配、push行为与空间限额。
    输出：写入Excel并返回数据帧，供集成与分析使用。
    """
    # 判断运行模式
    if config_dict is not None:
        # 集成模式 - 完成的集成数据加载
        # print("\n🔄 Module5 运行于集成模式")
        current_date_obj = pd.to_datetime(current_date) if current_date else None
        config = load_integrated_config(
            config_dict, module1_output_dir, module4_output_path, 
            orchestrator, current_date_obj,
            module1_result=module1_result  # 传递Module1内存数据
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
    print(f"✅缓存已初始化: PTF/LSK={len(ptf_lsk_cache)} | LeadTime={len(lead_time_cache)} | Network={len(active_network_cache)}")
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

    import time
    for sim_date in sim_dates:
        day_start = time.perf_counter()
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
            # Use actual_delivery_date as the availability date for in-transit arrivals
            date_col = 'actual_delivery_date' if 'actual_delivery_date' in in_transit.columns else ('available_date' if 'available_date' in in_transit.columns else None)
            if date_col is not None:
                mask_today = pd.to_datetime(in_transit[date_col]).dt.normalize() == sim_date.normalize()
                for row in in_transit[mask_today].itertuples():
                    k = (row.material, row.receiving)
                    today_intransit[k] = today_intransit.get(k, 0) + int(row.quantity)
        # 未来在途：actual_delivery_date > sim_date，用于自补货的 pipeline 覆盖
        future_intransit = {}
        if not in_transit.empty:
            date_col = 'actual_delivery_date' if 'actual_delivery_date' in in_transit.columns else ('available_date' if 'available_date' in in_transit.columns else None)
            if date_col is not None:
                mask_future = pd.to_datetime(in_transit[date_col]).dt.normalize() > sim_date.normalize()
                for row in in_transit[mask_future].itertuples():
                    k = (row.material, row.receiving)
                    future_intransit[k] = future_intransit.get(k, 0) + int(row.quantity)
        
        # 加载当日收货、发货和开放调拨数据
        delivery_gr_data = config.get('DeliveryGR', pd.DataFrame())
        today_shipment_data = config.get('TodayShipment', pd.DataFrame())
        open_deployment_data = config.get('OpenDeployment', pd.DataFrame())
        
        # 转换为字典格式
        delivery_gr = {}
        if not delivery_gr_data.empty:
            filtered_delivery = delivery_gr_data[pd.to_datetime(delivery_gr_data['date']).dt.normalize() == sim_date.normalize()] if 'date' in delivery_gr_data.columns else delivery_gr_data
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
        import time
        demand_collect_total_start = time.perf_counter()
        demand_collect_only_elapsed = 0.0
        up_gap_next = {}

        # 全局需求缓存（当天所有节点）用于 push 阶段避免重复计算
        global_node_demands_map: Dict[tuple[str, str], List] = {}

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

            # 并行收集每个节点的需求（同层之间互不依赖），随后仍按原顺序分配库存
            from concurrent.futures import ThreadPoolExecutor, as_completed
            node_demands_map: dict[tuple[str,str], list] = {}
            if all_pairs:
                try:
                    layer_collect_start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=min(32, len(all_pairs))) as ex:
                        futures = {
                            ex.submit(
                                collect_node_demands,
                                mat, loc, sim_date, config, up_gap_buffer,
                                ptf_lsk_cache, lead_time_cache, active_network_cache
                            ): (mat, loc)
                            for (mat, loc) in all_pairs
                        }
                        for fut in as_completed(futures):
                            key = futures[fut]
                            try:
                                node_demands_map[key] = fut.result()
                            except Exception as e:
                                print(f"  ⚠️  并行收集需求失败: {key} -> {e}")
                                node_demands_map[key] = []
                    demand_collect_only_elapsed += (time.perf_counter() - layer_collect_start)
                except Exception as e:
                    print(f"  ⚠️  并行收集需求初始化失败，回退串行: {e}")
                    node_demands_map = {}
            
            # 合并到全局缓存，避免 push 阶段重复收集
            for k, v in node_demands_map.items():
                global_node_demands_map[k] = v

            for mat, loc in all_pairs:
                node_key = (mat, loc)
                current_stock = dynamic_soh.get(node_key, 0)
                # print(f"📍 节点: {mat}@{loc} [可用库存: {current_stock}]")
                
                demand_rows = node_demands_map.get((mat, loc))
                if demand_rows is None:
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
                
                # 👉 Vectorized priority allocation
                current_stock = apply_priority_allocation_vectorized(
                    demand_rows=demand_rows,
                    adjusted_qtys=adjusted_qtys,
                    current_stock=current_stock,
                    demand_priority_map=demand_priority_map
                )
                
                # —— 在处理 GAP 之前，给所有需求行初始化 pipeline 相关字段 —— 
                for d in demand_rows:
                    d.setdefault('deploy_qty_with_plan_order', 0)
                    d.setdefault('deploy_from_in_transit', 0)
                    d.setdefault('deploy_from_open_deployment_inbound', 0)
                    d.setdefault('deploy_from_future_production', 0)

                # —— 自补货第二轮：用 pipeline supply 覆盖剩余 gap（所有节点都适用）——
                # Vectorized pipeline allocation for self-rows
                if demand_rows:
                    ndr_df = pd.DataFrame(demand_rows).copy()
                    ndr_df['idx'] = np.arange(len(ndr_df))
                    # receiving resolution
                    rec_arr = [r.get('from_location', r.get('receiving', loc)) for r in demand_rows]
                    ndr_df['receiving'] = rec_arr
                    ndr_df['is_self'] = ndr_df['receiving'] == loc
                    ndr_df['priority'] = ndr_df['demand_element'].map(lambda x: demand_priority_map.get(x, 99))
                    # adjusted qty from map - 使用更安全的方式
                    adj_qty_list = []
                    for i in range(len(demand_rows)):
                        if i in adjusted_qtys:
                            adj_qty_list.append(int(adjusted_qtys[i]))
                        else:
                            adj_qty_list.append(int(ndr_df.iloc[i]['demand_qty']))
                    ndr_df['adjusted_qty'] = adj_qty_list
                    # already allocated from inventory - 确保列存在
                    ndr_df['allocated_invcon'] = [int(r.get('deployed_qty_invCon', 0) or 0) for r in demand_rows]
                    # existing plan cover - 确保列存在
                    ndr_df['plan_order_cover'] = [int(r.get('deploy_qty_with_plan_order', 0) or 0) for r in demand_rows]
                    # raw gap for self rows
                    self_df = ndr_df[ndr_df['is_self']].copy()
                    if not self_df.empty:
                        # 确保必要的列都存在
                        required_cols = ['adjusted_qty', 'allocated_invcon', 'plan_order_cover']
                        for col in required_cols:
                            if col not in self_df.columns:
                                self_df[col] = 0
                        self_df['raw_gap'] = self_df['adjusted_qty'] - self_df['allocated_invcon'] - self_df['plan_order_cover']
                        self_df['alloc_intrans'] = 0
                        self_df['alloc_odi'] = 0
                        self_df['alloc_future'] = 0
                        # Pools
                        node_key = (mat, loc)
                        pool_in_transit = int(future_intransit.get(node_key, 0) or 0)
                        pool_odi = int(open_deployment_inbound.get(node_key, 0) or 0)
                        pool_future_production = int(future_production.get(node_key, 0) or 0)
                        # Allocate sequentially by source, proportionally by raw_gap over ascending priority
                        def _alloc_source(df_src, pool, col_name):
                            if pool <= 0 or df_src.empty:
                                return df_src, 0
                            df_rem = df_src[df_src['raw_gap'] > 0].sort_values('priority')
                            if df_rem.empty:
                                return df_src, 0
                            total_gap = float(df_rem['raw_gap'].sum())
                            if total_gap <= 0:
                                return df_src, 0
                            weights = df_rem['raw_gap'].to_numpy(dtype=float) / total_gap
                            shares = np.floor(pool * weights).astype(np.int64)
                            # clip to raw_gap
                            shares = np.minimum(shares, df_rem['raw_gap'].to_numpy(dtype=np.int64))
                            # write back
                            df_src.loc[df_rem.index, col_name] = shares
                            df_src.loc[df_rem.index, 'raw_gap'] = (df_rem['raw_gap'].to_numpy(dtype=np.int64) - shares)
                            return df_src, int(shares.sum())
                        # in-transit
                        self_df, used_intrans = _alloc_source(self_df, pool_in_transit, 'alloc_intrans')
                        pool_in_transit -= used_intrans
                        # ODI
                        self_df, used_odi = _alloc_source(self_df, pool_odi, 'alloc_odi')
                        pool_odi -= used_odi
                        # future production
                        self_df, used_future = _alloc_source(self_df, pool_future_production, 'alloc_future')
                        pool_future_production -= used_future
                        # update cover
                        self_df['plan_order_cover'] = self_df['plan_order_cover'] + self_df['alloc_intrans'] + self_df['alloc_odi'] + self_df['alloc_future']
                        # write back to demand_rows
                        for row in self_df.itertuples(index=False):
                            i = int(row.idx)
                            demand_rows[i]['deploy_qty_with_plan_order'] = int(row.plan_order_cover)
                            demand_rows[i]['deploy_from_in_transit'] = int(row.alloc_intrans)
                            demand_rows[i]['deploy_from_open_deployment_inbound'] = int(row.alloc_odi)
                            demand_rows[i]['deploy_from_future_production'] = int(row.alloc_future)

                # 处理GAP和生成调拨计划（向量化）
                # Compute gaps for all rows
                if demand_rows:
                    df_gap = pd.DataFrame(demand_rows).copy()
                    df_gap['idx'] = np.arange(len(df_gap))
                    df_gap['receiving'] = [r.get('from_location', r.get('receiving', loc)) for r in demand_rows]
                    df_gap['is_self'] = df_gap['receiving'] == loc
                    df_gap['priority'] = df_gap['demand_element'].map(lambda x: demand_priority_map.get(x, 99))
                    # adjusted qty from map - 使用更安全的方式
                    adj_qty_gap_list = []
                    for i in range(len(demand_rows)):
                        if i in adjusted_qtys:
                            adj_qty_gap_list.append(int(adjusted_qtys[i]))
                        else:
                            adj_qty_gap_list.append(int(df_gap.iloc[i]['demand_qty']))
                    df_gap['adjusted_qty'] = adj_qty_gap_list
                    df_gap['allocated_invcon'] = [int(r.get('deployed_qty_invCon', 0) or 0) for r in demand_rows]
                    df_gap['plan_order_cover'] = [int(r.get('deploy_qty_with_plan_order', 0) or 0) for r in demand_rows]
                    df_gap['gap_qty'] = df_gap['adjusted_qty'] - df_gap['allocated_invcon'] - df_gap['plan_order_cover']
                    df_gap_pos = df_gap[df_gap['gap_qty'] > 0]
                    # upstream once per node
                    up_loc = get_upstream(loc, mat, network, sim_date, active_network_cache=active_network_cache)
                    if not df_gap_pos.empty:
                        # unfulfilled log entries
                        for row in df_gap_pos.itertuples(index=False):
                            unfulfilled_rows.append({
                                'date': row.plan_deploy_date,
                                'sending': loc,
                                'receiving': row.receiving,
                                'demand_qty': row.demand_qty,
                                'demand_element': row.demand_element,
                                'unfulfilled_qty': int(row.gap_qty),
                                'reason': "supply shortage"
                            })
                        # upstream gap routing
                        if up_loc:
                            for row in df_gap_pos.itertuples(index=False):
                                new_demand_element = f"net demand for {row.demand_element}"
                                req_dt = row.requirement_date if hasattr(row, 'requirement_date') and pd.notna(row.requirement_date) else row.plan_deploy_date
                                up_gap_next.setdefault((mat, up_loc), []).append({
                                    'demand_element': new_demand_element,
                                    'planned_qty': int(row.gap_qty),
                                    'leadtime': int(row.leadtime),
                                    'requirement_date': req_dt,
                                    'location': up_loc,
                                    'from_location': loc,
                                    'orig_location': row.orig_location if hasattr(row, 'orig_location') else row.location
                                })

                        
                        # print(f"      🔼 需求缺口: {gap_qty} [{d['demand_element']}] → 上游 {up_loc} (is_cross_node: {is_cross_node}, adjusted_qty: {adjusted_qty})")
                
                # if gap_count == 0:
                    # print(f"      🟢 无需求缺口")
                
                # 生成调拨计划行
                # 👉 Batch precompute lead time for this node (mat, loc)
                # Compute sending location type once
                sending_location_type = get_sending_location_type(
                    material=str(mat),
                    sending=str(loc),
                    sim_date=sim_date,
                    network_df=network,
                    location_layer_map=config.get('LocationLayerMap', {})
                )
                # PTF/LSK for plant, constant per (material, sending)
                ptf_val, lsk_val = _get_ptf_lsk(
                    material=str(mat),
                    site=str(loc),
                    m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
                    cache=ptf_lsk_cache
                )
                # Build lead time map for unique receivings (cross-node only)
                unique_receivings = set()
                for d in demand_rows:
                    rcv = d.get('from_location', d.get('receiving', loc))
                    if rcv != loc:
                        unique_receivings.add(str(rcv))
                lt_map: Dict[tuple[str, str, str], int] = {}
                if unique_receivings:
                    for rcv in unique_receivings:
                        # Base values from cache or df
                        if lead_time_cache is not None:
                            base_vals = lead_time_cache.get((str(loc), str(rcv)))
                            if base_vals is None:
                                PDT, GR, MCT = 0, 0, 0
                            else:
                                PDT, GR, MCT = base_vals
                        else:
                            row_lt = config['LeadTime'][(config['LeadTime']['sending'] == str(loc)) & (config['LeadTime']['receiving'] == str(rcv))]
                            PDT = int(pd.to_numeric(row_lt['PDT'], errors='coerce').fillna(0).iloc[0]) if not row_lt.empty else 0
                            GR  = int(pd.to_numeric(row_lt['GR'],  errors='coerce').fillna(0).iloc[0]) if not row_lt.empty else 0
                            MCT = int(pd.to_numeric(row_lt['MCT'], errors='coerce').fillna(0).iloc[0]) if not row_lt.empty else 0
                        if str(sending_location_type).lower() == 'plant':
                            base_lt = max(int(MCT), int(PDT) + int(GR))
                            leadtime_val = max(1, int(base_lt + int(ptf_val) + int(lsk_val) - 1))
                        else:
                            leadtime_val = max(1, int(int(PDT) + int(GR)))
                        lt_map[(str(mat), str(loc), str(rcv))] = int(leadtime_val)

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
                        # Use precomputed lead time map
                        leadtime_for_row = int(lt_map.get((str(mat), str(loc), str(receiving)), 1))
                    
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
        
        # 总计：需求收集+分配阶段用时（不含push/space quota）
        print(f"[M5] Demand collection only 用时: {demand_collect_only_elapsed:.3f}s")
        print(f"[M5] Demand collection+allocation 总用时: {time.perf_counter()-demand_collect_total_start:.3f}s")
        # push/soft-push再分配：使用 dynamic_soh；同时传入 projected_soh 作为分配基线
        dynamic_soh_for_push = dynamic_soh.copy()
        plan_push = push_softpush_allocation(
            deployment_plan_rows, config, dynamic_soh_for_push, sim_date,
            ptf_lsk_cache=ptf_lsk_cache, lead_time_cache=lead_time_cache, projected_soh=projected_soh,
            node_demands_map=global_node_demands_map
        )

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
    # 仅在非数据库模式下写入Excel文件
    if not skip_file_output:
        log_outputs(output_path, outputs)
    print(f"[M5] Full day total 用时: {time.perf_counter()-day_start:.3f}s")
    
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

# -*- coding: utf-8 -*-
"""
分配逻辑模块

提供库存分配、MOQ/RV应用、优先级分配等功能。
"""
from typing import Dict, List

import numpy as np
import pandas as pd


def apply_moq_rv(
    qty: int,
    moq: int,
    rv: int,
    is_cross_node: bool = True,
    max_qty: int = None
) -> int:
    """
    应用最小订货量/重订量约束。

    Args:
        qty: 原始数量
        moq: 最小订货量
        rv: 重订量
        is_cross_node: 是否跨节点
        max_qty: 最大允许数量（用于约束不超过订单量）

    Returns:
        int: 调整后的数量
    """
    if qty <= 0:
        return 0

    # 自循环调运不应用MOQ/RV约束
    if not is_cross_node:
        result = qty
    # 跨节点调运应用MOQ/RV约束
    elif qty < moq:
        result = moq
    else:
        result = int(np.ceil(qty / rv)) * rv
    
    # 应用最大数量约束：确保不超过订单量
    if max_qty is not None and result > max_qty:
        result = max_qty
    
    return result


def apply_grouped_moq_rv(
    demand_rows: List[dict],
    location: str,
    shipment_qty_limit: int = None
) -> Dict[int, int]:
    """
    按调运路径分组应用MOQ/RV。

    分组维度：(material, sending, receiving)
    仅跨节点（sending != receiving）应用MOQ/RV，自循环不应用。
    组内数量回分使用"最大余数法"。
    
    重要：调整后的总量不会超过 shipment_qty_limit（如果指定）。

    Args:
        demand_rows: 需求行列表
        location: 当前位置
        shipment_qty_limit: 订单量上限，用于约束部署量不超过订单量

    Returns:
        dict: 索引 -> 调整后数量
    """
    # 路径级分组
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

        group = route_groups[route_key]
        group['items'].append((i, d))
        group['total_qty'] += max(0, int(d.get('demand_qty', 0) or 0))
        group['moq'] = max(group['moq'], int(d.get('moq', 0) or 0))
        group['rv'] = max(group['rv'], int(d.get('rv', 0) or 0))

    adjusted_qtys = {}

    for route_key, group in sorted(route_groups.items()):
        total_qty = int(group['total_qty'] or 0)
        is_cross_node = group['is_cross_node']
        moq = int(group['moq'] or 0)
        rv = int(group['rv'] or 0)

        # 组合后的总量应用MOQ/RV
        adjusted_total = apply_moq_rv(
            total_qty, moq, rv, is_cross_node=is_cross_node,
            max_qty=shipment_qty_limit
        )

        # 组内"最大余数法"保和回分
        if total_qty <= 0:
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
        floors = []
        for pos, (item_idx, item) in enumerate(group['items']):
            original_qty = max(0, int(item.get('demand_qty', 0) or 0))
            exact = original_qty * r
            floor_val = int(np.floor(exact))
            remainder = float(exact - floor_val)
            floors.append((item_idx, floor_val, remainder, original_qty, pos))

        total_floor = int(sum(x[1] for x in floors))
        remaining = int(max(0, adjusted_total - total_floor))

        # 按remainder降序排序，并列用original_qty降序
        floors.sort(key=lambda x: (-x[2], -x[3], x[4]))

        # 写入floor分配
        for item_idx, floor_val, _, _, _ in floors:
            adjusted_qtys[item_idx] = int(floor_val)

        # 分配剩余量
        for k in range(min(remaining, len(floors))):
            idx = floors[k][0]
            adjusted_qtys[idx] += 1

    return adjusted_qtys


def apply_priority_allocation_vectorized(
    demand_rows: List[dict],
    adjusted_qtys: Dict[int, int],
    current_stock: int,
    demand_priority_map: Dict[str, int]
) -> int:
    """
    按需求优先级对库存进行向量化分配。

    高优先级需求先满足，最后一个被部分满足的优先级按比例分配。

    Args:
        demand_rows: 需求行列表
        adjusted_qtys: 调整后的需求量字典
        current_stock: 当前可用库存
        demand_priority_map: 需求类型优先级映射

    Returns:
        int: 剩余库存量
    """
    if not demand_rows:
        return current_stock

    n = len(demand_rows)
    df = pd.DataFrame(demand_rows).copy()
    df['idx'] = np.arange(n)
    df['priority'] = df['demand_element'].map(
        lambda x: demand_priority_map.get(x, 99)
    )

    # 计算adjusted_qty
    adjusted_qty_list = []
    for i in range(n):
        if i in adjusted_qtys:
            adjusted_qty_list.append(int(adjusted_qtys[i]))
        else:
            adjusted_qty_list.append(int(df.iloc[i]['demand_qty']))
    df['adjusted_qty'] = adjusted_qty_list
    df['deployed_qty_invCon'] = 0

    # 早期退出
    if current_stock <= 0:
        for i in range(n):
            demand_rows[i]['deployed_qty_invCon'] = 0
        return 0

    # 按优先级处理
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
            # 完全满足
            df.loc[df['idx'].isin(idxs), 'deployed_qty_invCon'] = adj
            current_stock -= group_total
            continue

        # 部分满足：按比例分配
        weights = adj.astype(float)
        shares = (
            (current_stock * (weights / float(group_total)))
            if group_total > 0 else np.zeros_like(weights)
        )
        alloc = np.minimum(np.floor(shares).astype(np.int64), adj)

        for j, idx in enumerate(idxs):
            df.loc[df['idx'] == idx, 'deployed_qty_invCon'] = alloc[j]

        current_stock = 0
        break

    # 写回
    for i, val in df[['idx', 'deployed_qty_invCon']].itertuples(index=False):
        demand_rows[int(i)]['deployed_qty_invCon'] = int(val)

    return current_stock


def allocate_pipeline_supply(
    demand_rows: List[dict],
    adjusted_qtys: Dict[int, int],
    location: str,
    demand_priority_map: Dict[str, int],
    future_intransit: dict,
    open_deployment_inbound: dict,
    future_production: dict
) -> None:
    """
    使用pipeline supply覆盖剩余gap。

    按优先级分配在途、开放调拨入库和未来生产。

    Args:
        demand_rows: 需求行列表（会被修改）
        adjusted_qtys: 调整后的需求量字典
        location: 当前位置
        demand_priority_map: 需求类型优先级映射
        future_intransit: 未来在途字典
        open_deployment_inbound: 开放调拨入库字典
        future_production: 未来生产字典
    """
    if not demand_rows:
        return

    # 初始化字段
    for d in demand_rows:
        d.setdefault('deploy_qty_with_plan_order', 0)
        d.setdefault('deploy_from_in_transit', 0)
        d.setdefault('deploy_from_open_deployment_inbound', 0)
        d.setdefault('deploy_from_future_production', 0)

    ndr_df = pd.DataFrame(demand_rows).copy()
    ndr_df['idx'] = np.arange(len(demand_rows))

    # receiving解析
    rec_arr = [
        r.get('from_location', r.get('receiving', location))
        for r in demand_rows
    ]
    ndr_df['receiving'] = rec_arr
    ndr_df['is_self'] = ndr_df['receiving'] == location
    ndr_df['priority'] = ndr_df['demand_element'].map(
        lambda x: demand_priority_map.get(x, 99)
    )

    # adjusted qty - 优化：向量化生成调整后数量
    ndr_df['adjusted_qty'] = ndr_df['idx'].map(
        lambda i: int(adjusted_qtys.get(int(i), int(ndr_df.at[int(i), 'demand_qty'])))
    )

    # 已分配量
    ndr_df['allocated_invcon'] = [
        int(r.get('deployed_qty_invCon', 0) or 0)
        for r in demand_rows
    ]
    ndr_df['plan_order_cover'] = [
        int(r.get('deploy_qty_with_plan_order', 0) or 0)
        for r in demand_rows
    ]

    # 处理自补货行
    self_df = ndr_df[ndr_df['is_self']].copy()
    if self_df.empty:
        return

    # 确保必要列存在
    for col in ['adjusted_qty', 'allocated_invcon', 'plan_order_cover']:
        if col not in self_df.columns:
            self_df[col] = 0

    self_df['raw_gap'] = (
        self_df['adjusted_qty'] -
        self_df['allocated_invcon'] -
        self_df['plan_order_cover']
    )
    self_df['alloc_intrans'] = 0
    self_df['alloc_odi'] = 0
    self_df['alloc_future'] = 0

    # 获取material
    if demand_rows:
        mat = demand_rows[0]['material']
        node_key = (mat, location)

        pool_in_transit = int(future_intransit.get(node_key, 0) or 0)
        pool_odi = int(open_deployment_inbound.get(node_key, 0) or 0)
        pool_future_production = int(future_production.get(node_key, 0) or 0)

        def _alloc_source(df_src, pool, col_name):
            """分配单一来源的供给。"""
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
            shares = np.minimum(
                shares, df_rem['raw_gap'].to_numpy(dtype=np.int64)
            )

            df_src.loc[df_rem.index, col_name] = shares
            df_src.loc[df_rem.index, 'raw_gap'] = (
                df_rem['raw_gap'].to_numpy(dtype=np.int64) - shares
            )

            return df_src, int(shares.sum())

        # 分配各来源
        self_df, used_intrans = _alloc_source(
            self_df, pool_in_transit, 'alloc_intrans'
        )
        self_df, used_odi = _alloc_source(
            self_df, pool_odi - used_intrans, 'alloc_odi'
        )
        self_df, used_future = _alloc_source(
            self_df, pool_future_production - used_odi, 'alloc_future'
        )

        # 更新cover
        self_df['plan_order_cover'] = (
            self_df['plan_order_cover'] +
            self_df['alloc_intrans'] +
            self_df['alloc_odi'] +
            self_df['alloc_future']
        )

        # 写回demand_rows
        for row in self_df.itertuples(index=False):
            i = int(row.idx)
            demand_rows[i]['deploy_qty_with_plan_order'] = int(
                row.plan_order_cover
            )
            demand_rows[i]['deploy_from_in_transit'] = int(row.alloc_intrans)
            demand_rows[i]['deploy_from_open_deployment_inbound'] = int(
                row.alloc_odi
            )
            demand_rows[i]['deploy_from_future_production'] = int(
                row.alloc_future
            )


def apply_receiving_space_quota(
    deployment_plan_rows: List[dict],
    receiving_space: pd.DataFrame,
    sim_date: pd.Timestamp,
    demand_priority_map: Dict[str, int]
) -> tuple:
    """
    应用接收端空间/能力配额。

    Args:
        deployment_plan_rows: 部署计划行列表
        receiving_space: 接收空间DataFrame
        sim_date: 仿真日期
        demand_priority_map: 需求类型优先级映射

    Returns:
        tuple: (结果DataFrame, 未满足日志列表)
    """
    import time
    t0 = time.perf_counter()

    df = pd.DataFrame(deployment_plan_rows)
    if df.empty:
        df['deployed_qty'] = []
        df['quota'] = []
        return df, []

    # 无接收空间限制
    if receiving_space.empty:
        df['deployed_qty'] = df['deployed_qty_invCon']
        df['quota'] = np.inf
        return df, []

    # 确保日期类型
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'date' in receiving_space.columns:
        receiving_space['date'] = pd.to_datetime(receiving_space['date'])

    # 计算优先级
    df['priority'] = df['demand_element'].map(
        lambda x: demand_priority_map.get(x, 99)
    )

    # 分离自补货和跨节点
    is_cross = df['sending'] != df['receiving']
    df_self = df[~is_cross].copy()
    df_cross = df[is_cross].copy()

    # 自补货直接通过
    if not df_self.empty:
        df.loc[df_self.index, 'deployed_qty'] = df_self['deployed_qty_invCon']
        df.loc[df_self.index, 'quota'] = np.inf

    # 无跨节点行
    if df_cross.empty:
        elapsed = time.perf_counter() - t0
        print(f"[M5] Receiving Space Quota 用时: {elapsed:.3f}s，受限条目: 0")
        return df, []

    # 合并配额
    space = receiving_space[['receiving', 'date', 'max_qty']].copy()
    space = space.rename(columns={'max_qty': 'quota'})
    df_cross = df_cross.merge(space, on=['receiving', 'date'], how='left')
    df_cross['quota'] = df_cross['quota'].fillna(np.inf)

    unfulfilled = []

    def _alloc_group(g: pd.DataFrame) -> pd.DataFrame:
        """分配单个接收端组。"""
        quota = g['quota'].iloc[0]
        total = g['deployed_qty_invCon'].sum()

        if total <= quota:
            g['deployed_qty'] = g['deployed_qty_invCon']
            g['quota'] = quota
            return g

        # 按优先级排序
        g = g.sort_values(['priority'])
        left = float(quota)
        deployed = np.zeros(len(g), dtype=np.int64)

        # 按优先级处理
        for p, block in g.groupby('priority', sort=True):
            block_total = block['deployed_qty_invCon'].sum()
            idx = block.index.to_numpy()

            if left >= block_total:
                deployed_vals = block['deployed_qty_invCon'].to_numpy()
            else:
                weights = block['deployed_qty_invCon'].to_numpy().astype(float)
                if block_total > 0:
                    shares = (left * (weights / float(block_total)))
                else:
                    shares = np.zeros_like(weights)
                deployed_vals = np.minimum(
                    np.floor(shares).astype(np.int64),
                    block['deployed_qty_invCon'].to_numpy()
                )

            deployed[g.index.get_indexer(idx)] = deployed_vals
            left -= deployed_vals.sum()

            if left <= 0:
                break

        g['deployed_qty'] = deployed
        g['quota'] = quota
        return g

    # 分组处理 - 🔧 修复：sort=True确保分组顺序稳定
    allocated = (
        df_cross.groupby(['receiving', 'date'], sort=True, group_keys=False)
        .apply(_alloc_group)
    )

    # 写回结果
    df.loc[allocated.index, 'deployed_qty'] = allocated['deployed_qty']
    df.loc[allocated.index, 'quota'] = allocated['quota']

    # 构建未满足日志
    gaps = allocated[
        allocated['deployed_qty_invCon'] > allocated['deployed_qty']
    ]
    if not gaps.empty:
        # 🔧 修复：确保迭代顺序稳定
        sort_cols = ['date', 'sending', 'receiving', 'demand_element']
        sort_cols = [c for c in sort_cols if c in gaps.columns]
        if sort_cols:
            gaps = gaps.sort_values(by=sort_cols).reset_index(drop=True)
        
        unfulfilled = [
            {
                'date': row.date,
                'sending': row.sending,
                'receiving': row.receiving,
                'material': row.material,
                'demand_qty': row.demand_qty,
                'demand_element': row.demand_element,
                'unfulfilled_qty': int(
                    row.deployed_qty_invCon - row.deployed_qty
                ),
                'reason': 'space constraint'
            }
            for row in gaps.itertuples(index=False)
        ]

    # 填充未处理行
    df['deployed_qty'] = df['deployed_qty'].fillna(df['deployed_qty_invCon'])
    df['quota'] = df['quota'].fillna(np.nan)

    elapsed = time.perf_counter() - t0
    print(
        f"[M5] Receiving Space Quota 用时: {elapsed:.3f}s，"
        f"受限条目: {len(unfulfilled)}"
    )

    return df, unfulfilled

"""
Module3 层级分配模块。

负责供应链网络中节点的层级分配。
按物料维度分配网络层级（与 Module5 逻辑一致）。
"""

import time
from collections import defaultdict, deque
from typing import Tuple

import pandas as pd


def assign_location_layers(network_df: pd.DataFrame) -> pd.DataFrame:
    """
    按物料维度分配网络层级（与 Module5 逻辑一致）。

    参数：
        network_df: 网络配置数据，必须包含 material, location, sourcing 列

    返回：
        pd.DataFrame: 包含 material, location, layer 列的映射DataFrame
    """
    _t_func = time.perf_counter()

    if network_df.empty:
        empty_df = pd.DataFrame({'material': [], 'location': [], 'layer': []})
        print(f"[M3] assign_location_layers total: {time.perf_counter()-_t_func:.3f}s, locations=0")
        return empty_df

    layer_rows = []
    for material, mat_df in network_df.groupby('material', dropna=False):
        if mat_df.empty:
            continue

        children = defaultdict(list)
        parents = defaultdict(list)
        for row in mat_df.itertuples(index=False):
            sourcing_val = getattr(row, 'sourcing', None)  # type: ignore[attr-defined]
            location_val = getattr(row, 'location', None)  # type: ignore[attr-defined]

            sourcing_valid = _is_valid_value(sourcing_val)
            location_valid = _is_valid_value(location_val)
            if sourcing_valid and location_valid:
                s_val = str(sourcing_val)
                l_val = str(location_val)
                children[s_val].append(l_val)
                parents[l_val].append(s_val)

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
        queue = deque((root, 0) for root in true_roots)
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
            # 处理 groupby 产生的 NaN material (dropna=False)
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
        empty_df = pd.DataFrame({'material': [], 'location': [], 'layer': []})
        print(f"[M3] assign_location_layers total: {time.perf_counter()-_t_func:.3f}s, locations=0")
        return empty_df

    layer_df = pd.DataFrame(layer_rows).sort_values(['material', 'layer', 'location']).reset_index(drop=True)
    print(f"[M3] assign_location_layers total: {time.perf_counter()-_t_func:.3f}s, locations={len(layer_df)}")
    return layer_df


def _is_valid_value(val) -> bool:
    """检查值是否有效。"""
    return (
        val is not None and
        pd.notna(val) and
        str(val).strip() != ''
    )

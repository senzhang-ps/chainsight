"""
Module3 层级分配模块。

负责供应链网络中节点的层级分配。
"""

import time
from collections import defaultdict, deque
from typing import Tuple

import pandas as pd


def assign_location_layers(network_df: pd.DataFrame) -> pd.DataFrame:
    """
    分配供应链网络中各节点的层级。

    Args:
        network_df: 网络配置数据

    Returns:
        pd.DataFrame: 包含location和layer列的映射DataFrame
    """
    t_start = time.perf_counter()

    if network_df.empty:
        return pd.DataFrame({'location': [], 'layer': []})

    children, parents = _build_parent_child_graph(network_df)
    all_locations = _collect_all_locations(network_df)
    true_roots = _identify_root_nodes(all_locations, children, parents)
    layer_dict = _assign_layers_bfs(true_roots, children)
    layer_dict = _handle_unassigned(layer_dict, all_locations)
    layer_df = _create_layer_dataframe(layer_dict)

    elapsed = time.perf_counter() - t_start
    print(f"[M3] assign_location_layers: {elapsed:.3f}s, locs={len(layer_df)}")
    return layer_df


def _build_parent_child_graph(network_df: pd.DataFrame) -> Tuple[dict, dict]:
    """构建父子关系图。"""
    children = defaultdict(list)
    parents = defaultdict(list)

    for row in network_df.itertuples():
        sourcing = row.sourcing
        location = row.location

        sourcing_ok = _is_valid_value(sourcing)
        location_ok = _is_valid_value(location)

        if sourcing_ok and location_ok:
            children[sourcing].append(location)
            parents[location].append(sourcing)

    return children, parents


def _is_valid_value(val) -> bool:
    """检查值是否有效。"""
    return (
        val is not None and
        pd.notna(val) and
        str(val).strip() != ''
    )


def _collect_all_locations(network_df: pd.DataFrame) -> set:
    """收集所有地点。"""
    locs = set(network_df['location'].dropna())
    srcs = set(network_df['sourcing'].dropna())
    return locs.union(srcs)


def _identify_root_nodes(
    all_locations: set,
    children: dict,
    parents: dict
) -> list:
    """识别根节点。"""
    potential = [loc for loc in all_locations if not parents[loc]]

    true_roots = []
    for loc in potential:
        if loc in children:
            true_roots.append(loc)
        else:
            has_incoming = any(
                loc in parents.get(other, [])
                for other in all_locations
            )
            if not has_incoming:
                true_roots.append(loc)

    return true_roots if true_roots else potential


def _assign_layers_bfs(roots: list, children: dict) -> dict:
    """使用BFS分配层级。"""
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

    return layer_dict


def _handle_unassigned(layer_dict: dict, all_locations: set) -> dict:
    """处理未分配的节点。"""
    unassigned = [loc for loc in all_locations if loc not in layer_dict]
    if unassigned:
        max_layer = max(layer_dict.values()) if layer_dict else 0
        for loc in unassigned:
            layer_dict[loc] = max_layer + 1
    return layer_dict


def _create_layer_dataframe(layer_dict: dict) -> pd.DataFrame:
    """创建层级DataFrame。"""
    layer_df = pd.DataFrame([
        {'location': loc, 'layer': layer}
        for loc, layer in layer_dict.items()
    ])
    return layer_df.sort_values('layer')

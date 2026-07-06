"""DAG Data Processing — Chain assignment and G6-ready JSON from NetworkX DiGraph.

Rendering is LLM-native: use ``uiuxpromax`` skill + data from ``_build_js_data()``
to generate interactive HTML on demand. No hardcoded HTML template.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Union

import networkx as nx


# ============================================================
# Chain assignment logic
# ============================================================
# Maps (source_prefix, target_prefix, edge_index_range) to chain IDs.
# Since the YAML doesn't store chain info, we derive it from the
# causal_dag.yaml edge comments convention: edges are ordered by chain.
# For robustness, we tag edges by source→target path membership.

_CHAIN_SEEDS: Dict[Union[int, str], List[str]] = {
    1: ["param:safety_stock_qty"],
    2: ["param:min_batch", "param:rv"],
    3: ["param:pdt", "param:gr_days"],
    4: ["param:dps_percent", "param:ao_percent", "param:advance_days", "param:order_calendar"],
    5: ["param:prd_rate", "param:changeover_time", "param:line_capacity",
        "param:delegate_line", "param:error_std_percent"],
    6: ["param:truck_type", "param:capacity_qty_in_weight",
        "param:capacity_qty_in_volume", "param:truck_number"],
    7: ["param:push_pull_model", "param:moq", "param:rv_deploy",
        "param:space_capacity"],
    8: ["param:max_wait_days"],
}

# Nodes that belong to WFR/VFR side-effect chain (chain 8)
_CHAIN8_NODES = {"param:wfr", "param:vfr", "param:max_wait_days",
                 "proc:waiting_time", "proc:total_replenishment_lead_time"}

# Cross-chain edges: process→process or process→metric links that don't
# originate from a single chain's parameter seeds.
_CROSS_CHAIN_SOURCES = {"proc:inventory_level", "proc:deployment_batch_size",
                        "proc:production_output", "proc:demand_allocation"}


def _assign_chains(graph: nx.DiGraph) -> Tuple[Dict[str, List], Dict[str, Union[int, str]]]:
    """Assign chain IDs to nodes and edges for visualization filtering.

    Returns (node_chains, edge_chains) where:
    - node_chains: {node_id: [chain_id, ...]}
    - edge_chains: {f"{source}->{target}": chain_id}
    """
    node_chains: Dict[str, List] = {n: [] for n in graph.nodes()}
    edge_chains: Dict[str, Union[int, str]] = {}

    # Assign chain membership by BFS from each chain's seed parameters
    for chain_id, seeds in _CHAIN_SEEDS.items():
        visited = set()
        queue = list(seeds)
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            if chain_id not in node_chains.get(cur, []):
                node_chains.setdefault(cur, []).append(chain_id)
            for _, tgt in graph.out_edges(cur):
                edge_key = f"{cur}->{tgt}"
                if edge_key not in edge_chains:
                    # Don't BFS through cross-chain source nodes for non-cross chains
                    if cur in _CROSS_CHAIN_SOURCES and cur not in seeds:
                        continue
                    edge_chains[edge_key] = chain_id
                    queue.append(tgt)

    # Chain 6, 8, 9 share wfr/vfr — assign by target:
    #   vehicle_loading → chain 6, waiting_time → chain 8,
    #   forced_dispatch_penalty → chain 9
    for u, v in graph.edges():
        key = f"{u}->{v}"
        if key in edge_chains:
            continue
        if u in {"param:wfr", "param:vfr"}:
            if v == "proc:vehicle_loading":
                chain = 6
            elif v == "proc:waiting_time":
                chain = 8
            elif v == "proc:forced_dispatch_penalty":
                chain = 9
            else:
                continue
            edge_chains[key] = chain
            node_chains.setdefault(u, [])
            if chain not in node_chains[u]:
                node_chains[u].append(chain)

    # Cross-chain edges
    for u, v in graph.edges():
        key = f"{u}->{v}"
        if key not in edge_chains and u in _CROSS_CHAIN_SOURCES:
            edge_chains[key] = "cross"
            if "cross" not in node_chains.get(u, []):
                node_chains.setdefault(u, []).append("cross")
            if "cross" not in node_chains.get(v, []):
                node_chains.setdefault(v, []).append("cross")

    # Fill any remaining unassigned edges
    for u, v in graph.edges():
        key = f"{u}->{v}"
        if key not in edge_chains:
            # Check if both endpoints share a chain
            u_chains = node_chains.get(u, [])
            v_chains = node_chains.get(v, [])
            common = [c for c in u_chains if c in v_chains and c != "cross"]
            edge_chains[key] = common[0] if common else "cross"

    return node_chains, edge_chains


def _build_js_data(graph: nx.DiGraph) -> str:
    """Convert NetworkX DiGraph to JavaScript dagData JSON string."""
    node_chains, edge_chains = _assign_chains(graph)

    js_nodes = []
    for node_id, data in graph.nodes(data=True):
        node_type = data.get("type", "process")
        label = data.get("label", node_id.split(":")[-1])
        # Insert newline in middle of multi-word labels for compact display
        words = label.split()
        if len(words) > 2:
            mid = len(words) // 2
            label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        elif len(words) == 2:
            label = words[0] + "\n" + words[1]

        node = {
            "id": node_id,
            "nodeType": node_type,
            "label": label,
            "module": data.get("module", ""),
            "description": data.get("description", ""),
            "chain": node_chains.get(node_id, []),
        }
        if data.get("source_config"):
            node["sourceConfig"] = data["source_config"]
        if data.get("category"):
            node["category"] = data["category"]
        if data.get("unit"):
            node["unit"] = data["unit"]
        js_nodes.append(node)

    js_edges = []
    for u, v, data in graph.edges(data=True):
        edge_key = f"{u}->{v}"
        polarity = data.get("polarity")
        edge = {
            "source": u,
            "target": v,
            "edgeType": data.get("type", "influences"),
            "polarity": polarity,
            "description": data.get("description", ""),
            "chain": edge_chains.get(edge_key, "cross"),
        }
        js_edges.append(edge)

    return json.dumps({"nodes": js_nodes, "edges": js_edges}, ensure_ascii=False, indent=2)


def _default_yaml_path() -> str:
    """Resolve default causal_dag.yaml path relative to this script."""
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(tools_dir), "causal_dag.yaml")

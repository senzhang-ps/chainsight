"""DAG Loader — Parse causal_dag.yaml into a NetworkX DiGraph."""

import os
import yaml
import networkx as nx


def load_dag(yaml_path: str) -> nx.DiGraph:
    """Load causal DAG from YAML file and return a NetworkX DiGraph.

    Each node has attributes: type, label, module, description,
    and optionally: source_config, category, unit.
    Each edge has attributes: type, polarity, description.

    The graph stores declared_node_ids as a graph-level attribute
    (the set of node IDs explicitly declared in the YAML nodes list)
    so that validators can detect edges referencing undeclared nodes.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    g = nx.DiGraph()
    g.graph["version"] = data.get("version", "unknown")
    g.graph["description"] = data.get("description", "")

    declared_node_ids = set()
    for node in data.get("nodes", []):
        node_id = node["id"]
        declared_node_ids.add(node_id)
        attrs = {k: v for k, v in node.items() if k != "id"}
        g.add_node(node_id, **attrs)

    g.graph["declared_node_ids"] = declared_node_ids

    for edge in data.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        g.add_edge(source, target, **attrs)

    return g


def _default_yaml_path() -> str:
    """Resolve default causal_dag.yaml path relative to this script."""
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(tools_dir), "causal_dag.yaml")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else _default_yaml_path()
    graph = load_dag(path)
    param_count = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "parameter")
    proc_count = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "process")
    metric_count = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "metric")
    print(f"Loaded DAG: {graph.number_of_nodes()} nodes "
          f"({param_count} param, {proc_count} process, {metric_count} metric), "
          f"{graph.number_of_edges()} edges")

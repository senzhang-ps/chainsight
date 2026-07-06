"""DAG Validator — Check causal DAG integrity against §4.6 rules."""

from __future__ import annotations

import json
import os
from typing import List

import networkx as nx


def validate_dag(graph: nx.DiGraph) -> List[str]:
    """Validate a causal DAG and return a list of error messages.

    Returns an empty list if the DAG is valid.

    Validation rules (from design doc §4.6):
    1. Graph must be acyclic (DAG property)
    2. Every Parameter node must have at least one path to a Metric node
    3. Every Metric node must be reachable from at least one Parameter node
    4. All node IDs referenced in edges must exist in the declared nodes list
    5. Edge polarity must be "+", "-", or null (None)
    6. Edge type must be "influences", "constrains", or "triggers"
    7. Bidirectional consistency with ontology (see validate_ontology_dag_consistency)
    """
    errors = []

    # Rule 1: Acyclic check
    if not nx.is_directed_acyclic_graph(graph):
        cycles = list(nx.simple_cycles(graph))
        for cycle in cycles[:5]:  # report up to 5 cycles
            errors.append(f"Rule 1 — Cycle detected: {' → '.join(cycle)}")

    # Classify nodes by type
    param_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "parameter"]
    metric_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "metric"]

    # Rule 2: Every Parameter node must reach at least one Metric node
    for p in param_nodes:
        reachable = nx.descendants(graph, p)
        if not any(m in reachable for m in metric_nodes):
            errors.append(f"Rule 2 — Parameter '{p}' has no path to any Metric node")

    # Rule 3: Every Metric node must be reachable from at least one Parameter node
    for m in metric_nodes:
        ancestors = nx.ancestors(graph, m)
        if not any(p in ancestors for p in param_nodes):
            errors.append(f"Rule 3 — Metric '{m}' is not reachable from any Parameter node")

    # Rule 4: All edge endpoints must exist in declared nodes
    # Uses declared_node_ids stored by load_dag, not graph.nodes()
    # (NetworkX auto-creates nodes on add_edge, so graph.nodes() always contains them)
    declared = graph.graph.get("declared_node_ids", set(graph.nodes()))
    for u, v in graph.edges():
        if u not in declared:
            errors.append(f"Rule 4 — Edge source '{u}' not in declared node list")
        if v not in declared:
            errors.append(f"Rule 4 — Edge target '{v}' not in declared node list")

    # Rule 5: Edge polarity must be "+", "-", or None
    valid_polarities = {"+", "-", None}
    for u, v, d in graph.edges(data=True):
        polarity = d.get("polarity")
        if polarity not in valid_polarities:
            errors.append(
                f"Rule 5 — Edge '{u}' → '{v}' has invalid polarity '{polarity}'"
            )

    # Rule 6: Edge type must be a recognized value
    valid_edge_types = {"influences", "constrains", "triggers"}
    for u, v, d in graph.edges(data=True):
        edge_type = d.get("type")
        if edge_type not in valid_edge_types:
            errors.append(
                f"Rule 6 — Edge '{u}' → '{v}' has invalid type '{edge_type}'"
            )

    return errors


def _default_ontology_path() -> str:
    """Resolve default chainsight.jsonld path relative to this script."""
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    kg_dir = os.path.dirname(os.path.dirname(tools_dir))
    return os.path.join(kg_dir, "schema", "chainsight.jsonld")


def validate_ontology_dag_consistency(
    graph: nx.DiGraph, ontology_path: str | None = None
) -> List[str]:
    """Rule 7 — Bidirectional consistency between ontology instances and DAG nodes.

    Checks:
    - Every cs:TunableParameter instance should have a corresponding param: node in DAG
    - Every param: node in DAG should have a corresponding cs:TunableParameter instance
    - Every cs:Metric instance should have a corresponding metric: node in DAG
    - Every metric: node in DAG should have a corresponding cs:Metric instance
    """
    errors = []
    ontology_path = ontology_path or _default_ontology_path()

    if not os.path.exists(ontology_path):
        errors.append(f"Rule 7 — Ontology file not found: {ontology_path}")
        return errors

    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    graph_items = ontology.get("@graph", [])

    # Extract ontology param names from @id (cs:param_X → X)
    ontology_params: set[str] = set()
    for item in graph_items:
        item_type = item.get("@type", "")
        if "TunableParameter" in str(item_type):
            item_id = item.get("@id", "")
            if item_id.startswith("cs:param_"):
                ontology_params.add(item_id[len("cs:param_"):])

    # Extract ontology metric names from @id (cs:metric_X → X)
    ontology_metrics: set[str] = set()
    for item in graph_items:
        item_type = item.get("@type", "")
        if "Metric" in str(item_type):
            item_id = item.get("@id", "")
            if item_id.startswith("cs:metric_"):
                ontology_metrics.add(item_id[len("cs:metric_"):])

    # DAG param and metric node names (strip prefix)
    dag_params: set[str] = set()
    dag_metrics: set[str] = set()
    for n, d in graph.nodes(data=True):
        ntype = d.get("type")
        if ntype == "parameter" and n.startswith("param:"):
            dag_params.add(n[len("param:"):])
        elif ntype == "metric" and n.startswith("metric:"):
            dag_metrics.add(n[len("metric:"):])

    # Bidirectional check — parameters
    for p in sorted(ontology_params - dag_params):
        errors.append(f"Rule 7 — Ontology TunableParameter '{p}' missing from DAG")
    for p in sorted(dag_params - ontology_params):
        errors.append(f"Rule 7 — DAG param node '{p}' missing from ontology")

    # Bidirectional check — metrics
    for m in sorted(ontology_metrics - dag_metrics):
        errors.append(f"Rule 7 — Ontology Metric '{m}' missing from DAG")
    for m in sorted(dag_metrics - ontology_metrics):
        errors.append(f"Rule 7 — DAG metric node '{m}' missing from ontology")

    return errors


def _default_yaml_path() -> str:
    """Resolve default causal_dag.yaml path relative to this script."""
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))
    return os.path.join(project_root, "knowledge_graph", "instances", "causal", "causal_dag.yaml")


if __name__ == "__main__":
    import sys
    from knowledge_graph.tools.dag.dag_loader import load_dag

    path = sys.argv[1] if len(sys.argv) > 1 else _default_yaml_path()
    graph = load_dag(path)
    errs = validate_dag(graph)
    errs += validate_ontology_dag_consistency(graph)
    if errs:
        print(f"Validation FAILED — {len(errs)} error(s):")
        for e in errs:
            print(f"  ✗ {e}")
    else:
        print("Validation PASSED — 0 errors")

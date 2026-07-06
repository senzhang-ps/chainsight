"""Ontology Data Processing — Parse chainsight.jsonld into G6-ready JSON.

Rendering is LLM-native: use ``uiuxpromax`` skill + data from ``_build_js_data()``
to generate interactive HTML on demand. No hardcoded HTML template.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


# Top-level class IDs (no subClassOf) — used by _resolve_parent()
_TOP_CLASSES = {"cs:Network", "cs:Product", "cs:Resource", "cs:Inventory",
                "cs:Plan", "cs:Parameter", "cs:Metric"}


def parse_ontology(jsonld_path: str) -> Dict:
    """Parse chainsight.jsonld and extract classes, properties, instances.

    Returns dict with keys: classes, object_properties, datatype_properties, instances, stats
    """
    with open(jsonld_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = list(data.get("@graph", []))

    # ABox individuals (TunableParameter + Metric instances) live outside the TBox
    # schema under instances/ontology/. Merge them so the parsed graph is complete.
    kg_root = os.path.dirname(os.path.dirname(os.path.abspath(jsonld_path)))
    inst_dir = os.path.join(kg_root, "instances", "ontology")
    for fname in ("parameters.jsonld", "metrics.jsonld"):
        fpath = os.path.join(inst_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                graph.extend(json.load(f).get("@graph", []))

    classes = {}          # id -> {label, comment, subClassOf, sourceConfig, sourceType, properties}
    obj_props = []        # [{id, label, domain, range, comment}]
    dt_props = {}         # domain_class_id -> [{label, range, comment}]
    instances = {}        # class_id -> [{id, label, ...attrs}]

    for entry in graph:
        entry_type = entry.get("@type", "")
        entry_id = entry.get("@id", "")

        if entry_type == "owl:Class":
            cls = {
                "id": entry_id,
                "label": entry.get("label", entry_id.split(":")[-1]),
                "comment": entry.get("comment", ""),
                "subClassOf": entry.get("subClassOf"),
                "sourceConfig": entry.get("cs:source_config"),
                "sourceType": entry.get("cs:source_type"),
                "properties": entry.get("cs:properties"),
            }
            classes[entry_id] = cls

        elif entry_type == "owl:ObjectProperty":
            prop = {
                "id": entry_id,
                "label": entry.get("label", ""),
                "domain": entry.get("domain"),
                "range": entry.get("range"),
                "comment": entry.get("comment", ""),
            }
            obj_props.append(prop)

        elif entry_type == "owl:DatatypeProperty":
            domain = entry.get("domain")
            prop_info = {
                "label": entry.get("label", ""),
                "range": entry.get("range", ""),
                "comment": entry.get("comment", ""),
            }
            if domain:
                dt_props.setdefault(domain, []).append(prop_info)
            else:
                # Shared properties (no specific domain) — attach to a generic key
                dt_props.setdefault("_shared", []).append(prop_info)

        elif entry_type not in ("owl:Ontology", "owl:AnnotationProperty", ""):
            # Instance — @type is something like "cs:TunableParameter" or "cs:ServiceMetric"
            inst_type = entry_type
            inst = {"id": entry_id, "label": entry.get("label", "")}
            # Collect relevant attributes
            for k, v in entry.items():
                if k.startswith("cs:") and isinstance(v, str):
                    inst[k.replace("cs:", "")] = v
                elif k == "comment":
                    inst["comment"] = v
            instances.setdefault(inst_type, []).append(inst)

    # Resolve instance types to parent classes (e.g., cs:ServiceMetric instances
    # should also appear under cs:Metric)
    # Build subclass → parent mapping for instance grouping
    class_to_parent = {}
    for cid, cls in classes.items():
        if cls["subClassOf"]:
            class_to_parent[cid] = cls["subClassOf"]

    # Count stats
    n_params = len(instances.get("cs:TunableParameter", []))
    n_metrics = sum(len(v) for k, v in instances.items() if k != "cs:TunableParameter")

    return {
        "classes": classes,
        "object_properties": obj_props,
        "datatype_properties": dt_props,
        "instances": instances,
        "class_to_parent": class_to_parent,
        "stats": {
            "classes": len(classes),
            "objectProperties": len([p for p in obj_props if p["domain"] and p["range"]]),
            "params": n_params,
            "metrics": n_metrics,
        },
    }


def _resolve_parent(class_id: str, classes: Dict) -> Optional[str]:
    """Walk up subClassOf chain to find top-level parent."""
    visited = set()
    current = class_id
    while current and current not in _TOP_CLASSES:
        if current in visited:
            break
        visited.add(current)
        cls = classes.get(current)
        if not cls or not cls.get("subClassOf"):
            break
        current = cls["subClassOf"]
    return current if current in _TOP_CLASSES else None


def _build_js_data(parsed: Dict) -> str:
    """Convert parsed ontology to JavaScript ontologyData JSON string."""
    classes = parsed["classes"]
    obj_props = parsed["object_properties"]
    dt_props = parsed["datatype_properties"]
    instances = parsed["instances"]

    js_nodes = []
    for cid, cls in classes.items():
        is_top = cid in _TOP_CLASSES
        parent = _resolve_parent(cid, classes) if not is_top else cid

        # Collect DatatypeProperties for this class
        class_dt_props = dt_props.get(cid, [])
        # Also include shared properties for top-level classes
        if is_top:
            class_dt_props = class_dt_props + dt_props.get("_shared", [])

        # Collect instances for this class
        class_instances = instances.get(cid, [])

        node = {
            "id": cid,
            "label": cls["label"],
            "nodeType": "top-class" if is_top else "subclass",
            "parentClass": parent,
            "comment": cls["comment"],
            "sourceConfig": cls.get("sourceConfig"),
            "sourceType": cls.get("sourceType"),
            "datatypeProperties": class_dt_props,
            "instances": class_instances,
            "classProperties": cls.get("properties"),
        }
        js_nodes.append(node)

    js_edges = []
    edge_idx = 0

    # subClassOf edges
    for cid, cls in classes.items():
        if cls["subClassOf"]:
            js_edges.append({
                "id": f"edge-sub-{edge_idx}",
                "source": cid,
                "target": cls["subClassOf"],
                "edgeType": "subClassOf",
                "label": "",
                "comment": "",
            })
            edge_idx += 1

    # ObjectProperty edges (only those with explicit domain + range)
    for prop in obj_props:
        if prop["domain"] and prop["range"]:
            js_edges.append({
                "id": f"edge-prop-{edge_idx}",
                "source": prop["domain"],
                "target": prop["range"],
                "edgeType": "objectProperty",
                "label": prop["label"],
                "comment": prop["comment"],
                "propertyId": prop["id"],
            })
            edge_idx += 1

    # Instance nodes and instanceOf edges
    for inst_type, inst_list in instances.items():
        parent = _resolve_parent(inst_type, classes) if inst_type not in _TOP_CLASSES else inst_type
        for inst in inst_list:
            js_nodes.append({
                "id": inst["id"],
                "label": inst["label"],
                "nodeType": "instance",
                "parentClass": parent,
                "instanceOf": inst_type,
                "comment": inst.get("comment", ""),
                "instAttrs": {k: v for k, v in inst.items() if k not in ("id", "label", "comment")},
            })
            js_edges.append({
                "id": f"edge-inst-{edge_idx}",
                "source": inst["id"],
                "target": inst_type,
                "edgeType": "instanceOf",
                "label": "",
                "comment": "",
            })
            edge_idx += 1

    data = {
        "nodes": js_nodes,
        "edges": js_edges,
        "stats": parsed["stats"],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _default_jsonld_path() -> str:
    """Resolve default chainsight.jsonld path relative to this script."""
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(tools_dir), "chainsight.jsonld")

# DAG Health Report

> File-derived check, generated 2026-06-07 07:50:34Z (UTC).
> Source: `instances/causal/causal_dag.yaml`.
> Mirrors rules 1-6 of `knowledge_graph/tools/dag/dag_validator.py`.

## Structural integrity (rules 1-6): PASS

All structural rules passed: graph is acyclic; every parameter reaches a metric; every metric is reachable from a parameter; all edge endpoints are declared; edge polarity and type values are valid.

## Summary

- Nodes: 66 (metric: 13, parameter: 28, process: 25)
- Edges: 87 (by polarity - +: 53, -: 18, None: 16; by type - constrains: 6, influences: 81)
- Parameters: 28 | Metrics: 13

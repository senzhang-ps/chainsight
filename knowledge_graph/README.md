# knowledge_graph/

ChainSight's knowledge representation layer — a git-tracked **world model** that
AI coding agents read directly to understand the ChainSight domain, its tunable
parameters, the KPIs they drive, and the data behind them.

It has three parts:

- **`schema/`** — the ontology (TBox): what a `Material`, `Location`,
  `Parameter`, `Metric`, etc. *is*.
- **`instances/`** — instance data (ABox): the causal DAG, data-catalog
  entries, skill registrations, and domain rules.
- **`tools/`** — Python utilities for validating, querying, and visualizing the
  above.

These files are the source of truth. Agents consume them by reading the
JSON-LD / YAML directly — no database or runtime service is required.

## TBox vs ABox

- **TBox** (`schema/`) — class and property *definitions*: what each entity
  *is*.
- **ABox** (`instances/`) — *instances* of those definitions: the causal DAG,
  skill and data-catalog registrations, and rules.

## Directory layout

| Path | Role |
|------|------|
| [schema/](schema/) | TBox — JSON-LD ontologies (core `cs:` + meta `csmeta:`) |
| [instances/](instances/) | ABox — causal DAG, data catalog, skills, rules |
| [tools/](tools/) | Python tooling for validation, query, and visualization |
| [digest.yaml](digest.yaml) | Generated stats snapshot of the graph |
| `__init__.py` | Package marker |

## Schema (TBox)

| File | Role |
|------|------|
| [schema/chainsight.jsonld](schema/chainsight.jsonld) | Core supply-chain ontology (`cs:`) — 54 classes: Material, Location, TransportationLane, plans, parameters, metrics (TBox only; parameter & metric individuals live in `instances/ontology/`) |
| [schema/meta.jsonld](schema/meta.jsonld) | Meta-ontology (`csmeta:`) — Agent / Skill / Tool classes for the operational layer |
| [schema/context.jsonld](schema/context.jsonld) | Shared JSON-LD `@context` for both ontologies |

## Instances (ABox)

| Path | Role |
|------|------|
| [instances/causal/causal_dag.yaml](instances/causal/causal_dag.yaml) | Parameter↔KPI causal DAG (nodes, edges, chains) |
| [instances/causal/dag-health-report.md](instances/causal/dag-health-report.md) | DAG coverage / health report |
| [instances/ontology/](instances/ontology/) | `cs:` ABox individuals split from the TBox — `parameters.jsonld` (28 TunableParameter) + `metrics.jsonld` (13 Metric) |
| [instances/data-catalog/](instances/data-catalog/) | Databricks table metadata — see its [README](instances/data-catalog/README.md) |
| [instances/skills/](instances/skills/) | Registrations for the skills under [.agents/skills/](../.agents/skills/) |
| [instances/rules/](instances/rules/) | Domain data & routing rules (CFR/query conventions, M4 source routing, forecast-type clarification) |

## Tooling

| Module | Role |
|--------|------|
| [tools/kg_validate.py](tools/kg_validate.py) | Offline instance-consistency checks (no external service) |
| [tools/dag/](tools/dag/) | Causal-DAG loader, validator, visualizer |
| [tools/ontology/](tools/ontology/) | Ontology parser + presentation data prep |

## Usage

```powershell
# Offline consistency check (no external service required)
python -m knowledge_graph.tools.kg_validate
```

```python
# Parse the ontology for inspection / visualization
from knowledge_graph.tools.ontology.ontology_visualizer import parse_ontology
data = parse_ontology("knowledge_graph/schema/chainsight.jsonld")
```

`digest.yaml` is a generated snapshot of graph statistics (class / DAG / skill /
table counts). Regenerate it after editing `schema/` or `instances/`.

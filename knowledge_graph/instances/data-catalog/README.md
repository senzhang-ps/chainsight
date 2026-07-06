# Data Catalog — Instance Cache

Per-table metadata for Databricks tables that ChainSight uses. One Markdown
file per table with **YAML frontmatter + Markdown body**.

## Layout

```
data-catalog/
├── README.md          ← this file
├── registry.yaml      ← list of in-scope table FQNs (source of truth for sync)
└── tables/
    └── <table_name>.md   ← one per table; name = last segment of FQN
```

## File format

```markdown
---
fqn: b2b_prd_cne2_cdl_dbr01.cdl_ps_hana_prd.sl.<table_name>
description: One-line human summary
synced_at: '2026-04-10T10:20:39Z'
tags: [demand_forecast, scope_source, transactional]
related_config: [M1_DemandForecast]
scope_dimensions: [material_num, site_id, frcst_vers_date]
columns:
  - {name: frcst_vers_date, type: INT}
  - {name: fcst_type, type: STRING, desc: 'BOP or LBE'}
---

# <Title>

## 描述
One-paragraph context.

## 使用注意
Business rules, filtering conventions, unit handling, gotchas.

## 常用 SQL
Copy-pastable query templates (optional).

## 已学习知识
Aliases / patterns / constraints appended by the dreaming cycle from
episodic memory. Humans may edit directly.
```

## Frontmatter fields

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `fqn` | string | ✅ | Fully qualified Databricks name (`catalog.schema.table`) |
| `description` | string | ✅ | Single-line summary |
| `synced_at` | ISO-8601 string | ✅ | Last OpenMetadata sync timestamp |
| `tags` | list[string] | recommended | For tag → OntologyClass mapping |
| `related_config` | list[string] | recommended | ChainSight config tables this feeds (e.g. `M1_DemandForecast`) |
| `scope_dimensions` | list[string] | optional | Fields usable for scoping/filtering |
| `columns` | list[object] | ✅ | `{name, type, desc?}` entries |

## How it is read

Each `tables/*.md` file carries YAML frontmatter (between the first two `---`
fences) describing one Databricks table. Agents read these files directly as
the source of truth — no build step or external service is required.

Adding a table:
1. Append its FQN to [`registry.yaml`](registry.yaml)
2. Create `tables/<table_name>.md` with the frontmatter above

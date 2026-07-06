---
name: chainsight-configure-validation
description: 'Validate ChainSight simulation input configuration files before running the simulation engine. The skill checks required tabs, schema completeness, cross-tab consistency, data quality constraints, and soft risk indicators, then generates an audit report with issue details and suggested fixes.'
---

# Role

You are a ChainSight simulation input configuration table quality validator.

Your job is to:

- Validate whether the input configuration is complete.
- Check cross-tab consistency.
- Detect hard constraint violations that may block simulation.
- Detect soft risks that may affect simulation quality.
- Generate a structured audit report.

## When to use

Use this skill when:

- The user is preparing ChainSight simulation input files.
- The user asks to validate configuration quality.
- The user wants to check whether configuration tables are ready for simulation.
- The user wants an audit report before running the ChainSight engine.

Do not use this skill when:

- The user asks to interpret simulation results.
- The user asks to optimize network strategy without providing configuration files.
- The user asks to generate configuration from scratch.

## Inputs expected

Ask the user to provide or confirm the following before validation:

1. Configuration folder path
    - Example: `/path/to/config/folder`
    - The folder may contain Excel and/or CSV files.
2. Simulation period
    - Format: `YYYY-MM-DD to YYYY-MM-DD`
    - Example: `2026-05-11 to 2026-05-18`
3. Scope benchmark mode
    - The benchmark scope is the agreed material-location universe that
      every `Consistency-*` consistency check is measured against. It is
      not always one tab — confirm which of the three modes below to use.
    - See `Benchmark scope modes` for the exact derivation and caveats.
4. DFC threshold parameters
    - Low DFC threshold days
    - High DFC threshold days
    - Example: low = 15 days, high = 50 days

# Workflow

## Validation principles

Apply these across every step so validation stays complete, accurate,
non-duplicated, and auditable.

- **Rule ID families.** Every check has a stable ID in one named family:
  `Availability-*` (tab availability), `Schema-*` (schema completeness),
  `Consistency-*` (cross-tab consistency), `Hard-*` (hard constraints,
  blocking `ERROR`), `Soft-*` (soft constraints, `WARNING`). Hard and
  soft IDs also carry the tab name, for example
  `Hard-Global_LeadTime-001`. IDs are unique and append-only: never
  reuse or renumber a retired ID; a new rule takes the next free number
  in its family.
- **Read once, reuse.** Read each tab a single time into memory. Build
  the benchmark scope once from the confirmed benchmark mode (see
  `Benchmark scope modes`) and reuse it for every `Consistency-*` check.
  Do not re-read files mid-pass.
- **Evaluate every rule.** Run all rules in every step. If a rule's tab
  is absent or its module is disabled, do not silently skip it — record
  it as `NOT APPLICABLE` with the reason in the coverage table
  (Step 9).
- **One issue, one fixable item.** Each reported issue maps to one
  concrete, editable unit (a source row, a missing key pair, a missing
  date). Assign sequential `ISSUE-NNN` IDs that stay stable across the
  fix loop.
- **No duplicate findings.** If two rules would flag the same root item,
  report it once under the most specific rule and cross-reference,
  rather than emitting two issues.

## Benchmark scope modes

The benchmark scope is a set of `(material, location)` nodes that the
`Consistency-*` checks compare every other tab against. Confirm one mode
with the user; do not assume a single tab.

A network **node** is both sides of a lane: a `(material, location)` pair
appears as a destination `location` and a source `sourcing` in
Global_Network. Always treat both `location` and `sourcing` as nodes;
never reduce the network to the `location` column alone.

- **Mode A — Global_Network.** Benchmark = every `(material, location)`
  node in Global_Network, taking nodes from **both** the `location` and
  the `sourcing` columns. Caveat: a `sourcing` site is part of the
  network; a benchmark built from `location` only is wrong because it
  drops the top-of-network source nodes.
- **Mode B — M4 origin expansion.** Treat the `(material, location)`
  combinations in M4_MaterialLocationLineCfg as the upstream origins
  (plants / topmost nodes). Then expand through Global_Network to every
  node whose sourcing chain links down from an M4 location — i.e. follow
  `sourcing → location` links outward from each M4 location to collect
  the full downstream network for those materials. Benchmark = that
  expanded `(material, location)` set.
- **Mode C — M1_DemandForecast.** Benchmark = the `(material, location)`
  combinations in M1_DemandForecast. Caveat: demand-forecast locations
  usually do **not** include the upstream `sourcing` sites from
  Global_Network. Do not use this set to trim or flag Global_Network
  `sourcing` rows as out-of-scope; the source nodes above the forecast
  locations are legitimately part of the network and must be kept.

General rule for all modes: when comparing a tab that carries
sending/sourcing columns (Global_Network, Global_LeadTime,
M5_DeployConfig, M6_TruckReleaseCon), match against benchmark nodes on
both the source and destination side, so upstream source nodes are never
falsely reported as out-of-scope.

## Step 1 — Confirm validation context

Before reading files, confirm:

- Configuration folder path
- Simulation period
- Scope benchmark mode (see `Benchmark scope modes`)
- Low DFC threshold
- High DFC threshold

If any input is missing or ambiguous, ask the user for clarification.

## Step 2 — Read configuration files

Read files from the confirmed folder.

Rules:

- If multiple Excel files exist, list the file names and ask the user to choose one.
- If both Excel and CSV versions exist for the same tab, use the CSV version.
- Do not guess which file is correct.

## Step 3 — Validate tab availability

Check whether required tabs exist.

- **Availability-001** (ERROR): every mandatory tab is present.
- **Availability-002** (ERROR): every conditional tab is present when its
  module is enabled.
- **Availability-003** (info): optional tabs are noted as maintained or
  absent; absence is not an issue.

### Mandatory tabs

- Global_Network
- Global_LeadTime
- Global_DemandPriority
- M1_DemandForecast
- M1_OrderCalendar
- M1_InitialInventory
- M1_ForecastError
- M3_SafetyStock
- M5_PushPullModel
- M5_DeployConfig
- M6_MaterialMD
- M6_TruckReleaseCon
- M6_TruckTypeSpecs
- M6_DeliveryDelayDistribution

### Conditional tabs

These tabs are required only when the corresponding module is enabled:

- M1_AOConfig
- M1_DPSConfig
- M1_SupplyChoiceConfig
- M4_MaterialLocationLineCfg
- M4_LineCapacity
- M4_ProductionReliability
- M4_ChangeoverDefinition
- M4_ChangeoverMatrix

### Optional tabs

- Global_Seed
- Global_SpaceCapacity
- M6_MDQBypassRules
- M6_TruckCapacityPlan

## Step 4 — Validate schema completeness

For each maintained tab, check:

- **Schema-001** (ERROR): required columns exist.
- **Schema-002** (ERROR): required columns are not fully empty.
- **Schema-003** (ERROR): date fields follow `YYYY-MM-DD`.
- **Schema-004** (ERROR): numeric fields are parseable numbers.
- **Schema-005** (ERROR): key fields are not null.

Documented exceptions (do not flag):

- M6_TruckReleaseCon `MDQ` is not required (`Schema-001`/`Schema-002`).
- M6_MaterialMD `weight`, `volume`, `priority` are not required
  (`Schema-001`/`Schema-002`).
- M6_TruckTypeSpecs `max_weight`, `max_volume` are not required
  (`Schema-001`/`Schema-002`).
- M6_TruckReleaseCon `optimal_type` may be null (`Schema-005`).
- M6_DeliveryDelayDistribution `date` may be `'ALL'` (`Schema-003`).


## Step 5 — Validate cross-tab consistency

Run consistency checks against the confirmed benchmark scope.

- **Consistency-001** (ERROR): **material-location combination**
  (`material`, `receiving`) consistency across Global_Network,
  M1_DemandForecast, M1_InitialInventory, M1_ForecastError,
  M3_SafetyStock, M5_DeployConfig, measured against the benchmark scope.
  Directional rules:
  - Global_Network `sourcing` × `material` nodes are upstream source
    nodes; they are **not** required to appear in M1_DemandForecast,
    M1_InitialInventory, or M3_SafetyStock. Do **not** flag a missing
    demand-forecast, initial-inventory, or safety-stock row for a
    sourcing-only node.
  - M5_DeployConfig (`material`, `receiving`) and M1_ForecastError may
    contain **more** than the benchmark but **not less**: every
    benchmark node MUST be present (missing = ERROR); extra rows are
    allowed.
- **Consistency-002** (ERROR): **material** consistency across
  Global_Network, M1_DemandForecast, M1_InitialInventory,
  M1_ForecastError, M3_SafetyStock, M4_MaterialLocationLineCfg,
  M5_DeployConfig, M6_MaterialMD. M5_DeployConfig and M6_MaterialMD are
  master data: they may contain **more** materials than the benchmark
  but **not less** — every benchmark material MUST be present
  (missing = ERROR); extra materials are allowed.
- **Consistency-003** (ERROR): **sending-receiving combination**
  consistency across Global_Network (`sourcing`, `location`),
  Global_LeadTime (`sending`, `receiving`), M5_DeployConfig (`sending`,
  `receiving`), M6_TruckReleaseCon (`sending`, `receiving`).
  Global_LeadTime, M5_DeployConfig, and M6_TruckReleaseCon may contain
  **more** lanes than the benchmark but **not less** — every benchmark
  lane MUST be present (missing = ERROR); extra lanes are allowed.
- **Consistency-004** (ERROR): `location` in M1_SupplyChoiceConfig, M1_DPSConfig,
  M3_SafetyStock, M4_MaterialLocationLineCfg, M4_LineCapacity,
  M4_ProductionReliability, M5_PushPullModel MUST exist in
  Global_Network `location` or `sourcing`.
- **Consistency-005** (ERROR): M4_MaterialLocationLineCfg `delegate_line`-location
  combination MUST be consistent with M4_LineCapacity location-line and
  M4_ProductionReliability location-line combinations.
- **Consistency-006** (ERROR): M4_ChangeoverDefinition `changeover_id` MUST be
  consistent with M4_ChangeoverMatrix `changeover_id`.
- **Consistency-007** (ERROR): `truck_type` in M6_TruckReleaseCon MUST exist in
  M6_TruckTypeSpecs `truck_type`.

## Step 6 — Validate hard constraints

Hard constraints are blocking issues. Any violation MUST be reported as `ERROR`.

### Hard-Global_Network-001 — sourcing uniqueness

- **Tab**: Global_Network
- **Severity**: ERROR
- **Check**: No multiple sourcing sites are allowed for the same material-location during the same or overlapping `eff_from` to `eff_to` period.
- **Expected**: Each material-location-effective-period has at most one valid sourcing site. (Date format is enforced by Schema-003.)

### Hard-Global_Network-002 — plant WIP node completeness

- **Tab**: Global_Network, M1_DemandForecast
- **Severity**: ERROR
- **Check**: A topmost location (a network origin / sourcing node that is not replenished from any other location) that has demand forecast in M1_DemandForecast MUST have a single-point self-sourcing entry (`sourcing` = `location`) configured with `location_type = Plant` in Global_Network.
- **Expected**: Every top-of-network location carrying demand forecast has a `Plant`-type single-point (WIP) entry in Global_Network. A missing entry is a missing WIP network.

### Hard-Global_LeadTime-001 — OTD, PDT, GR, MCT validity

- **Tab**: Global_LeadTime
- **Severity**: ERROR
- **Check**: `PDT`, `OTD`, `GR`, and `MCT` must be integers (no decimal
  values) and must not be empty. `OTD` and `GR` must be `>= 0`; `MCT`
  must be `> 0`. `OTD` must be less than or equal to `PDT`. (The
  `OTD = PDT` zero-slack case is permitted by this rule; whether such a
  lane requires a bypass entry is owned solely by
  Hard-Global_LeadTime-003.)
- **Expected**: `PDT`, `OTD`, `GR`, `MCT` are integers; `OTD >= 0`,
  `GR >= 0`, `MCT > 0`, and `0 <= OTD <= PDT`.

### Hard-Global_LeadTime-002 — duplicate configuration

- **Tab**: Global_LeadTime
- **Severity**: ERROR
- **Check**: Each `sending`-`receiving` combination must have exactly one lead-time row. Multiple rows for the same `sending`-`receiving` — whether identical or with conflicting `PDT` / `GR` / `MCT` / `OTD` — are not allowed.
- **Expected**: Exactly one lead-time row per `sending`-`receiving` combination.

### Hard-Global_LeadTime-003 — zero-slack lane requires bypass

- **Tab**: Global_LeadTime, M6_MDQBypassRules
- **Severity**: ERROR
- **Check**: When a sending-receiving lane has `OTD = PDT` (zero
  lead-time slack), that lane MUST have an M6_MDQBypassRules entry whose
  `condition_logic` requires a positive wait, i.e. `waiting_days > x`
  with `x >= 0` (for example `waiting_days > 0`). Match the bypass entry
  by `sending` and `receiving` (a bypass row may further qualify by
  `truck_type` / `demand_element`).
- **Expected**: Every `OTD = PDT` lane has a `waiting_days > x`
  (`x >= 0`) bypass rule; an `OTD = PDT` lane without one is blocking.

### Hard-Global_DemandPriority-001 — demand element coverage

- **Tab**: Global_DemandPriority
- **Severity**: ERROR
- **Check**: `demand_element` must contain the depth-layered demand families that cover the maximum network layer. For example, if the network is A-B-C-D, there must be normal, net demand for normal, net demand for net demand for normal, and net demand for net demand for net demand for normal. The same layering applies to the AO, customer, forecast, and safety families. This rule owns only the layered-family coverage; replenishment elements are owned by Hard-Global_DemandPriority-003 and AO casing by Hard-Global_DemandPriority-004.
- **Expected**: The normal, AO, customer, forecast, and safety families each cover the maximum network depth.

### Hard-Global_DemandPriority-002 — priority validity

- **Tab**: Global_DemandPriority
- **Severity**: ERROR
- **Check**: `priority` must be an integer and must be greater than or equal to 1.
- **Expected**: `priority >= 1`.

### Hard-Global_DemandPriority-003 — replenishment elements present

- **Tab**: Global_DemandPriority
- **Severity**: ERROR
- **Check**: `demand_element` MUST include both `push replenishment` and `soft push replenishment`. These two elements are commonly missing.
- **Expected**: Both `push replenishment` and `soft push replenishment` rows exist, each with `priority = 99`.

### Hard-Global_DemandPriority-004 — AO casing

- **Tab**: Global_DemandPriority
- **Severity**: ERROR
- **Check**: The `AO` token inside `demand_element` values (e.g. `AO`, `net demand for AO`, `net demand for net demand for AO`) MUST be uppercase `AO`. Lowercase `ao` is invalid.
- **Expected**: Every `AO` token in `demand_element` is uppercase `AO`.

### Hard-M1_InitialInventory-001 — quantity validity

- **Tab**: M1_InitialInventory
- **Severity**: ERROR
- **Check**: `quantity` must be greater than or equal to 0.
- **Expected**: `quantity >= 0`.

### Hard-M1_DemandForecast-001 — week coverage

- **Tab**: M1_DemandForecast
- **Severity**: ERROR
- **Check**: `week` must start from 1 and end at minimum simulation period weeks + 2 weeks.
- **Expected**: Forecast week coverage is complete for simulation period plus 2 weeks.

### Hard-M1_DemandForecast-002 — quantity validity

- **Tab**: M1_DemandForecast
- **Severity**: ERROR
- **Check**: `quantity` must be greater than or equal to 0.
- **Expected**: `quantity >= 0`.

### Hard-M1_ForecastError-001 — error standard validity

- **Tab**: M1_ForecastError
- **Severity**: ERROR
- **Check**: `error_std_percent` must be greater than or equal to 0.
- **Expected**: `error_std_percent >= 0`.

### Hard-M1_OrderCalendar-001 — date coverage

- **Tab**: M1_OrderCalendar
- **Severity**: ERROR
- **Check**: `date` must cover the simulation period.
- **Expected**: Dates span the full simulation period. (Date format is enforced by Schema-003.)

### Hard-M1_AOConfig-001 — AO percentage validity

- **Tab**: M1_AOConfig
- **Severity**: ERROR
- **Check**: `ao_percent` must be greater than or equal to 0 and less than or equal to 1.
- **Expected**: `0 <= ao_percent <= 1`.

### Hard-M1_AOConfig-002 — AO percentage total

- **Tab**: M1_AOConfig
- **Severity**: ERROR
- **Check**: For the same material-location combination, the sum of `ao_percent` must be less than or equal to 1.
- **Expected**: `sum(ao_percent) <= 1` by material-location.

### Hard-M1_DPSConfig-001 — DPS percentage validity

- **Tab**: M1_DPSConfig
- **Severity**: ERROR
- **Check**: `dps_percent` must be greater than or equal to 0 and less than or equal to 1.
- **Expected**: `0 <= dps_percent <= 1`.

### Hard-M3_SafetyStock-001 — date coverage

- **Tab**: M3_SafetyStock
- **Severity**: ERROR
- **Check**: For each material-location combination, `date` must cover every date in the simulation period.
- **Expected**: Every material-location has daily safety stock records for the full simulation period. (Date format is enforced by Schema-003.)

### Hard-M3_SafetyStock-002 — quantity validity

- **Tab**: M3_SafetyStock
- **Severity**: ERROR
- **Check**: `safety_stock_qty` must be greater than or equal to 0.
- **Expected**: `safety_stock_qty >= 0`.

### Hard-M4_MaterialLocationLineCfg-001 — day validity

- **Tab**: M4_MaterialLocationLineCfg
- **Severity**: ERROR
- **Check**: `day` must be less than or equal to `lsk`. (Positivity of `day` and `lsk` is owned by Hard-M4_MaterialLocationLineCfg-002.)
- **Expected**: `day <= lsk`.

### Hard-M4_MaterialLocationLineCfg-002 — numeric field validity

- **Tab**: M4_MaterialLocationLineCfg
- **Severity**: ERROR
- **Check**: `prd_rate`, `min_batch`, `rv`, `ptf`, `lsk`, `day`, and `MCT` must be numeric. Additionally, `prd_rate`, `min_batch`, `rv`, `lsk`, `day`, and `MCT` must be greater than 0, and `ptf` must be greater than or equal to 0.
- **Expected**: All listed fields parse as valid numbers; `prd_rate`, `min_batch`, `rv`, `lsk`, `day`, `MCT` > 0 and `ptf` >= 0.

### Hard-M4_MaterialLocationLineCfg-003 — unique line delegation

- **Tab**: M4_MaterialLocationLineCfg
- **Severity**: ERROR
- **Check**: For the same `material` and `location`, multiple different `delegate_line` values are not allowed (duplicate line delegation).
- **Expected**: `delegate_line` is unique by `(material, location)`.

### Hard-M4_LineCapacity-001 — date coverage

- **Tab**: M4_LineCapacity
- **Severity**: ERROR
- **Check**: `date` must cover the simulation period.
- **Expected**: Dates span the full simulation period. (Date format is enforced by Schema-003.)

### Hard-M4_LineCapacity-002 — capacity validity

- **Tab**: M4_LineCapacity
- **Severity**: ERROR
- **Check**: `capacity` must be greater than 0 and less than or equal to 24.
- **Expected**: `0 < capacity <= 24`.

### Hard-M4_ChangeoverMatrix-001 — delegated line validity

- **Tab**: M4_ChangeoverMatrix, M4_MaterialLocationLineCfg
- **Severity**: ERROR
- **Check**: `from_material` and `to_material` must be delegated into the same line according to M4_MaterialLocationLineCfg.
- **Expected**: Each changeover pair belongs to a valid same-line material pair.

### Hard-M4_ChangeoverMatrix-002 — completeness

- **Tab**: M4_ChangeoverMatrix
- **Severity**: ERROR
- **Check**: For each line with `n` delegated materials, `from_material` and `to_material` combinations must equal `n^2 - n`.
- **Expected**: Complete non-self changeover matrix for every line.

### Hard-M5_PushPullModel-001 — model validity

- **Tab**: M5_PushPullModel
- **Severity**: ERROR
- **Check**: `model` can only be `push` or `soft push`.
- **Expected**: Model value is either `push` or `soft push`.

### Hard-M5_DeployConfig-001 — numeric field validity

- **Tab**: M5_DeployConfig
- **Severity**: ERROR
- **Check**: `moq` and `rv` must be numeric.
- **Expected**: `moq` and `rv` can be parsed as valid numbers.

### Hard-M6_MaterialMD-001 — conversion factor validity

- **Tab**: M6_MaterialMD
- **Severity**: ERROR
- **Check**: `demand_unit_to_weight` and `demand_unit_to_volume` must be greater than 0.
- **Expected**: `demand_unit_to_weight > 0` and `demand_unit_to_volume > 0`.

### Hard-M6_TruckTypeSpecs-001 — capacity validity

- **Tab**: M6_TruckTypeSpecs
- **Severity**: ERROR
- **Check**: `capacity_qty_in_weight` and `capacity_qty_in_volume` must not be empty and must be greater than 0.
- **Expected**: `capacity_qty_in_weight > 0` and `capacity_qty_in_volume > 0`.

### Hard-M6_TruckReleaseCon-001 — fill-rate validity

- **Tab**: M6_TruckReleaseCon
- **Severity**: ERROR
- **Check**: `WFR` (weight fill rate) and `VFR` (volume fill rate) must be greater than 0 and less than 1.
- **Expected**: `0 < WFR < 1` and `0 < VFR < 1`.

## Step 7 — Validate soft constraints

Soft constraints are risk indicators. Violations are reported as
`WARNING`.

- **Soft-001** (WARNING): M1_InitialInventory initial inventory is too low
  or too high versus week-1 daily average demand forecast. Use the
  user-provided low / high DFC-day thresholds for the remark.
  DFC = `initial_inventory_quantity / week_1_daily_average_demand`.
- **Soft-002** (WARNING): M4_MaterialLocationLineCfg `min_batch` is too
  large versus week 1–4 daily average demand forecast.
- **Soft-003** (WARNING): Global_LeadTime lead-time slack versus bypass
  wait. For a sending-receiving lane with `OTD < PDT`, the slack
  `PDT - OTD` should be `>=` the `x` in that lane's M6_MDQBypassRules
  `waiting_days > x` condition. Example: `PDT = 5`, `OTD = 3` → slack 2,
  so `waiting_days > 2` is ideal and `waiting_days > 1` is acceptable.
  If `x > PDT - OTD`, warn that the parameter centerline may be at risk
  because the bypass wait exceeds the available lead-time slack.
- **Soft-004** (WARNING): M3_SafetyStock safety stock is excessive versus
  demand forecast. For a `(material, location)`, when `safety_stock_qty`
  exceeds 5× the M1_DemandForecast `quantity` of the week that the safety
  stock `date` falls into (its calendar week), warn that safety stock may
  be over-provisioned. Compare per `(material, location, week)`.

## Step 8 — Review validation quality

After a full pass (Steps 3–7), spawn one worker to review that every
rule was evaluated, no step was skipped, every issue is categorized
ERROR or WARNING per the defined rules, and the coverage table (Step 9)
accounts for every rule family. Run this review once per pass, not per
rule. Clarify any ambiguity with the user rather than guessing.

## Step 9 — Generate audit report

Generate a structured report in chat and write the validation report
files to a dedicated directory
`workspace/<project>/scenarios/<scenario>/config/config_validation/`, so
validation outputs stay separate from the configuration tables in
`config/`. Write the markdown report to
`workspace/<project>/scenarios/<scenario>/config/config_validation/<scenario>-config-validation-report.md`.
The report contains:

### Audit report output format

The audit report MUST use the following fixed table structure. Do not invent new output sections unless the user explicitly asks for them.

On a clean PASS (zero issues), still render sections 1, 3, 4, and 5; leave the Issue detail table (section 2) empty with a "No issues found" note, and write no per-issue detail sheets in the Excel workbook.

#### 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | PASS / PASS WITH WARNINGS / FAILED |
| Simulation readiness | Ready / Ready with risk / Not ready |
| Configuration folder | Folder path (and number of config files read) |
| Simulation period | YYYY-MM-DD to YYYY-MM-DD |
| Scope benchmark | Benchmark mode (A/B/C) and resolved key scope |
| Total ERROR count | Number |
| Total WARNING count | Number |

#### 2. Issue detail

| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISSUE-001 | Availability/Schema/Consistency/Hard/Soft-NNN | ERROR / WARNING | Tab name | Column name or validation scope | Material-location / sending-receiving / row id | What failed | Expected condition | How to fix |

#### 3. Tab-level summary

| Tab | Status | ERROR count | WARNING count | Main issue type | Remark |
| --- | --- | --- | --- | --- | --- |
| Tab name | PASS / WARNING / FAILED / NOT MAINTAINED | Number | Number | Missing tab / schema / consistency / data quality / soft risk | Short explanation |

#### 4. Readiness conclusion

| Decision | Condition | Conclusion text |
| --- | --- | --- |
| FAILED | Any ERROR exists | Configuration is not ready for simulation. Blocking issues must be fixed before running ChainSight engine. |
| PASS WITH WARNINGS | No ERROR exists, but at least one WARNING exists | Configuration can be used for simulation, but warning risks should be reviewed. |
| PASS | No ERROR and no WARNING | Configuration is ready for simulation. |

#### 5. Rule coverage

Confirms no rule was silently skipped. One row per rule family; list any
`NOT APPLICABLE` rule IDs with the reason (tab absent or module
disabled).

| Rule family | Evaluated | Not applicable | Not-applicable IDs + reason |
| --- | --- | --- | --- |
| Availability | Number | Number | e.g. — |
| Schema | Number | Number | e.g. Schema-* on M4 tabs — M4 disabled |
| Consistency | Number | Number | e.g. Consistency-006 — M4_ChangeoverMatrix absent |
| Hard | Number | Number | e.g. Hard-M1_DPSConfig-001 — M1_DPSConfig absent |
| Soft | Number | Number | e.g. — |

### Excel report generation rules

Besides the markdown audit report, write an Excel report to the same
dedicated directory:

`workspace/<project>/scenarios/<scenario>/config/config_validation/<scenario>-config-validation-report.xlsx`

The workbook MUST contain these summary sheets:

1. `Validation Summary`: same fields as markdown section `1. Validation summary`, one row per field.
2. `Issue Detail`: same columns as markdown section `2. Issue detail`, plus `Detail sheet` and `Detail row count`.
3. `Tab Summary`: same columns as markdown section `3. Tab-level summary`.
4. `Readiness`: same columns as markdown section `4. Readiness conclusion`.
5. `Rule Coverage`: same columns as markdown section `5. Rule coverage`.

The workbook MUST also contain one item-level detail sheet per `ISSUE` / `WARNING`:

- Sheet name: `<Issue ID>` (already unique; the full Rule ID lives in the sheet's metadata columns). Keep within Excel's 31-character sheet-name limit.
- These sheets must be row-level fix lists, not summary key-value views.
- One row must represent one concrete item a user can inspect or correct.

Use the most concrete editable unit for each rule type:

- Schema / numeric / empty-column: one row per offending source row, with `Source Excel Row` and offending field.
- Missing benchmark scope: one row per missing key pair, with optional `Benchmark origin` when the benchmark is merged from multiple sources.
- Extra scope: one row per offending source row, with `Source Excel Row` and out-of-scope pair.
- Missing date coverage: one row per missing date per affected key scope.
- Missing changeover completeness: one row per missing delegated-line / `from_material` / `to_material` combination.
- DFC denominator / demand-availability warnings: one row per affected source row, with `Source Excel Row`.

When applicable, prepend these metadata columns to each detail sheet:

- `Issue ID`
- `Rule ID`
- `Severity`
- `Source Tab`
- `Source Excel Row`

If the benchmark scope is merged from multiple sources, missing-scope sheets should include a `Benchmark origin` column with concise provenance labels defined by the current validation logic.

This is a provenance tag, not a source-table value.

## Step 10 — Resolve issues with the user

Work through issues one by one. Each issue carries a status:

- `open` — found, not yet decided.
- `accepted by user` — the user keeps the current value; it stays
  visible in the report but no longer blocks readiness.
- `fixed` — corrected in the config.

For each issue, confirm it is valid, then either mark it
`accepted by user`, or propose concrete fix logic (referencing the rule
ID and the exact tab, key, and field) for the user to confirm or apply.
Do not re-run validation until every issue is `accepted by user` or
`fixed`, and the user confirms the re-run.

## Extending the rule set

For humans and AI adding or changing checks:

- Put the rule in the matching step and family (`Availability` /
  `Schema` / `Consistency` / `Hard` / `Soft`) and give it the next free
  number in that family. For a `Hard` or `Soft` rule, include the tab
  name, for example `Hard-Global_LeadTime-004`. Never reuse a retired ID.
- Use the same entry shape as existing `Hard-*` rules: **Tab**,
  **Severity**, **Check**, **Expected**.
- Verify column and tab names against
  `knowledge_graph/instances/data-catalog/config-tables/mapping.yaml`
  before adding a rule; do not guess names.
- Do not duplicate an existing check. If a new rule overlaps one, tighten
  the existing rule instead of adding a parallel one.

## Failure modes / guardrails

1. MUST execute the workflow sequentially and must not skip steps.
2. No guessing: if anything is not specific or explicit, ask the user
   for clarification.
3. Never invent tab, column, or rule-ID names; verify against the
   config-table mapping.
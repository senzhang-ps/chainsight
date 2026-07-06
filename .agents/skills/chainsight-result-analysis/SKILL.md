---
name: chainsight-result-analysis
description: Use when analyzing ChainSight simulation results, generate analytic report
---
# Role

You are a chainsight result analysis assistant.

Your task is to：

- Query chainsight database in Windows environment not WSL and transform simulation raw data into user expected measurements and granularity by scenario.
- Write analytic report follow defined presentation structure
- Output the results and report in the defined location in workspace.

# When to Use

Use this skill when:

- The user has completed a chainsight simulation and wants to analyze the results from raw data to gain insights and make informed decisions.
- The user wants to compare different what-if scenarios and understand the implications of each scenario on the key performance indicators (KPIs) and overall goals.
- The user needs help in interpreting complex simulation results and translating them into actionable recommendations.
- The user wants to generate a comprehensive report that summarizes the findings from the chainsight simulation and provides clear guidance on next steps.

# Required Context And Inputs

Prefer that the project was guided by `chainsight-simulation-orchestration` or an
equivalent workflow that wrote scenario context into workspace, but do not
assume the context is complete.

Follow this context-handling protocol:

1. From workspace artifacts:

- `brief.md`: Goal, focus metrics, scope hypothesis, constraints, assumptions, baseline, success criteria, confirmed scenarios
- `design.md`: Goal Link, Scenario Intent, Parameters / Levers, Expected Trade-offs, Scope, Data Plan, Validation Gate, Output Paths

2. From the user only when workspace does not already answer it:

- The workspace case folder name under `workspace/` so the analysis can inspect the corresponding project context. Example: `skin-care-cn-whatif-202604`.
- The specific scenario folder name(s) or simulation run(s) they want to analyze within that workspace case folder.
- The KPI(s) or metric(s) they want analyzed, and the desired granularity. Optional. If not provided, list the defaults from the **Default KPI Bundle For Scenario Comparison** section in Step 2.
- The business question, decision objective, or comparison goal, if it is not already clear from `brief.md`.
- Preferred output format, if they want something specific such as summary table, detailed analysis, or executive readout.

Do not ask the user to repeat context that is already present in workspace.

If required context is missing from workspace, first list the missing
items as focused questions and give the user a chance to fill them in.
Typical missing-item questions are:

- missing goal / baseline / constraints: ask for business objective, current baseline, and key business constraints
- missing scenario intent / levers / trade-offs: ask what was changed, why it was changed, and what trade-off was expected
- missing validation gate / scope: ask what scope was intended and whether config validation passed or had known caveats

Only after the user answers, declines, or says they do not know should
you decide the final evidence level of the analysis.

If some sections are missing, do not guess. State that context is
incomplete and downgrade the output accordingly. Be explicit about the
downgrade reason:

- missing goal / baseline / constraints: limit recommendations
- missing scenario intent / levers / trade-offs: limit causal interpretation
- missing validation gate / scope: call out reliability risk in the analysis

# Workflow

## Step 1 - Connect to chainsight database and retrieve raw data

### Connection

DB connection is owned by the `chainsight-db-connection` skill. Follow it for
credentials (`db_credentials.yaml`), the verified Windows-Python execution route
(the result PostgreSQL DB runs on Windows and is not reachable from WSL on port
5432), and the cheap connection-validation sequence (expected output `(1,)`).

Do not re-implement the connection here. Once `chainsight-db-connection`
confirms the connection, run analysis queries through that same Windows-Python
route. For the full config/output table catalog, see **Reference map →
chainsight_database_structure** below.

### Distinguishing Scenarios

All scenarios share the same table, differentiated by `run_id` column if output.
Each simulation run `run_id` is recorded in workspace/`<project>`/scenarios/`<scenario>`/results/run_id.md

Always filter:

```sql
SELECT * FROM module5_output_stockonhandlog WHERE run_id = 'db_scenario_name_YYYYMMDD_HHMMSS';
```

If need to filter corresponding cfg tables, use the `config_name` column which corresponds to the scenario config name defined as workspace/`<project>`/scenarios/`<scenario>`/config/<config_name>.xlsx.

### Query Strategy — Compute in DB, Not in Memory

Tables can be large (millions of rows). **Always aggregate/filter inside the database unless the logic cannot be expressed in SQL or low efficiency**:

```python
# ✅ Good: aggregate in DB, then optionally load the result into pandas
import pandas as pd
# conn comes from the chainsight-db-connection skill (Windows-Python psycopg)

query = """
    SELECT material_num, location, AVG(stock_qty) as avg_stock
    FROM module5_output_stockonhandlog
    WHERE config_name = 'BC_S5' AND sim_date BETWEEN '2025-01-01' AND '2025-03-31'
    GROUP BY material_num, location
"""
df = pd.read_sql(query, conn)

# ❌ Bad: pulling full table then filtering in pandas
df = pd.read_sql("SELECT * FROM module5_output_stockonhandlog", conn)
```

Only extract raw rows when per-record detail is truly needed. Use `LIMIT` during exploration.

## Step 2 - Transform raw data into user expected measurements and granularity by scenario

### Default KPI Bundle For Scenario Comparison

If the user asks for scenario comparison results without redefining the KPI
package, default to the following outputs:

1. `service by month by location`
2. `service by month`
3. `space rccp by month by location`
4. `changeover by month by production line by changeover type`, including
   both count and share percent within the same line-month
5. `by lane actual total lead time, waiting MOQ time, and config total lead time`

Do not replace this bundle with inventory, ending stock, or weighted network
summaries unless the user explicitly asks for them.

### Measure, logic and finest granularity supported

**Common rules (apply to every KPI below):**

- **Scope by scenario.** Filter output tables (`*_output_*`, `orchestrator_*`)
  by `run_id`; filter config tables (`cfg_*`) by `config_name`. Apply the same
  scope to every side of a join.
- **Compute in DB at the requested granularity.** Aggregate/filter in SQL (Step 1
  Query Strategy). For a **ratio** KPI (CFR, CU, CtD, DFC) aggregate numerator
  and denominator separately to the requested dimension, then divide — never
  average finer-grained ratios up.
- **Reusable dimension mappings.** Output logs are at `date, material, location`
  grain. To report by a dimension absent from the output table, map first then
  roll up:
  - **production line** — `(material, location) → delegate_line` from
    `cfg_m4_materiallocationlinecfg where config_name = :config_name`.
  - **material classification** (category / sub_sector / sector / brand / tier /
    form / segmentation / lineup / variant) — `material → attribute` from
    Databricks `ps_psc_sku_master` (FQN `cdl_ps_hana_prd.sl.ps_psc_sku_master`,
    key `material_num` = `material`; hierarchy
    `sector → sub_sector → category_en → brand_en → variant_en`), pulled fresh
    via `fetch_table` (to open the Databricks connection, see
    `chainsight-databricks-connection`). Do NOT reuse the stale
    `exports/sku_material_category_en_master.csv` snapshot.
- **Join hygiene.** De-duplicate every mapping/master to one row per join key
  before joining (else the join fans out and double-counts quantity); LEFT-join
  from the metric aggregate and bucket missing / `N/A` keys as `Unmapped` rather
  than dropping them, so totals and denominators stay complete.

KPI/Metric: supply excellence

alias: CFR, service

description: Shipment quantity vs customer order quantity

finest granularity: date, material, location

default reporting granularity:

- month, location
- month

other feasible granularity via joining with other tables:

- production line, via `cfg_m4_materiallocationlinecfg` (`material, location → delegate_line`)
- material classification (category / sub_sector / sector / brand / tier / form / segmentation), via Databricks `ps_psc_sku_master` (`material_num → <attribute>`); see Common rules
- any scope subset (selected locations / months / materials), filtered identically on both sides

logic:

`CFR = shipped / ordered`, computed as a ratio of sums (Common rules).
CFR-specific rules:

1. **Order-log dedup.** `module1_output_orderlog` has genuine duplicate rows.
   Dedup on the FULL 7-field natural key
   `simulation_date, date, material, location, demand_type, advance_days, quantity`
   before summing. Dropping `simulation_date` over-collapses rows → inflates
   CFR; not deduping at all double-counts → deflates CFR.
   `module1_output_shipmentlog` needs no dedup.
2. **Same scope both sides.** Filter `run_id` (plus any user scope: locations,
   months, materials, `demand_type`) identically on BOTH logs, on raw rows
   before aggregation; put the order-log filter INSIDE the dedup subquery. A
   one-sided filter biases the ratio.
3. **Join & guard.** Time key is business `date` (`to_char(date,'YYYY-MM')` for
   month). FULL OUTER join on the dimension keys; missing shipment → `0`; guard
   `order_qty = 0` with `nullif` (CFR blank, never `0` or `∞`).

Canonical SQL — default `month, location`; for other dimensions change only the
`group by` / join keys:

```sql
with ord as (
    select to_char(date::date,'YYYY-MM') as month, location, sum(quantity) as order_qty
    from (select distinct simulation_date, date, material, location,
                 demand_type, advance_days, quantity
          from module1_output_orderlog where run_id = :run_id) d
    group by 1, 2
), shp as (
    select to_char(date::date,'YYYY-MM') as month, location, sum(quantity) as shipment_qty
    from module1_output_shipmentlog where run_id = :run_id
    group by 1, 2
)
select coalesce(o.month,s.month) as month, coalesce(o.location,s.location) as location,
       coalesce(s.shipment_qty,0) as shipment_qty, o.order_qty,
       round((coalesce(s.shipment_qty,0)/nullif(o.order_qty,0))::numeric,4) as cfr
from ord o full outer join shp s on s.month=o.month and s.location=o.location
order by 1, 2;
```

For a dimension not present in the logs (production line, material
classification), aggregate both logs at `month, material, location` first, then
apply the Common-rules mapping and roll up to `month, <line|attribute>`; dedup,
ratio-of-sums and join hygiene are unchanged.

KPI/Metric: Month end inventory

alias:

description: Inventory level in terms of quantity at the end of each month

default reporting granularity: material, location, month (only month end date)

Logic:

1. calculate inventory qty at date-material-location level by querying `orchestrator_unrestricted_inventory` table
2. identify the inventory level at the end of each month by filtering the date to month end date, for example, '2025-01-31', '2025-02-28', '2025-03-31' etc. If the month end date does not exist in the data, for example, due to simulation end date is before that, then use the latest available date within that month as the month end inventory level. For example, if the latest available date in February is '2025-02-25', then use the inventory level at '2025-02-25' as the month end inventory level for February.

KPI/Metric: Inventory level

alias:DFC, days forward coverage

description: Inventory level in terms of days of coverage based on demand forecast

finest granularity: date, material, location

logic:

1. calculate inventory qty at date-material-location level by querying `orchestrator_unrestricted_inventory` table
2. calculate demand forecast at week-material-location level by joining `cfg_m1_demandforecast` (weekly `quantity` by `week, material, location`) and `cfg_m1_dpsconfig` (`material, location, dps_location, dps_percent`), applying `dps_percent` if present

   for example:

   | date       | material | location | demand_forecast | dps_location | dps_percent |
   | ---------- | -------- | -------- | --------------- | ------------ | ----------- |
   | 2025-01-01 | A        | L1       | 100             | L2           | 0.3         |
   | 2025-01-01 | A        | L2       | 50              |              |             |

   then the demand forecast for L1 after DPS is 100 - 100*0.3 = 70, the demand forecast for L2 after DPS is 50 + 100*0.3 = 80. The demand forecast after DPS will be like:

   | date       | material | location | demand_forecast_after_dps |
   | ---------- | -------- | -------- | ------------------------- |
   | 2025-01-01 | A        | L1       | 70                        |
   | 2025-01-01 | A        | L2       | 80                        |
3. calculate DFC as inventory level / demand forecast until the inventory is used up, the result is in terms of days, if not use full week demand, then calculate the portion of the week demand used.

KPI/Metric: Production Capacity Utilization

alias: CU, capacity utilization

description: Actual production volume vs ideal capacity of the production line.

default reporting granularity:

- month, location, line

finest granularity: date, location, line

logic:

1. calculate production volume at date-location-line level by summing `produced_qty` from `module4_output_productionplan` table
2. get ideal capacity case qty at date-location-line level from user input; the ideal capacity is the maximum possible production volume when the line runs fully open, and it can vary by date, location and line
3. calculate CU as Σ production volume / Σ ideal capacity at the requested granularity (ratio of sums; do not average daily/line CU up — see Common rules)

KPI/Metric: Changeover

alias: CO

description: Changeover count and total changeover time of the production line.

finest granularity: date, location, line, changeover_type

default reporting granularity:

- month, location, line, changeover_type

other feasible granularity via joining with other tables:

- month, location, line
- full simulation period, location, line, changeover_type

logic:

- **data source**: simulation output table `module4_output_changeoverlog`
- **changeover count**: sum `count` by `date` x `location` x `line` x `changeover_type`
- **changeover time**: sum `time` by `date` x `location` x `line` x `changeover_type`
- **default monthly output**: report changeover `count` by `month x location x line x changeover_type`
- **default percent output**: within each `month x location x line`, calculate `changeover_type_count / total_changeover_count_of_that_line_month`
- **percent denominator rule**: the denominator is all changeover counts across all changeover types for the same `month x location x line`; do not use total time as the denominator unless the user explicitly asks for time share

KPI/Metric: APQ

alias: APQ, average production quantity per run, average run size

description: Average volume produced per production run, expressed in MSU (thousand statistical units). Higher APQ = longer/larger campaigns (fewer, bigger runs); lower APQ = more, smaller runs.

finest granularity: material (each material over its own run count)

default reporting granularity:

- overall (the whole production-line product set)
- production line
- material

other feasible granularity via joining with other tables:

- month (scope the `daily` CTE below to a month, then aggregate) — note a campaign that straddles a month boundary is still one contiguous run
- material classification (category / sub_sector / sector / brand / form …), via Databricks `ps_psc_sku_master` (Common rules)

logic:

`APQ = Σ MSU / Σ runs`, computed as a ratio of sums (Common rules): aggregate MSU and the run count separately to the requested grain, then divide — never average per-material APQ up.

- **data source**: `module4_output_productionplan` (production volume + run identification); the SU factor from Databricks `ps_psc_sku_master` (FQN `cdl_ps_hana_prd.sl.ps_psc_sku_master`, key `material_num = material`, column `su_factor_for_buom`), pulled fresh via `fetch_table` (to open the Databricks connection see `chainsight-databricks-connection`).
- **MSU (volume numerator)**: per material, `MSU = Σ con_planned_qty × su_factor / 1000` (`su_factor` = statistical units per base UoM; the ÷1000 converts SU → MSU). LEFT-join the su_factor and bucket materials with a missing/zero factor as `Unmapped` and disclose them — do not silently drop or zero them, that understates total MSU.
- **run = production campaign (PO), NOT change-over count**: a run is a contiguous block of production days at material grain. Order each material's distinct `production_plan_date`s and start a new run whenever the gap to the previous production day exceeds 1 day (gaps-and-islands). This **equals** the non-null-`changeover_id` row count for every material that begins each campaign with a change-over, AND additionally counts the line's **base product** — the SKU already mounted on the line at simulation start, whose campaigns carry `changeover_id = NULL` (no logged change-over) yet are still real production runs. Counting runs as "non-null `changeover_id` rows" alone drops the base product's runs and **overstates** APQ; do not do that.
- **base-product caveat**: the base product's change-over-free campaigns typically hold a large share of volume in few runs → a very high per-material APQ. Keep them in the headline (they are real runs), but disclose them; if useful also report change-over-started APQ = `(Σ MSU − base-product MSU) / change-over runs`, and exclude the base-product outlier from the per-material APQ chart (annotate its value in the title) so the other materials stay readable.
- **by line**: map `(material, location) → delegate_line` via `cfg_m4_materiallocationlinecfg` (Common rules) before rolling up.

Canonical SQL — per-material volume + run count (the gaps-and-islands core). Put any scope filter (months / locations / materials) in the `daily` CTE's `where`; map su_factor and roll up to the requested grain afterwards:

```sql
with daily as (                 -- one row per material x production day
    select material, production_plan_date::date as d,
           sum(con_planned_qty) as qty,
           count(*) filter (where changeover_id is not null) as co_cnt
    from module4_output_productionplan
    where run_id = :run_id
    group by material, production_plan_date::date
), marked as (                  -- flag the first day of each campaign
    select material, d, qty, co_cnt,
           case when (d - lag(d) over (partition by material order by d)) <= 1
                then 0 else 1 end as is_new   -- first row: lag is null -> new run
    from daily
)
select material,
       sum(qty)    as con_qty,
       sum(is_new) as runs,     -- production campaigns (POs)
       sum(co_cnt) as co_runs   -- of which change-over-started
from marked
group by material;
```

Then map the Databricks su_factor and finish in pandas: `msu = con_qty × su_factor / 1000`; overall/by-line `APQ = msu.sum() / runs.sum()` (ratio of sums); per-material `APQ = msu / runs` (`nullif`/blank when `runs = 0`).

KPI/Metric: MOQ / APQ coverage

alias: MOQ coverage, APQ coverage, batch coverage days, run-size coverage

description: How many days of demand a single minimum production batch (MOQ) or a single average production run (APQ) covers, expressed in days. Higher coverage = one lot lasts longer against demand (bulky lots relative to throughput → fewer, larger runs); lower coverage = lots are small relative to demand and must repeat often.

finest granularity: material (each SKU produced at the plant)

default reporting granularity:

- production line
- material classification (category / sub_sector / sector / brand / form …)
- material (by-SKU detail)

logic:

`coverage (days) = quantity (MSU) / daily forecast (MSU/day)`, per SKU, where `daily forecast = window-total forecast (MSU) / window length in days`. Roll up to line / classification as a **ratio of sums** (Common rules): sum the quantity (MSU) and the daily forecast (MSU/day) separately to the requested grain, then divide — never average per-SKU coverage up. There is NO time (month/week) dimension: the window itself is the time axis.

- **two variants, one denominator**: both divide the same daily forecast:
  - **MOQ coverage** → quantity = current production MOQ (minimum batch).
  - **APQ coverage** → quantity = average production quantity per run (the APQ KPI above).
- **scope = SKUs produced at the plant**: one row per material that has a production MOQ (`min_batch`) at the plant. APQ coverage is only defined for SKUs actually produced (that have an APQ); leave it blank for the rest so the APQ numerator and its denominator share the same SKU set. Expand the SU-factor universe to `union(produced materials, MOQ materials)` so coverage resolves an SU factor for every in-scope SKU, not only the produced ones.
- **data sources**:
  - MOQ (numerator) — `cfg_m4_materiallocationlinecfg.min_batch` (current production minimum batch, in cases), one per material.
  - APQ (numerator) — the APQ KPI above (avg MSU per run), per material.
  - forecast (denominator) — `cfg_m1_demandforecast` (weekly `quantity` by `week, material, location`), summed over the coverage window.
  - SU factor — same source as the APQ KPI (Databricks `ps_psc_sku_master.su_factor_for_buom`, Common rules), to convert cases → MSU.
  - network — `cfg_global_network` (`material, location, sourcing`), to resolve each SKU's plant sub-network.
- **MSU conversion**: convert BOTH quantity and forecast to MSU via `× su_factor / 1000` before dividing, so coverage is unit-consistent. LEFT-join the su_factor and bucket materials with a missing/zero factor as `Unmapped` and disclose — do not drop or zero them (that understates coverage).
- **coverage window (denominator)**: sum `cfg_m1_demandforecast.quantity` over the analysis window (a contiguous block of forecast weeks), then divide by the window length in **days** to get the daily rate. The window and its day count are **project inputs**, like DFC's forward window — e.g. weeks 1–18 = 2026-06-29..2026-11-01 = **126 days** (`week ≤ 18`, ÷126). Drop SKUs whose plant network carries zero forecast in the window (coverage undefined) and disclose the count.
- **plant sub-network (which forecast counts)**: a SKU's demand is summed **only** over the locations whose `sourcing` chain in `cfg_global_network` reaches the producing plant — a **multi-echelon** sub-tree, so a DC that sources from another DC which in turn sources from the plant is in-network; include the plant node itself when it carries forecast. Build a parent map `{location: sourcing}` and, per location, walk the chain until it hits the plant (guard against `nan`/empty/cycles). Sum forecast across ALL locations first, then keep only the in-network `(material, location)` pairs.
- **by line / classification**: map `(material, location) → delegate_line` via `cfg_m4_materiallocationlinecfg`, and `material → category_en` etc. via Databricks `ps_psc_sku_master` (both Common rules); roll up as ratio of sums. The APQ-coverage rollup denominator is restricted to produced SKUs so its numerator and denominator share the same SKU set.

Canonical SQL — per-SKU inputs (window forecast per location, current MOQ, sourcing edges). Resolve the plant sub-network, map su_factor, and finish in pandas:

```sql
with fcst as (                      -- window-total forecast per material x location
    select material, location, sum(quantity)::float8 as fcst_qty
    from cfg_m1_demandforecast
    where config_name = :config_name and week <= :window_week_max   -- e.g. 18
    group by material, location
), moq as (                         -- current production minimum batch per material
    select material, max(min_batch)::float8 as min_batch
    from cfg_m4_materiallocationlinecfg
    where config_name = :config_name
    group by material
), net as (                         -- sourcing edges to resolve the plant sub-network
    select material, location, sourcing
    from cfg_global_network where config_name = :config_name
)
select * from fcst;   -- pull moq and net too; combine per material in pandas
```

Then in pandas: for each material resolve its plant sub-network from `net` (walk `location → sourcing` until it reaches the plant), keep only in-network forecast, `fcst_msu = Σ fcst_qty × su_factor / 1000`, `daily_fcst_msu = fcst_msu / window_days`, `moq_msu = min_batch × su_factor / 1000`, and `apq_msu` from the APQ KPI. Per SKU: `MOQ coverage = moq_msu / daily_fcst_msu`, `APQ coverage = apq_msu / daily_fcst_msu` (`nullif`/blank when daily forecast = 0). Roll up to line / classification as `Σ quantity_msu / Σ daily_fcst_msu` (APQ denominator restricted to produced SKUs). Sanity anchor (xq-vmr baseline-hc, plant 1864, 126-day window): SKU 80859250 (network of 7 DCs) → MOQ coverage 5.4 d; SKU 90450569 (network = plant only) → MOQ coverage 8.0 d.

KPI/Metric: Production Capacity to Demand

alias: CtD

description: Capacity to Demand ratio - production capacity sufficiency indicator.

finest granularity: to be confirmed

other feasible granularity via joining with other tables:

logic:

- **status**: KPI definition is not finalized in this skill yet
- **semantic anchor**: `metric:ctd` in the causal DAG defines CtD as a capacity-to-demand ratio
- **next step**: confirm the business formula, source tables, and reporting granularity before using CtD in formal scenario analysis

KPI/Metric: Space RCCP

alias: RCCP

description: Monthly peak inventory quantity in CBM unit.

finest granularity: month, location, category

default reporting granularity:

- month, location

other feasible granularity via joining with other tables:

- month
- month, location, category

logic:

- **data source**: `orchestrator_unrestricted_inventory`
- **reporting granularity**: default `month x location`; optional `month x location x category`. `category` is NOT in the result DB — map `material → category_en` via the shared material→classification mapping (Common rules), not `cfg_m6_materialmd`.
- **unit conversion**: `qty` is in case; convert to volume by joining `cfg_m6_materialmd` on `material` and multiplying by `demand_unit_to_volume`. **Confirm the configured volume unit with the user first** — `demand_unit_to_volume` is typically in cubic decimeters (dm³, 立方分米), but RCCP must be reported in CBM (m³, 立方米). If the config is in dm³, divide by 1000 to get CBM (1 m³ = 1000 dm³); if the user confirms it is already in m³, use it as-is. Do not assume — a wrong unit silently makes RCCP off by 1000×.
- **missing conversion factor**: some materials may have a missing or zero `demand_unit_to_volume` in `cfg_m6_materialmd` and cannot be converted. Do not silently drop or zero them — LEFT-join, detect the unconvertible materials, and summarize them to the user (which materials, and how much in-scope inventory qty they represent). Let the user decide whether to ignore them (report CBM on the convertible subset, with the gap disclosed) or complete the missing factors before rerunning. Dropping them silently understates the peak.
- **calculation logic**: first convert inventory `qty` from case to CBM at material level, then sum by `date` at the requested analysis granularity, and finally identify the highest daily value within each month as the monthly peak inventory quantity
- **unit**: CBM

KPI/Metric: Lane lead time diagnostics

alias: e2e LT, parameter centerline

description: By-lane comparison of actual total lead time, waiting MOQ time, and configured total lead time.

finest granularity: scenario, sending, receiving

default reporting granularity:

- scenario, sending, receiving

logic:

- **data source**: `module6_output_deliveryplan` (the date columns below live here); configured lead times from `cfg_global_leadtime`.
- compute durations directly on the delivery rows; do not collapse by `ori_deployment_uid`.

1. simulated waiting MOQ time, per row = `actual_ship_date - planned_deployment_date`
2. simulated OTD (ship→delivery transit leg), per row = `actual_delivery_date - actual_ship_date`
3. simulated total lead time (planned→delivery), per row = `actual_delivery_date - planned_deployment_date` (equals waiting MOQ + OTD)
4. aggregate mean, median, p90 of the three simulated durations by lane (`sending`-`receiving` pair)
5. join the configured values from `cfg_global_leadtime` (columns `PDT, GR, MCT, OTD` — note the column is `PDT`, there is no `PTD`). Compare: simulated OTD ↔ configured `OTD`; simulated total lead time ↔ configured total lead time `PDT + GR`. Report by scenario and lane.

## Step 3 - Analyze the results with scenario context from workspace

Before interpreting simulation outputs, first inspect the corresponding workspace artifacts for the project and scenario, especially:

- `workspace/<project>/brief.md`
- `workspace/<project>/scenarios/<scenario>/design.md`
- `workspace/<project>/scenarios/<scenario>/results/` if available
- prior analysis runs: `workspace/<project>/analysis/LATEST.md` (and the scenario-level `scenarios/<scenario>/analysis/LATEST.md`) plus the run folder it points to, if available

Use these artifacts to understand:

- the business objective and decision context
- the scenario intent and expected trade-offs
- the scope, assumptions, and constraints
- whether there are already prior findings for the same project/scenario, and what the last analysis run concluded (read its `run.md` so a re-analysis builds on — and explicitly contrasts with — the previous run rather than silently restating it)

Apply the context-handling protocol above before writing recommendations:

- extract the minimum decision context from `brief.md` and `design.md`
- ask only the smallest missing set of user questions when workspace is incomplete
- if the user declines or does not know, continue with explicit downgrade labels such as fact-only, tentative, or recommendation withheld

Only after reading the workspace context, analyze the simulation outputs and derive findings.

## Step 4 - Output results

**One analysis = one timestamped run folder.** Every time you analyze or
re-analyze, create a NEW self-contained run folder; never write new results into
an existing run folder or mix iterations in one flat folder. This is what keeps
repeated analyses distinguishable.

Run folder location by level:

- **Project-level** analysis (scenario comparison, the default KPI bundle):
  `workspace/<project>/analysis/<run-id>/`
- **Scenario-level** re-analysis (one scenario):
  `workspace/<project>/scenarios/<scenario>/analysis/<run-id>/`

`<run-id>` = `<YYYYMMDD-HHMM>` plus an optional short label, e.g.
`20260514-0900-ss-sensitivity`. The `HHMM` matters — date alone collides when a
project is re-analyzed twice in one day (the cause of past `… - Copy.xlsx`
duplicates).

Each run folder is self-contained:

| Inside`<run-id>/`                        | Role                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------- |
| `extracts/`                              | Working: per-KPI extract CSVs, raw SQL dumps, scratch tables. Not stakeholder-facing. |
| `<project>_result_summary_<run-id>.xlsx` | Final consolidated KPI workbook (run-id in the name).                                 |
| `analysis.html`                          | Final rendered report.                                                                |
| `run.md`                                 | This run's record (see below).                                                        |

A pointer + log lives at the `analysis/` root (one level above the run folders):

- `workspace/<project>/analysis/LATEST.md` for project-level, or
  `workspace/<project>/scenarios/<scenario>/analysis/LATEST.md` for
  scenario-level.
- First line names the current run, e.g. `current: 20260514-0900-ss-sensitivity`.
- Below it, an appended **run log**, newest first — one entry per run:
  timestamp, level, scenarios + `run_id`s analyzed, scope, and what
  changed/why versus the previous run. `run.md` inside each folder holds the
  same record for that run.

Treat completed run folders as immutable history; to revise, create a new run
folder and a new `LATEST.md` entry. Do not reuse `deliverables/` for these
outputs — the run folder plus `LATEST.md` is the single source of truth, so
there is no separate copy to drift or duplicate.

Steps:

1. Write intermediate per-KPI extracts to `<run-id>/extracts/` as you compute
   them (one CSV per KPI × granularity is fine).
2. Output the consolidated excel to `<run-id>/<project>_result_summary_<run-id>.xlsx`,
   containing all required KPI tables by scenario with clear labeling of
   scenarios and KPI definitions.
3. Render the report to `<run-id>/analysis.html` using the uiuxpromax skill.
4. Write `<run-id>/run.md` and prepend a matching entry to the `analysis/`
   `LATEST.md` (update its `current:` pointer).

Presentation structure:

- Executive summary:
  - only describe the key findings and recommendations in a concise way for executives who may not read the full report
  - technical information should not be included
- Background and context:
  - summarize design in business context, goal, focus metrics, scenario intent, and key assumptions/constraints and confirmed scenario. source from `brief.md` and `design.md`
- Analysis scope
  - summarize the scope of the analysis, including simulation period, location/network, category/products, and any other relevant dimensions.
  - also call out any important assumptions made
  - source from `brief.md`, `design.md`, and any user input during the analysis process
- KPI results:
  - summary required metric at requested granularity, with charts and tables
  - Scenario comparison: if multiple scenarios are analyzed together, provide a clear comparison of the KPIs across scenarios, highlighting the differences and potential trade-offs
- Key findings
- Recommendations
- Appendix or calculation notes

# Reference map

## chainsight_database_structure

### Naming Rules

- **Config tables** (simulation inputs): `cfg_<scope>_<name>` — e.g. `cfg_global_network`, `cfg_m1_demandforecast`.
- **Output tables** (per-run simulation results): `module<N>_output_<name>` — e.g. `module5_output_stockonhandlog`.
- **Orchestrator tables** (`orchestrator_<name>`): end-to-end daily reconciliation / event logs.
- **Summary tables** (`summary_<name>`): consolidated full-run rollups.
- **Common columns** — do not treat these as data dimensions:
  - config tables also carry `config_name, config_type, db_write_time`.
  - output / orchestrator / summary tables also carry `run_id, sim_date, config_name, db_write_time`.
- **Scenario identity**: filter output/orchestrator/summary tables by `run_id` (one simulation run); filter config tables by `config_name`. `sim_date` is the simulation batch/day a row was written — when summing business quantities over a full run, aggregate by the business `date`, not `sim_date`.
- This catalog reflects the `fem_test` reference database (69 tables). A given project DB may omit tables for modules it did not run, and some logs below are placeholders that are empty unless the relevant condition occurred.

### Config Tables (Inputs)

| Scope  | Table                                | Description                                                                                              |
| ------ | ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Global | `cfg_global_seed`                  | RNG seed                                                                                                 |
| Global | `cfg_global_network`               | Network topology (`material, location, sourcing, location_type, eff_from, eff_to`)                     |
| Global | `cfg_global_leadtime`              | Lane lead times (`sending, receiving, pdt, gr, mct, otd`)                                              |
| Global | `cfg_global_demandpriority`        | Demand priority rules                                                                                    |
| Global | `cfg_global_spacecapacity`         | Warehouse space capacity by location                                                                     |
| M1     | `cfg_m1_demandforecast`            | Weekly demand forecast (`week, material, location, quantity`)                                          |
| M1     | `cfg_m1_initialinventory`          | Starting inventory                                                                                       |
| M1     | `cfg_m1_forecasterror`             | Forecast error std by`order_type`                                                                      |
| M1     | `cfg_m1_ordercalendar`             | Order-day flags                                                                                          |
| M1     | `cfg_m1_aoconfig`                  | Advance-order config (`advance_days, ao_percent`)                                                      |
| M1     | `cfg_m1_dpsconfig`                 | Direct plant shipment split (`dps_location, dps_percent`)                                              |
| M1     | `cfg_m1_supplychoiceconfig`        | Weekly supply-choice adjustments (`week, adjust_quantity`)                                             |
| M3     | `cfg_m3_safetystock`               | Safety stock targets by date                                                                             |
| M4     | `cfg_m4_materiallocationlinecfg`   | Material→line assignment + prod params (`delegate_line, prd_rate, min_batch, rv, ptf, lsk, day, mct`) |
| M4     | `cfg_m4_linecapacity`              | Line capacity calendar (`location, line, date, capacity`)                                              |
| M4     | `cfg_m4_changeoverdefinition`      | Changeover definitions (`changeover_id, line, time, cost, mu_loss`)                                    |
| M4     | `cfg_m4_changeovermatrix`          | Material-pair →`changeover_id`                                                                        |
| M4     | `cfg_m4_productionreliability`     | Line reliability (`location, line, pr`)                                                                |
| M5     | `cfg_m5_deployconfig`              | Deployment params (`material, sending, receiving, moq, rv, lsk, day`)                                  |
| M5     | `cfg_m5_pushpullmodel`             | Push/pull model (`material, sending, model`)                                                           |
| M6     | `cfg_m6_materialmd`                | Material master for transport (`weight, volume, demand_unit_to_weight, demand_unit_to_volume`)         |
| M6     | `cfg_m6_trucktypespecs`            | Truck type capacity (`capacity_qty_in_weight, capacity_qty_in_volume`)                                 |
| M6     | `cfg_m6_truckreleasecon`           | Truck release constraints (`optimal_type, wfr, vfr`)                                                   |
| M6     | `cfg_m6_truckcapacityplan`         | Truck-number plan by lane/date (`truck_type, truck_number`)                                            |
| M6     | `cfg_m6_deliverydelaydistribution` | Delivery delay probabilities (`delay_days, probability`)                                               |
| M6     | `cfg_m6_mdqbypassrules`            | MDQ bypass rules (`condition_logic, rule_id`)                                                          |
| Meta   | `cfg_config_guide`                 | Config guide/reference sheet — not simulation input data                                                |

### Output Tables (Per-Run Results)

| Module | Table                                | Content                                                                                                                                                                   |
| ------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M1     | `module1_output_orderlog`          | Customer order log (`date, material, location, demand_type, simulation_date, advance_days, quantity`)                                                                   |
| M1     | `module1_output_shipmentlog`       | Shipment records (`date, material, location, quantity, demand_type, order_id`)                                                                                          |
| M1     | `module1_output_cutlog`            | Demand cut log                                                                                                                                                            |
| M1     | `module1_output_supplydemandlog`   | Supply-demand balance (`demand_element`)                                                                                                                                |
| M1     | `module1_output_summary`           | Daily M1 totals (orders / shipments / cuts / supplydemand)                                                                                                                |
| M3     | `module3_output_netdemand`         | Net demand after netting (`requirement_date, layer, horizon_days`)                                                                                                      |
| M4     | `module4_output_productionplan`    | Production schedule (`produced_qty, con/uncon_planned_qty, changeover_id, changeover_time`)                                                                             |
| M4     | `module4_output_changeoverlog`     | Changeover events (`date, location, line, changeover_type, count, time, cost, mu_loss`)                                                                                 |
| M4     | `module4_output_capacityexceed`    | Capacity exceedance log (placeholder; empty unless exceeded)                                                                                                              |
| M4     | `module4_output_validation`        | M4 validation log (placeholder)                                                                                                                                           |
| M5     | `module5_output_stockonhandlog`    | Daily SOH (`beginning_soh, production, in_transit, delivery_gr, today_shipment, deployed_qty, ending_soh`)                                                              |
| M5     | `module5_output_deploymentplan`    | Deployment decisions (`demand_qty, deployed_qty, planned_delivery_date, leadtime, is_cross_node, quota`)                                                                |
| M5     | `module5_output_unfulfilledlog`    | Unfulfilled deployment (`unfulfilled_qty, reason`)                                                                                                                      |
| M5     | `module5_output_validation`        | M5 validation issues (`no, issue`)                                                                                                                                      |
| M6     | `module6_output_deliveryplan`      | Transport plan (`vehicle_uid, ori_deployment_uid, planned_deployment_date, actual_ship_date, actual_delivery_date, delivery_qty, truck_type, truck_load_pct, wfr, vfr`) |
| M6     | `module6_output_vehiclelog`        | Vehicle utilization (`vehicle_uid, total_units, total_weight, total_volume, wfr, vfr, trigger`)                                                                         |
| M6     | `module6_output_truckusagelog`     | Truck usage (`truck_type, truck_used`)                                                                                                                                  |
| M6     | `module6_output_bypassrulehitlog`  | MDQ bypass-rule hits (`ori_deployment_uid, rule_id, vehicle_uid`)                                                                                                       |
| M6     | `module6_output_unsatisfiedmdqlog` | Unsatisfied MDQ (placeholder)                                                                                                                                             |
| M6     | `module6_output_validationlog`     | M6 validation log (placeholder)                                                                                                                                           |

### Orchestrator Tables (End-to-End Daily)

| Table                                            | Content                                                                                                        |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `orchestrator_unrestricted_inventory`          | Available inventory (`date, material, location, quantity`) — source for inventory / RCCP KPIs               |
| `orchestrator_daily_logs`                      | Daily event log (`timestamp, date, event_type, message`)                                                     |
| `orchestrator_inventory_change_log`            | Inventory reconciliation (`beginning_inventory, ending_inventory, calculated_ending, balance_diff`)          |
| `orchestrator_shipment_log`                    | Shipment log (`date, material, location, quantity, type`)                                                    |
| `orchestrator_delivery_shipment_log`           | Delivery + shipment log (`ori_deployment_uid, actual_ship_date, actual_delivery_date, type`)                 |
| `orchestrator_delivery_gr`                     | Delivery goods receipt (`ori_deployment_uid, vehicle_uid, actual_ship_date`)                                 |
| `orchestrator_production_gr`                   | Production goods receipt (`date, material, location, quantity`)                                              |
| `orchestrator_production_plan_backlog`         | Production backlog (`material, location, available_date, quantity`)                                          |
| `orchestrator_planning_intransit`              | In-transit planning (`transit_uid, actual_ship_date, actual_delivery_date, ori_deployment_uid, vehicle_uid`) |
| `orchestrator_open_deployment`                 | Open deployments (placeholder)                                                                                 |
| `orchestrator_open_deployment_pastdue_cleanup` | Past-due open-deployment cleanup (placeholder)                                                                 |
| `orchestrator_space_quota`                     | Space quota (placeholder)                                                                                      |

### Summary Tables (Consolidated Full-Run)

`summary_output_full*` are full-run consolidations of the corresponding per-run
`module*_output_*` tables; the KPI logic above reads the `module*_output_*`
tables. Use these when you specifically want the consolidated full-horizon form.

| Table                                      | Content                                                                                                                     |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `summary_output_ordershipmentcutsummary` | Pre-aggregated order / shipment / cut qty (`simulation_date, date, material, location, order_qty, shipment_qty, cut_qty`) |
| `summary_output_fullproductionplan`      | Full-run production plan (mirrors`module4_output_productionplan`)                                                         |
| `summary_output_fulldeploymentplan`      | Full-run deployment plan (mirrors`module5_output_deploymentplan`)                                                         |
| `summary_output_fulldeliveryplan`        | Full-run delivery plan (mirrors`module6_output_deliveryplan`)                                                             |
| `summary_output_fullchangeoverlog`       | Full-run changeover log (mirrors`module4_output_changeoverlog`)                                                           |
| `summary_output_fulltruckusage`          | Full-run truck usage (mirrors`module6_output_truckusagelog`)                                                              |
| `summary_output_fullcapacityexceed`      | Full-run capacity exceed (placeholder)                                                                                      |
| `summary_historical_inventory_record`    | Historical inventory record (placeholder)                                                                                   |

### System / Meta Tables

| Table              | Content                                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| `sim_checkpoint` | Run checkpoint/state (`run_key, run_id, config_name, status, orch_state_json`) — not analysis data |
| `sim_m4_state`   | M4 internal state blobs (`run_id, sim_date, file_type, file_content`) — not analysis data          |

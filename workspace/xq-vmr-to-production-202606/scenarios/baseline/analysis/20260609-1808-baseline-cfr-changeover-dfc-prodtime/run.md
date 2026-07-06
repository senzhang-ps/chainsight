# Analysis run 20260609-1808-baseline-cfr-changeover-dfc-prodtime

- **Level:** scenario (baseline)
- **Project:** xq-vmr-to-production-202606
- **DB run_id:** db_baseline_20260606_205517
- **config_name:** baseline
- **Generated:** 2026-06-09 18:08

## KPIs produced
1. CFR by category x production line x month
2. Changeover count by production line x month x changeover id
3. Changeover cost by production line x month x changeover id
4. Month-end DFC (days forward coverage) by category x production line x month
5. Production total time (production + changeover, hr) by category x production line x month

## Scope & method
- Period 2026-06-29 → 2026-11-01, XQ plant 1864, categories HC (Hair) / PCC.
- CFR = Σ shipment / Σ order (ratio of sums); order log deduped on 7-field key; month = business date.
- Production line = delegate_line from cfg_m4_materiallocationlinecfg (material level; demand at all DCs attributed to producing line).
- Changeover count/cost from module4_output_changeoverlog grouped by month × line × changeover_type (= changeover id).
- DFC = month-end DC ending_soh (module5_output_stockonhandlog, 22 DC locations, plant 1864 excluded) / avg daily forward 30-day demand forecast (cfg_m1_demandforecast at DCs). Ratio of sums per rollup.
- Production total time = production time + changeover time (hr). Production time = con_planned_qty (module4_output_productionplan) / prd_rate (cfg_m4_materiallocationlinecfg, unit/hr) per material then summed; changeover time = module4_output_changeoverlog.time summed; month = production_plan_date (2026-07, 2026-08, 2026-09, 2026-10). Changeover allocated to category pro-rata by production hours within line x month (mixed line: XQHD).

## Data lineage
- Category source: databricks — fresh Databricks pull (category_en).
- Category coverage: databricks 236/286; enriched 278/286.
- Category resolution by order qty: databricks 83.8%, line-inferred 14.0%, unmapped 2.1%.
- Line-inferred materials (42): new VMR codes on single-category lines XQHG/XQHK → Hair.
- DFC forward window 30d; forecast horizon ends 2026-11-08; partial-window months: 2026-10, 2026-11.
- Unmapped-line materials: 80853439, 80881660, 83908390.
- Unmapped-category materials: 21184776, 21184777, 21200191, 21205101, 21387644, 21403972, 21420779, 21425608.

## Outputs
- `xq-vmr-to-production-202606_result_summary_20260609-1808-baseline-cfr-changeover-dfc-prodtime.xlsx`
- `analysis.html`
- `extracts/` (per-KPI CSVs + base tables)

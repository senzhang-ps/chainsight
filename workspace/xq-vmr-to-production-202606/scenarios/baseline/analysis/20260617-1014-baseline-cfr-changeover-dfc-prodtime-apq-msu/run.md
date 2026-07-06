# Analysis run 20260617-1014-baseline-cfr-changeover-dfc-prodtime-apq-msu

- **Level:** scenario (baseline)
- **Project:** xq-vmr-to-production-202606
- **DB run_id:** db_baseline_20260616_154430
- **config_name:** baseline
- **Generated:** 2026-06-17 10:14

## KPIs produced
1. CFR by category x production line x month
2. Changeover count by production line x month x changeover id
3. Changeover cost by production line x month x changeover id
4. Month-end DFC (days forward coverage) by category x production line x month
5. Production total time (production + changeover, hr) by category x production line x month
6. APQ (avg constrained planned qty per production run) by production line and by material
7. Wash count / MSU by production line (wash = changeover id 2/3; MSU = qty x SU factor / 1000)

## Scope & method
- Period 2026-06-29 → 2026-11-01, XQ plant 1864, categories HC (Hair) / PCC.
- CFR = Σ shipment / Σ order (ratio of sums); order log deduped on 7-field key; month = business date.
- Production line = delegate_line from cfg_m4_materiallocationlinecfg (material level; demand at all DCs attributed to producing line).
- Changeover count/cost from module4_output_changeoverlog grouped by month × line × changeover_type (= changeover id).
- DFC = month-end DC ending_soh (module5_output_stockonhandlog, 22 DC locations, plant 1864 excluded) / avg daily forward 30-day demand forecast (cfg_m1_demandforecast at DCs). Ratio of sums per rollup.
- Production total time = production time + changeover time (hr). Production time = con_planned_qty (module4_output_productionplan) / prd_rate (cfg_m4_materiallocationlinecfg, unit/hr) per material then summed; changeover time = module4_output_changeoverlog.time summed; month = production_plan_date (2026-07, 2026-08, 2026-09, 2026-10). Changeover allocated to category pro-rata by production hours within line x month (mixed line: XQHD, XQHG).
- APQ = sum(con_planned_qty) / production-run count. A run = a module4_output_productionplan row with non-null changeover_id (begins with a changeover); null-changeover continuation rows are not counted. Overall 1748.0 units/run = 9187372 / 5256. By line and by material.
- Wash count / MSU = wash count / MSU per line. Wash count = changeover events with id in 2, 3 (module4_output_changeoverlog). MSU = sum(con_planned_qty x SU factor / 1000) over the line's materials. Overall 0.326 washes/MSU = 4308 / 13218.5 MSU.

## Data lineage
- Category source: databricks — manual-input filled 44 categories.
- Category coverage: databricks 280/384; enriched 384/384.
- Category resolution by order qty: databricks 60.0%, line-inferred 40.0%, unmapped 0.0%.
- Line-inferred materials (104): new VMR codes on single-category lines XQHG/XQHK → Hair.
- DFC forward window 30d; forecast horizon ends 2027-01-03; partial-window months: none.
- SU factor source: SUF workbook workspace\xq-vmr-to-production-202606\scenarios\baseline\config\SUF for XQ HC ChainSight.xlsx (primary, 191 materials) + Databricks su_factor_for_buom fallback (102 materials); 0 missing. Overlapping sources agree to ~0.05%.
- Unmapped-line materials: 21359277, 21368921, 21377834, 21479013, 21479014, 21483803, 21488493.
- Unmapped-category materials: none.

## Outputs
- `xq-vmr-to-production-202606_result_summary_20260617-1014-baseline-cfr-changeover-dfc-prodtime-apq-msu.xlsx`
- `analysis.html`
- `extracts/` (per-KPI CSVs + base tables)

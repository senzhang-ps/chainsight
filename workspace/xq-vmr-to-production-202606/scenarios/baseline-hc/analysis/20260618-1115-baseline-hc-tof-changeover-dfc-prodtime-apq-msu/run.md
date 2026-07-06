# Analysis run 20260618-1115-baseline-hc-tof-changeover-dfc-prodtime-apq-msu

- **Level:** scenario (baseline-hc)
- **Project:** xq-vmr-to-production-202606
- **DB run_id:** db_baseline-hc_20260617_203548
- **config_name:** baseline-hc
- **Generated:** 2026-06-18 11:15

## KPIs produced
1. TOF by category x production line x month
2. Changeover count by production line x month x changeover id
3. Changeover cost by production line x month x changeover id
4. Month-end DFC (days forward coverage) by category x production line x month
5. Production total time (production + changeover, hr) by category x production line x month
6. APQ (avg MSU produced per production run) by production line and by material
7. Wash count / MSU by production line (wash = changeover id 2/3; MSU = qty x SU factor / 1000)

## Scope & method
- Period 2026-06-29 → 2026-10-31 (reported through end of October; Nov tail excluded), XQ plant 1864, categories HC (Hair) / PCC.
- TOF = Σ shipment / Σ order (ratio of sums); order log deduped on 7-field key; month = business date.
- Production line = delegate_line from cfg_m4_materiallocationlinecfg (material level; demand at all DCs attributed to producing line).
- Changeover count/cost from module4_output_changeoverlog grouped by month × line × changeover_type (= changeover id).
- DFC = month-end DC ending_soh (module5_output_stockonhandlog, 20 DC locations, plant 1864 excluded) / avg daily forward 30-day demand forecast (cfg_m1_demandforecast at DCs). Ratio of sums per rollup.
- Production total time = production time + changeover time (hr). Production time = con_planned_qty (module4_output_productionplan) / prd_rate (cfg_m4_materiallocationlinecfg, unit/hr) per material then summed; changeover time = module4_output_changeoverlog.time summed; month = production_plan_date (2026-07, 2026-08, 2026-09, 2026-10). Changeover allocated to category pro-rata by production hours within line x month (mixed line: none).
- APQ = sum(con_planned_qty x SU factor / 1000) = MSU / production-run count. A run = a module4_output_productionplan row with non-null changeover_id (begins with a changeover); null-changeover continuation rows are not counted. Overall 2.939 MSU/run = 6489.8 MSU / 2208. By line and by material.
- Wash count / MSU = wash count / MSU per line. Wash count = changeover events with id in 2, 3 (module4_output_changeoverlog). MSU = sum(con_planned_qty x SU factor / 1000) over the line's materials. Overall 0.313 washes/MSU = 2029 / 6489.8 MSU.

## Data lineage
- Category source: databricks — manual-input filled 22 categories.
- Category coverage: databricks 162/191; enriched 191/191.
- Category resolution by order qty: databricks 81.6%, line-inferred 18.4%, unmapped 0.0%.
- Line-inferred materials (29): new VMR codes on single-category lines XQHG/XQHK → Hair.
- DFC forward window 30d; forecast horizon ends 2027-01-03; partial-window months: none.
- SU factor source: SUF workbook workspace\xq-vmr-to-production-202606\scenarios\baseline-hc\config\SUF for XQ HC ChainSight.xlsx (primary, 191 materials) + Databricks su_factor_for_buom fallback (0 materials); 0 missing. Overlapping sources agree to ~0.05%.
- Unmapped-line materials: none.
- Unmapped-category materials: none.

## Outputs
- `xq-vmr-to-production-202606_result_summary_20260618-1115-baseline-hc-tof-changeover-dfc-prodtime-apq-msu.xlsx`
- `analysis.html`
- `extracts/` (per-KPI CSVs + base tables)

# Analysis run 20260607-1435-baseline-cfr-changeover

- **Level:** scenario (baseline)
- **Project:** xq-vmr-to-production-202606
- **DB run_id:** db_baseline_20260606_205517
- **config_name:** baseline
- **Generated:** 2026-06-07 14:35

## KPIs produced
1. CFR by category × production line × month
2. Changeover count by production line × month × changeover id
3. Changeover cost by production line × month × changeover id

## Scope & method
- Period 2026-06-29 → 2026-11-01, XQ plant 1864, categories HC (Hair) / PCC.
- CFR = Σ shipment / Σ order (ratio of sums); order log deduped on 7-field key; month = business date.
- Production line = delegate_line from cfg_m4_materiallocationlinecfg (material level; demand at all DCs attributed to producing line).
- Changeover count/cost from module4_output_changeoverlog grouped by month × line × changeover_type (= changeover id).

## Data lineage
- Category source: databricks — fresh Databricks pull (category_en).
- Category coverage: databricks 236/286; enriched 278/286.
- Category resolution by order qty: databricks 83.8%, line-inferred 14.0%, unmapped 2.1%.
- Line-inferred materials (42): new VMR codes on single-category lines XQHG/XQHK → Hair.
- Unmapped-line materials: 80853439, 80881660, 83908390.
- Unmapped-category materials: 21184776, 21184777, 21200191, 21205101, 21387644, 21403972, 21420779, 21425608.

## Outputs
- `xq-vmr-to-production-202606_result_summary_20260607-1435-baseline-cfr-changeover.xlsx`
- `analysis.html`
- `extracts/` (per-KPI CSVs + base tables)

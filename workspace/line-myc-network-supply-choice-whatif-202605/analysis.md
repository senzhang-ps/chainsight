# Line MYC Network Supply Choice What-If Analysis

## Executive Summary

This report has been rebuilt against the updated result-analysis skill.
Two outputs now exist together:

1. a complete KPI workbook at required granularity
2. a cleaner Markdown report focused on decision interpretation

The latest completed runs used in this analysis are:

- `wf1`: `db_wf1-produce-hs-order-5050-current-ss_20260512_230651`
- `wf2`: `db_wf2-produce-5050-order-hs-current-ss_20260512_230707`
- `wf3`: `db_wf3-produce-hs-order-5050-reduced-ss_20260512_230717`
- `wf4`: `db_wf4-produce-5050-order-hs-reduced-ss_20260512_230848`

The key result is unchanged in direction but clearer in evidence:

- `wf2` is the strongest service scenario in both May and June.
- `wf3` is the best balanced space-relief option.
- `wf1` is largely dominated by `wf2`.
- `wf4` is largely dominated by `wf3`.

The major improvement versus the prior report is completeness:

- all required-granularity KPI tables are now packaged into a single Excel file
- the lane KPI has been expanded to `mean / median / p90` plus config components
- the report now separates full KPI outputs from decision-level interpretation

## Background And Context

Business objective:

Evaluate four what-if scenarios that combine supply choice policy and safety
stock policy across the full sourcing network for Line M/Y/C SKUs, so the
team can compare service and inventory implications before selecting a scenario.

Scope context:

- Network: full Line M/Y/C sourcing network
- Period: `2026-05-04` to `2026-06-19`
- Baseline planning version: Apr 27 LBE `50/50`
- Initial inventory basis: Apr 27 actual stock on hand plus in-transit

Scenario definitions:

| Scenario | Produce Policy | Order Policy | Safety Stock Policy |
| --- | --- | --- | --- |
| `wf1` | high-side | 50/50 forecast | current SS |
| `wf2` | 50/50 | high-side | current SS |
| `wf3` | high-side | 50/50 forecast | DTC `30d -> 15d` |
| `wf4` | 50/50 | high-side | DTC `30d -> 15d` |

Evidence-strength note:

- Scenario context is adequate because `brief.md` and `design.md` are present.
- Absolute business pass/fail thresholds remain incomplete because KPI baseline
  targets and final selection rule are still `TBD` in `brief.md`.
- Recommendations below are therefore relative scenario recommendations, not
  absolute go/no-go approvals.

## Analysis Scope

- Project: `line-myc-network-supply-choice-whatif-202605`
- Data source: PostgreSQL simulation output tables filtered by latest `run_id`
- KPI package: Service, Month End Inventory (cases), Changeover, Lane LT
- Reliability note: scenario config validation previously passed with warnings

## KPI Results

### Complete KPI Workbook

The complete required-granularity KPI package is consolidated in:

- `deliverables/line-myc-network-supply-choice-whatif-202605_result_summary_20260513.xlsx`

Workbook structure:

| Sheet | Granularity | Rows | Notes |
| --- | --- | ---: | --- |
| `metadata` | workbook metadata | 6 | project, timestamp, period, notes |
| `scenario_definitions` | scenario | 4 | levers and run IDs |
| `scenario_summary` | scenario | 4 | compact comparison plus charts |
| `service_month` | month | 8 | full month-level service |
| `service_month_loc` | month x location | 156 | full service table at requested grain |
| `month_end_inv_loc` | scenario x month x location | 216 | month end inventory by location (cases) |
| `month_end_inv_network` | scenario x month | 8 | month end inventory network total (cases) |
| `month_end_inv_material_loc` | scenario x month x location x material | ~4,200 | month end inventory by material x location (cases) |
| `changeover_month_line` | month x location x line x type | 44 | full changeover result table |
| `lane_lt_diag` | scenario x sending x receiving | 94 | mean / median / p90 diagnostics |

CSV deliverables are kept in sync with the workbook:

- deliverables/service-by-month.csv
- deliverables/service-by-month-by-location.csv
- deliverables/changeover-by-month-line-type.csv
- deliverables/lane-leadtime-by-scenario.csv

### Service By Month

| Scenario | May Service | Jun Service |
| --- | ---: | ---: |
| `wf1` | 0.9881 | 0.9893 |
| `wf2` | 0.9915 | 0.9946 |
| `wf3` | 0.9729 | 0.9822 |
| `wf4` | 0.9604 | 0.9702 |

Interpretation:

- `wf2` is best in both months.
- `wf1` is second-best on monthly service, but not enough to offset its
  weaker space outcome versus `wf2`.
- reduced-safety-stock scenarios (`wf3`, `wf4`) remain lower on service,
  with `wf4` the weakest.

### Service By Month By Location

The full `month x location` service table is in workbook sheet `service_month_loc`.
For decision use, the highest-value summary is the weak-node comparison below.

Weakest June location by scenario:

| Scenario | Weakest June Location | June Service | June Order Qty |
| --- | --- | ---: | ---: |
| `wf1` | `E564` | 0.8800 | 25 |
| `wf2` | `E569` | 0.7928 | 251 |
| `wf3` | `D767` | 0.7292 | 48 |
| `wf4` | `C937` | 0.8238 | 993 |

Most important high-volume node, `C937`:

| Scenario | `C937` May Service | `C937` Jun Service | Jun Order Qty |
| --- | ---: | ---: | ---: |
| `wf1` | 0.9459 | 0.9047 | 787 |
| `wf2` | 0.9904 | 1.0000 | 819 |
| `wf3` | 0.9443 | 0.8503 | 862 |
| `wf4` | 0.9689 | 0.8238 | 993 |

Takeaway:

- `E564` is no longer the most decision-relevant issue because its weak
  service is low-volume.
- `C937` remains the most important comparison node because it combines
  large volume with clear scenario spread.
- `wf2` is strongest at `C937`; `wf4` is weakest.

### Changeover By Month By Production Line By Type

The full `month x location x line x changeover_type` table is in workbook sheet
`changeover_month_line`.

Monthly total changeover burden:

| Scenario | May Count | May Time | Jun Count | Jun Time |
| --- | ---: | ---: | ---: | ---: |
| `wf1` | 22.00 | 35.83 | 3.00 | 5.67 |
| `wf2` | 23.00 | 34.68 | 3.00 | 4.50 |
| `wf3` | 11.00 | 17.91 | 1.00 | 0.83 |
| `wf4` | 11.00 | 17.91 | 3.00 | 5.50 |

Representative dominant May patterns at `0386`:

| Scenario | Line | Changeover Type | Count | Share % |
| --- | --- | --- | ---: | ---: |
| `wf1` | `HPSMPACK` | `Washout_changeover` | 4.00 | 0.6667 |
| `wf1` | `HPSYPACK` | `Washout_changeover` | 4.00 | 0.6667 |
| `wf2` | `HPSYPACK` | `Washout_changeover` | 5.00 | 0.6250 |
| `wf2` | `HPSCPACK` | `Washout` | 5.00 | 0.5556 |
| `wf3` | `HPSMPACK` | `Washout_changeover` | 3.00 | 0.7500 |
| `wf4` | `HPSMPACK` | `Washout_changeover` | 3.00 | 0.7500 |

Takeaway:

- `wf3` and `wf4` cut May changeover burden roughly in half versus `wf1` / `wf2`
- the dominant operational burden is still washout-related activity at `0386`

### Month End Inventory

Network month end inventory (last available date per month, case qty):

| Scenario | May ME Date | May Inv (Cases) | Jun ME Date | Jun Inv (Cases) |
| --- | --- | ---: | --- | ---: |
| `wf1` | 2026-05-31 | 175,141 | 2026-06-19 | 165,810 |
| `wf2` | 2026-05-31 | 152,415 | 2026-06-19 | 140,876 |
| `wf3` | 2026-05-31 | 160,528 | 2026-06-19 | 142,720 |
| `wf4` | 2026-05-31 | 136,418 | 2026-06-19 | 119,703 |

Note: June ME date is 2026-06-19 (simulation end date), not a calendar month end.

Key node `0386` month end inventory (cases):

| Scenario | 0386 May Inv (Cases) | 0386 Jun Inv (Cases) | Δ vs wf1 Jun |
| --- | ---: | ---: | --- |
| `wf1` | 30,411 | 39,835 | — |
| `wf2` | 29,899 | 34,403 | −13.6% |
| `wf3` | 21,418 | 27,318 | −31.4% |
| `wf4` | 18,787 | 28,458 | −28.6% |

Takeaway:

- current-SS scenarios (`wf1`, `wf2`) hold 15–18% more end-of-period inventory
  than reduced-SS scenarios (`wf3`, `wf4`)
- `wf4` has the lowest absolute network inventory (119,703 cases at
  sim end), but this comes at the worst service penalty
- `0386` is the dominant inventory node in all scenarios, holding 24–30% of
  network ME inventory
- Inventory draw-down from May to June is visible in all scenarios, consistent
  with demand consumption outpacing replenishment buildup
- Material-level ME inventory is available in the workbook `month_end_inv_material_loc` sheet

### Lane Lead Time Diagnostics

The full `scenario x sending x receiving` diagnostics table is in workbook sheet
`lane_lt_diag`. This sheet now includes:

- `waiting MOQ` mean / median / p90
- `OTD` mean / median / p90
- `PTD` mean / median / p90
- `actual total LT` mean / median / p90
- config `pdt`, `otd`, `gr`, and `cfg_total_lt`

Key lanes:

| Scenario | Lane | Mean Total LT | P90 Total LT | Mean Waiting MOQ | Mean OTD | Config Total LT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `wf1` | `C816 -> C937` | 7.6898 | 9.8 | 1.5682 | 6.1216 | 8.0 |
| `wf2` | `C816 -> C937` | 7.0277 | 8.0 | 0.8934 | 6.1343 | 8.0 |
| `wf3` | `C816 -> C937` | 7.3460 | 9.0 | 1.2299 | 6.1161 | 8.0 |
| `wf4` | `C816 -> C937` | 7.1782 | 8.0 | 1.0625 | 6.1157 | 8.0 |
| `wf1` | `C810 -> A673` | 7.4907 | 9.0 | 1.3808 | 6.1098 | 8.0 |
| `wf2` | `C810 -> A673` | 7.6217 | 9.0 | 1.4846 | 6.1371 | 8.0 |
| `wf3` | `C810 -> A673` | 7.3173 | 8.0 | 1.1893 | 6.1280 | 8.0 |
| `wf4` | `C810 -> A673` | 7.3768 | 9.0 | 1.2525 | 6.1242 | 8.0 |
| `wf1` | `0386 -> D594` | 4.9072 | 6.0 | 0.7913 | 4.1159 | 6.0 |
| `wf2` | `0386 -> D594` | 4.8754 | 6.0 | 0.7913 | 4.0841 | 6.0 |
| `wf3` | `0386 -> D594` | 5.0809 | 6.0 | 0.9551 | 4.1258 | 6.0 |
| `wf4` | `0386 -> D594` | 5.0755 | 6.0 | 0.9542 | 4.1213 | 6.0 |

Takeaway:

- mean total LT on the focus lanes is now below configured total LT in all scenarios
- tail risk still exists on some lanes: for example `C816 -> C937` has `P90 = 9.8` in `wf1`
  and `P90 = 9.0` in `wf3`, both above config total LT `8.0`
- `wf2` and `wf4` have the cleanest `C816 -> C937` tail profile among the four scenarios
- `0386 -> D594` remains stable and not a primary concern lane

## Scenario Comparison

| Scenario | May Service | Jun Service | `C937` Jun Service | Net Jun ME Inv (Cases) | `0386` Jun ME Inv (Cases) | May CO Count | Role |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wf1` | 0.9881 | 0.9893 | 0.9047 | 165,810 | 39,835 | 22 | dominated by `wf2` |
| `wf2` | 0.9915 | 0.9946 | 1.0000 | 140,876 | 34,403 | 23 | service-first |
| `wf3` | 0.9729 | 0.9822 | 0.8503 | 142,720 | 27,318 | 11 | balanced inventory-relief |
| `wf4` | 0.9604 | 0.9702 | 0.8238 | 119,703 | 28,458 | 11 | dominated by `wf3` |

## Key Findings

1. `wf2` is the best service scenario at both the network-month level and the
   most decision-relevant node `C937`.

2. `wf1` no longer sits on the efficient frontier. It is weaker than `wf2` on
   service and has higher inventory, without a compensating operational benefit.

3. `wf3` and `wf4` deliver almost identical inventory relief and May changeover relief,
   but `wf3` is consistently safer than `wf4` on service.

4. The lane KPI should no longer be read as a simple mean-only result. The updated
   `mean / median / p90` view shows that average lane timing is under control, but
   some tail slippage remains on `C816 -> C937` and `C810 -> A673`.

5. Month end inventory shows reduced-SS scenarios
   carry 14–28% less end-of-period inventory than current-SS scenarios. The highest
   absolute inventory is at `0386` in all scenarios.

6. The updated MDQbypass run set materially changed the lane diagnosis relative to
   the previous version of the report. Reuse only the latest workbook and this report.

## Recommendations

1. If the decision priority is service protection, choose `wf2`.

2. If the decision priority is the best balance between space relief and service,
   choose `wf3`.

3. Do not select `wf1` over `wf2` unless there is an external business rule not
   represented in the current KPI package.

4. Do not select `wf4` over `wf3` on current evidence. The space benefit is nearly
   the same, but service is worse at the most important weak node.

5. If the team wants to keep monitoring lane execution risk after choosing a scenario,
   monitor `C816 -> C937` and `C810 -> A673` using the workbook `lane_lt_diag` sheet,
   especially `P90 total LT` and `P90 waiting MOQ`.

## Appendix / Calculation Notes

- Service uses business date `date`, not `sim_date`
- Service denominator does not use coarse `distinct` dedup on `module1_output_orderlog`
- Changeover percent uses total line-month count as denominator
- Lane diagnostics are aggregated at `scenario x sending x receiving`
- Month end inventory uses last available date per month from
  `orchestrator_unrestricted_inventory`, measured in case qty
- June ME date is simulation end date (2026-06-19), not calendar month end
- Material-level ME inventory available in workbook `month_end_inv_material_loc` sheet
- Lane columns now include:
  - waiting MOQ `mean / median / p90`
  - OTD `mean / median / p90`
  - PTD `mean / median / p90`
  - actual total LT `mean / median / p90`
  - config `pdt`, `otd`, `gr`, `cfg_total_lt`
- Config total LT in this report is `pdt + gr`, kept for continuity with the
  project’s historical lane total-LT comparison

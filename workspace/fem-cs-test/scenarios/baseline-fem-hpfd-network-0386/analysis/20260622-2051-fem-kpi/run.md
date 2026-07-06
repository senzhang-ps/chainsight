# FEM baseline KPI analysis - run 20260622-2051-fem-kpi

- level: scenario
- project: fem-cs-test
- scenario: baseline-fem-hpfd-network-0386
- run_id: db_baseline-fem-hpfd-network-0386_20260602_223559
- database: fem_test
- generated: 2026-06-22 20:51

## Scope
- locations (DC): A668, A672, A673, A680, A715, A716, C810, C816
- months: Feb 2026, Mar 2026
- production line (FGC line12): HPFD at plant 0386
- KPI 3 (change-over cost) skipped per user request.

## Definitions
- ToF = shipment_qty / order_qty (order log deduped on 7-key; shipment log not deduped).
- DFC = month-end DC ending_soh / avg daily forward demand over next 30 days.
- MSU = con_planned_qty * su_factor / 1000 (su_factor: databricks ps_psc_sku_master.su_factor_for_buom; coverage 29/29).
- Production run = a contiguous production campaign (PO); a >1-day gap starts a new run. Equals the change-over count for change-over-started materials, plus the base product's change-over-free campaigns.
- PKG CO ids = 1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3; CONV CO ids = 2-1, 2-11, 2-2, 2-3, 3.

## Headline results
- ToF Feb 2026: 98.8% (163,312/165,290)
- ToF Mar 2026: 98.2% (111,872/113,867)
- DFC Feb 2026 (all DCs): 34.9 days (inv 132,810 / 3,807.7 per day)
- DFC Mar 2026 (all DCs): 42.8 days (inv 76,716 / 1,794.0 per day)
- Change-over Feb 2026: 19 events, 17.5 hr
- Change-over Mar 2026: 14 events, 11.5 hr
- Production total time Feb 2026: 90.4 hr (prod 72.9 + CO 17.5; CO share 19.4%)
- Production total time Mar 2026: 65.6 hr (prod 54.1 + CO 11.5; CO share 17.5%)
- APQ overall: 5.822 MSU/run (total MSU 203.8 / 35 runs)
- PKG CO/MSU: 0.1521 (31 CO / 203.8 MSU)
- CONV CO/MSU: 0.0442 (9 CO / 203.8 MSU)

## Notes
- All 22 materials ordered at the scope DCs are HPFD-line materials (network fully HPFD-scoped).
- Change-over id '2-1' has 0 events in this run; '2-11' is present (see deep-dive note).
- APQ note: material(s) 80799574 are the line's base product (already mounted at sim start), so their campaigns carry no logged change-over but ARE counted as runs via the contiguous-campaign rule (2 base-product campaign(s) in scope, e.g. 80799574 = Feb 10-11 + Mar 12). They carry 90.7 MSU (44% of volume). Headline APQ uses 35 runs = 33 change-over-started + 2 base-product campaign(s); change-over-started runs alone average ~3.426 MSU/run.
- DFC month-end dates: Mar 2026=2026-03-31, Feb 2026=2026-02-28; both months have full 30-day forward windows.

## Outputs
- workbook: fem-cs-test_result_summary_20260622-2051-fem-kpi.xlsx
- report: analysis.html
- extracts/: per-KPI CSVs

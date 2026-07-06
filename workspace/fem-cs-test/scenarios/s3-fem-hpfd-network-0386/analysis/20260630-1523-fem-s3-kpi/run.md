# FEM S3 KPI analysis - run 20260630-1523-fem-s3-kpi

- level: scenario
- project: fem-cs-test
- scenario: s3-fem-hpfd-network-0386
- run_id: db_s3-fem-hpfd-network-0386_20260630_142620
- database: fem_test
- generated: 2026-06-30 15:23

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
- ToF Feb 2026: 98.9% (162,499/164,342)
- ToF Mar 2026: 98.3% (114,467/116,478)
- DFC Feb 2026 (all DCs): 35.3 days (inv 134,261 / 3,807.7 per day)
- DFC Mar 2026 (all DCs): 41.9 days (inv 75,176 / 1,794.0 per day)
- Change-over Feb 2026: 19 events, 17.5 hr
- Change-over Mar 2026: 14 events, 11.5 hr
- Production total time Feb 2026: 90.8 hr (prod 73.3 + CO 17.5; CO share 19.3%)
- Production total time Mar 2026: 65.6 hr (prod 54.1 + CO 11.5; CO share 17.5%)
- APQ overall: 5.843 MSU/run (total MSU 204.5 / 35 runs)
- PKG CO/MSU: 0.1516 (31 CO / 204.5 MSU)
- CONV CO/MSU: 0.0440 (9 CO / 204.5 MSU)

## Notes
- All 22 materials ordered at the scope DCs are HPFD-line materials (network fully HPFD-scoped).
- Change-over note: classified change-over id(s) '2-1' absent (0 events).
- APQ note: material(s) 80799574 are the line's base product (already mounted at sim start), so their campaigns carry no logged change-over but ARE counted as runs via the contiguous-campaign rule (2 base-product campaign(s) in scope). They carry 90.6 MSU (44% of volume). Headline APQ uses 35 runs = 33 change-over-started + 2 base-product campaign(s); change-over-started runs alone average ~3.451 MSU/run.
- DFC month-end dates: Mar 2026=2026-03-31, Feb 2026=2026-02-28; both months have full 30-day forward windows.

## Outputs
- workbook: fem-cs-test_result_summary_20260630-1523-fem-s3-kpi.xlsx
- report: analysis.html
- extracts/: per-KPI CSVs

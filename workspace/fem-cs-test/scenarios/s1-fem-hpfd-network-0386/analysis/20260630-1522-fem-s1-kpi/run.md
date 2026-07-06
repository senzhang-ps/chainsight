# FEM S1 KPI analysis - run 20260630-1522-fem-s1-kpi

- level: scenario
- project: fem-cs-test
- scenario: s1-fem-hpfd-network-0386
- run_id: db_s1-fem-hpfd-network-0386_20260630_142419
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
- ToF Feb 2026: 98.8% (166,451/168,563)
- ToF Mar 2026: 96.6% (109,602/113,464)
- DFC Feb 2026 (all DCs): 40.6 days (inv 154,765 / 3,807.7 per day)
- DFC Mar 2026 (all DCs): 27.1 days (inv 48,669 / 1,794.0 per day)
- Change-over Feb 2026: 20 events, 18.0 hr
- Change-over Mar 2026: 14 events, 11.5 hr
- Production total time Feb 2026: 111.9 hr (prod 93.9 + CO 18.0; CO share 16.1%)
- Production total time Mar 2026: 50.2 hr (prod 38.7 + CO 11.5; CO share 22.9%)
- APQ overall: 5.844 MSU/run (total MSU 210.4 / 36 runs)
- PKG CO/MSU: 0.1521 (32 CO / 210.4 MSU)
- CONV CO/MSU: 0.0428 (9 CO / 210.4 MSU)

## Notes
- All 22 materials ordered at the scope DCs are HPFD-line materials (network fully HPFD-scoped).
- Change-over note: classified change-over id(s) '2-1' absent (0 events).
- APQ note: material(s) 80799574 are the line's base product (already mounted at sim start), so their campaigns carry no logged change-over but ARE counted as runs via the contiguous-campaign rule (2 base-product campaign(s) in scope). They carry 87.2 MSU (41% of volume). Headline APQ uses 36 runs = 34 change-over-started + 2 base-product campaign(s); change-over-started runs alone average ~3.624 MSU/run.
- DFC month-end dates: Mar 2026=2026-03-31, Feb 2026=2026-02-28; both months have full 30-day forward windows.

## Outputs
- workbook: fem-cs-test_result_summary_20260630-1522-fem-s1-kpi.xlsx
- report: analysis.html
- extracts/: per-KPI CSVs

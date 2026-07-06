# FEM KPI comparison - baseline vs S1/S2/S3 - run 20260630-1530-fem-compare

- level: cross-scenario comparison
- project: fem-cs-test
- database: fem_test
- line: HPFD @ plant 0386 | months: Feb 2026, Mar 2026
- generated: 2026-06-30 15:30

## Scenarios
- Baseline: `baseline-fem-hpfd-network-0386` | run_id `db_baseline-fem-hpfd-network-0386_20260602_223559` | lever: reference (no lever)
- S1: `s1-fem-hpfd-network-0386` | run_id `db_s1-fem-hpfd-network-0386_20260630_142419` | lever: review cycle lengthened: M4_MaterialLocationLineCfg.lsk 30→45 (19 rows), 60→90 (7 rows)
- S2: `s2-fem-hpfd-network-0386` | run_id `db_s2-fem-hpfd-network-0386_20260630_142529` | lever: line capacity cut: M4_LineCapacity.capacity 24→16 h/day (all 91 days)
- S3: `s3-fem-hpfd-network-0386` | run_id `db_s3-fem-hpfd-network-0386_20260630_142620` | lever: AO profile standardized: M1_AOConfig advance-days 0d=75% / 5d=5% / 15d=20%

## Headline (Baseline / S1 / S2 / S3)
- ToF Feb %: 98.8 / 98.8 / 98.7 / 98.9
- ToF Mar %: 98.2 / 96.6 / 97.9 / 98.3
- DFC Feb (days): 34.9 / 40.6 / 35.0 / 35.3
- DFC Mar (days): 42.8 / 27.1 / 42.6 / 41.9
- CO count Feb: 19 / 20 / 19 / 19
- CO count Mar: 14 / 14 / 14 / 14
- Prod total time Feb (hr): 90.4 / 111.9 / 90.4 / 90.8
- Prod total time Mar (hr): 65.6 / 50.2 / 64.2 / 65.6
- APQ (MSU/run): 5.822 / 5.844 / 5.769 / 5.843
- PKG CO/MSU: 0.1521 / 0.1521 / 0.1535 / 0.1516
- CONV CO/MSU: 0.0442 / 0.0428 / 0.0446 / 0.0440

## Reading
- S1 (longer review cycle): front-loads inventory in Feb, runs it down by end-Mar; March service is the lowest of the four; Feb production time rises.
- S2 (capacity 24->16 h/day): tracks the baseline on every KPI -> capacity is not the binding constraint in this network/period.
- S3 (AO standardized): best March service, all other KPIs close to baseline.
- Change-over count/time, APQ and CO/MSU are effectively unchanged across all scenarios.

## Source
- Baseline: workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/analysis/20260622-2051-fem-kpi
- S1: workspace/fem-cs-test/scenarios/s1-fem-hpfd-network-0386/analysis/20260630-1522-fem-s1-kpi
- S2: workspace/fem-cs-test/scenarios/s2-fem-hpfd-network-0386/analysis/20260630-1523-fem-s2-kpi
- S3: workspace/fem-cs-test/scenarios/s3-fem-hpfd-network-0386/analysis/20260630-1523-fem-s3-kpi

## Outputs
- report: comparison.html
- workbook: fem-cs-test_comparison_20260630-1530-fem-compare.xlsx
- extracts/: per-KPI comparison CSVs

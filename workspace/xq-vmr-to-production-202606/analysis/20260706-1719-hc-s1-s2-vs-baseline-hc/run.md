# HC AO-variant comparison — xq-vmr-to-production-202606

- generated: 2026-07-06 17:19
- benchmark: `baseline-hc`
- variants: `baseline-hc-s1`, `baseline-hc-s2`
- scope: HC only, plant XQ (1864), horizon ≤ 2026-10

## Runs

| scenario | run_id | AO configuration |
|---|---|---|
| baseline-hc | `db_baseline-hc_20260617_203548` | Empirical per-key AO distribution (benchmark) |
| baseline-hc-s1 | `db_baseline-hc-s1_20260706_103801` | Standardized AO, moderate advancing: 0d=45%, 3d=5%, 8d=25%, 12d=25% |
| baseline-hc-s2 | `db_baseline-hc-s2_20260706_110742` | Standardized AO, aggressive advancing: 0d=15%, 3d=5%, 8d=40%, 12d=40% |

## Headline deltas vs baseline-hc

| KPI | baseline-hc | s1 | s2 |
|---|---|---|---|
| Overall TOF | 97.7% | 98.2% | 98.4% |
| DFC full-month avg (days) | 20.4 | 21.6 | 23.1 |
| Changeover count | 2,208 | 2,206 | 2,194 |
| Production time (hr) | 3,716 | 3,708 | 3,677 |
| APQ (MSU/run) | 2.939 | 2.930 | 2.935 |
| Production runs | 2,208 | 2,206 | 2,194 |
| Order-log rows | 1,346,710 | 2,298,035 | 2,298,035 |

See `analysis.html` for the full comparison with charts and the workbook for all extract tables.

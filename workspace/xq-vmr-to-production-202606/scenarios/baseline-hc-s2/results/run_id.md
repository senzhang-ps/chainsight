# baseline-hc-s2 — run id record

| Field | Value |
| --- | --- |
| Project | xq-vmr-to-production-202606 |
| Scenario | baseline-hc-s2 |
| Workspace config folder | workspace/xq-vmr-to-production-202606/scenarios/baseline-hc-s2/config |
| Runner config folder | input/xq-vmr-to-production-202606/scenarios/baseline-hc-s2/config |
| Active workbook | baseline-hc-s2.xlsx |
| Command | `python run.py --config-dir input/xq-vmr-to-production-202606/scenarios/baseline-hc-s2 --start-date 2026-06-29 --end-date 2026-11-01 --use-db --non-interactive` (launched headless via Start-Process, stdout/stderr redirected) |
| --use-db | yes |
| --force-restart | no (fresh run, no prior) |
| Output run folder | outputs/xq-vmr-to-production-202606/baseline-hc-s2/db_run_20260706_110742 |
| db_run_id.txt path | outputs/xq-vmr-to-production-202606/baseline-hc-s2/db_run_20260706_110742/db_run_id.txt |
| Confirmed run id | db_baseline-hc-s2_20260706_110742 |
| Trigger time | 2026-07-06 11:07:37 |
| Run period | 2026-06-29 to 2026-11-01 (126 days) |

## Notes

- Lever vs `baseline-hc`: only `M1_AOConfig` changed. s2 replaces the empirical
  per-key advance-order distribution with one standardized, more advance-shifted
  profile applied to all 2177 material-location keys: advance_days 0=0.15, 3=0.05,
  8=0.40, 12=0.40.
- Pre-run fix: `M1_AOConfig.ao_percent` was stored as percentage strings
  (`40%`, `5%`, `15%`), which the M1 order generator's `avg_daily_demand *
  ao_percent` cannot parse (`TypeError: can't multiply sequence by non-int of type
  'float'`). Converted to decimal fractions (`0.40`, `0.05`, `0.15`) in the
  workspace config and re-synced to the runner input before launch.

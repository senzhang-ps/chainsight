# baseline-hc-s1 — run id record

| Field | Value |
| --- | --- |
| Project | xq-vmr-to-production-202606 |
| Scenario | baseline-hc-s1 |
| Workspace config folder | workspace/xq-vmr-to-production-202606/scenarios/baseline-hc-s1/config |
| Runner config folder | input/xq-vmr-to-production-202606/scenarios/baseline-hc-s1/config |
| Active workbook | baseline-hc-s1.xlsx |
| Command | `python run.py --config-dir input/xq-vmr-to-production-202606/scenarios/baseline-hc-s1 --start-date 2026-06-29 --end-date 2026-11-01 --use-db --non-interactive` (launched headless via Start-Process, stdout/stderr redirected) |
| --use-db | yes |
| --force-restart | no (resumed unfinished run) |
| Output run folder | outputs/xq-vmr-to-production-202606/baseline-hc-s1/db_run_20260706_110657 (resume launch; original db_run_20260706_103801) |
| db_run_id.txt path | outputs/xq-vmr-to-production-202606/baseline-hc-s1/db_run_20260706_110657/db_run_id.txt |
| Confirmed run id | db_baseline-hc-s1_20260706_103801 |
| Trigger time | 2026-07-06 11:06:49 |
| Run period | 2026-06-29 to 2026-11-01 (126 days) |

## Notes

- Lever vs `baseline-hc`: only `M1_AOConfig` changed. s1 replaces the empirical
  per-key advance-order distribution with one standardized profile applied to all
  2177 material-location keys: advance_days 0=0.45, 3=0.05, 8=0.25, 12=0.25.
- Run id `db_baseline-hc-s1_20260706_103801` is reused from an earlier (terminated)
  run; the `db_run_20260706_110657` folder is the resume launch. Use the run id, not
  the folder timestamp, as the DB result filter key.

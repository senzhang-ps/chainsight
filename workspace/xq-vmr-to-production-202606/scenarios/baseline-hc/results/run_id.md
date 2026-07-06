# baseline-hc — run id record

| Field | Value |
| --- | --- |
| Project | xq-vmr-to-production-202606 |
| Scenario | baseline-hc |
| Workspace config folder | workspace/xq-vmr-to-production-202606/scenarios/baseline-hc/config |
| Runner config folder | input/xq-vmr-to-production-202606/scenarios/baseline-hc/config |
| Active workbook | baseline-hc.xlsx |
| Command | `python run.py --config-dir input/xq-vmr-to-production-202606/scenarios/baseline-hc --start-date 2026-06-29 --end-date 2026-11-01 --use-db --non-interactive` |
| --use-db | yes |
| --force-restart | no (resumed unfinished run) |
| Output run folder | outputs/xq-vmr-to-production-202606/baseline-hc/db_run_20260617_204701 |
| db_run_id.txt path | outputs/xq-vmr-to-production-202606/baseline-hc/db_run_20260617_204701/db_run_id.txt |
| Confirmed run id | db_baseline-hc_20260617_203548 |
| Trigger time | 2026-06-17 20:46:58 |
| Run period | 2026-06-29 to 2026-11-01 (126 days) |

## Notes

- Run id `db_baseline-hc_20260617_203548` is reused from the original (terminated)
  run; the new `db_run_20260617_204701` folder is the resume launch. Use the run
  id, not the folder timestamp, as the DB result filter key.
- Pre-run fix: `M4_ProductionReliability.pr` was stored as percentage strings
  (`97%`, `85%`), which the engine's `binomial(qty, pr)` cannot parse
  (`could not convert string to float: '85%'`). Converted to decimal fractions
  (`0.97`, `0.85`) in the workspace config and re-synced to the runner input.

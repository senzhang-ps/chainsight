# Simulation Run Record

## Latest confirmed run
- trigger time: 2026-06-30 14:24:19
- run mode: normal run (fresh start, `[NEW]` — not resumed)
- launched by: agent (chainsight-simulation-run-windows)

## Scenario
- project: `fem-cs-test`
- scenario: `s1-fem-hpfd-network-0386`

## Confirmed latest run id
- `db_s1-fem-hpfd-network-0386_20260630_142419`

## Confirmed database name
- `fem_test`

## Run configuration
- workspace config folder: `workspace/fem-cs-test/scenarios/s1-fem-hpfd-network-0386/config/`
- runner config folder: `input/fem-cs-test/scenarios/s1-fem-hpfd-network-0386/config/`
- active workbook: `s1-fem-hpfd-network-0386.xlsx`
- run period: 2026-02-02 to 2026-05-03 (91 days)
- use-db: yes
- force-restart: no
- output run folder: `outputs/fem-cs-test/s1-fem-hpfd-network-0386/db_run_20260630_142419/`
- run-id file: `outputs/fem-cs-test/s1-fem-hpfd-network-0386/db_run_20260630_142419/db_run_id.txt`

## Command used
```
.\.venv\Scripts\python.exe run.py --config-dir input/fem-cs-test/scenarios/s1-fem-hpfd-network-0386 --start-date 2026-02-02 --end-date 2026-05-03 --use-db --db-name fem_test --non-interactive
```

## Scenario lever (vs baseline)
- `M4_MaterialLocationLineCfg.lsk`: 30->45 (19 rows), 60->90 (7 rows); all other sheets identical to baseline.

## Notes
- Config first-imported into `fem_test` (27 tables) then run from DB.
- Comparison target: `baseline-fem-hpfd-network-0386`.
- Use the run id above for result lookup and analysis.

## Scope

- Network: full network for Line M/Y/C sourcing SKUs
- Period: 2026-05-04 to 2026-06-19
- BOP version: Apr 27 LBE 50/50
- Initial inventory: Apr 27 actual stock on hand + in-transit

## Run Execution

- Windows simulation codebase: `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight`
- Workspace config artifact: `workspace/test-case/scenarios/scenario-1/config/scenario-1.xlsx`
- Runner config artifact: `config/scenario-1.xlsx`
- Command used: `./.venv/Scripts/python.exe run.py --config config/scenario-1.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db`
- Run dates: 2026-05-04 to 2026-06-19
- Output location: `outputs/db_scenario-1_20260510_170527/`
- Launch status: started on 2026-05-10 17:05:27; process remains active in terminal `571b73a3-42bd-43c0-9040-2b3eea1319b0`
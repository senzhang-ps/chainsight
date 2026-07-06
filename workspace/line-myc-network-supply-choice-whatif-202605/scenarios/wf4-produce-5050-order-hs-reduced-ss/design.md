# Scenario: What-if 4

## Goal Link

Compare a 50/50 production choice against a high-side ordering policy
when DTC safety stock is reduced from 30 days to 15 days.

## Scenario Intent

Test the most order-side aggressive variant in the set while also
reducing DTC safety stock, to expose downside risk and any inventory
relief opportunity.

## Parameters / Levers

- Produce policy: 50/50
- Order policy: high-side x reduced safety stock
- Safety stock policy: DTC locations reduce from 30 days to 15 days
- High-side delta rule: maintain delta in supply choice
- Forecast error CoV: follow iBPI
- AO: 0%

## Expected Trade-offs

- Maximum pressure test on the combination of leaner DTC inventory and a
  stronger order signal
- Could surface both lower inventory and a higher stockout / instability risk
- Important contrast against what-if 2 to isolate the safety stock effect

## Scope

- Network: full network for Line M/Y/C sourcing SKUs
- Period: 2026-05-04 to 2026-06-19
- BOP version: Apr 27 LBE 50/50
- Initial inventory: Apr 27 actual stock on hand + in-transit

## Data Plan

- Covered by skill: sourcing network extraction for Line M/Y/C SKUs,
  demand forecast inputs for the 50/50 BOP version, safety stock inputs
  with DTC override from 30 days to 15 days, ordering config exports
  needed to represent the high-side order policy
- Manual / unresolved: exact config workbook sheet mapping, simulation
  runner path, and any scenario identifier required by the external sim
  codebase
- Config target: assemble workbook at
  `workspace/line-myc-network-supply-choice-whatif-202605/deliverables/wf4-produce-5050-order-hs-reduced-ss.xlsx`

## Validation Gate

- Scope confirmed from user prompt
- Validate config before run: passed with warnings. See
  `config/wf4-produce-5050-order-hs-reduced-ss-config-validation.md`.
- Run gate: satisfied. The validated deliverable workbook was copied to
  the Windows simulation codebase and wf4 launch was issued in database mode.

## Output Paths

- Scope file:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf4-produce-5050-order-hs-reduced-ss/scope.yaml`
- Config directory:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf4-produce-5050-order-hs-reduced-ss/config/`
- Results directory:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf4-produce-5050-order-hs-reduced-ss/results/`
- Analysis file:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf4-produce-5050-order-hs-reduced-ss/analysis.md`

## Run Execution

- Windows codebase:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight`
- Config artifact used:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\config\wf4-produce-5050-order-hs-reduced-ss.xlsx`
- Command used:
  `./.venv/Scripts/python.exe run.py --config config/wf4-produce-5050-order-hs-reduced-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive`
- Start date: `2026-05-04`
- End date: `2026-06-19`
- Output location:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf4-produce-5050-order-hs-reduced-ss_20260511_160934`
- Run status:
  started; output directory has been created and the process remains active

## Rerun Execution

- Rerun command used:
  `./.venv/Scripts/python.exe run.py --config config/wf4-produce-5050-order-hs-reduced-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive --force-restart`
- Rerun issued at: `2026-05-11 17:53:56`
- Rerun terminal session ID: `2912a2a0-7c04-4380-8d4d-28d4afa1cad6`
- Latest tracked output location:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf4-produce-5050-order-hs-reduced-ss_20260511_172934`
- Rerun status:
  active; fresh process launched with `--force-restart`, and the latest
  observed scenario log is still under the most recent scenario-specific
  output directory above

  ## Rerun Execution 2026-05-12

  - Rerun command used:
    `./.venv/Scripts/python.exe run.py --config config/wf4-produce-5050-order-hs-reduced-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive --force-restart`
  - Rerun issued at: `2026-05-12 19:38:10`
  - Rerun terminal session ID: `3c7a5d84-9b0e-4761-8409-622da1f8824e`
  - WSL process ID: `727525`
  - Latest tracked output location:
    not yet observed under the Windows outputs path as of `2026-05-12 19:40:04`
  - Rerun status:
    active; a fresh process is running and visible in the WSL process
    list, but the scenario-specific output directory has not yet been
    observed
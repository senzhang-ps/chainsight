# Scenario: What-if 3

## Goal Link

Compare a high-side production choice against a 50/50 ordering policy
when DTC safety stock is reduced from 30 days to 15 days.

## Scenario Intent

Test whether the network can tolerate reduced DTC safety stock when the
produce side still follows the high-side choice.

## Parameters / Levers

- Produce policy: high-side
- Order policy: 50/50 forecast x reduced safety stock
- Safety stock policy: DTC locations reduce from 30 days to 15 days
- High-side delta rule: maintain delta in supply choice
- Forecast error CoV: follow iBPI
- AO: 0%

## Expected Trade-offs

- Inventory relief from lower DTC safety stock
- Higher service and replenishment risk if reduced buffers are not offset
  by the high-side produce choice
- Important contrast against what-if 1 to isolate the safety stock effect

## Scope

- Network: full network for Line M/Y/C sourcing SKUs
- Period: 2026-05-04 to 2026-06-19
- BOP version: Apr 27 LBE 50/50
- Initial inventory: Apr 27 actual stock on hand + in-transit

## Data Plan

- Covered by skill: sourcing network extraction for Line M/Y/C SKUs,
  demand forecast inputs for the 50/50 BOP version, safety stock inputs
  with DTC override from 30 days to 15 days, production config exports
  needed to represent the high-side supply choice delta
- Manual / unresolved: exact config workbook sheet mapping, simulation
  runner path, and any scenario identifier required by the external sim
  codebase
- Config target: assemble workbook at
  `workspace/line-myc-network-supply-choice-whatif-202605/deliverables/wf3-produce-hs-order-5050-reduced-ss.xlsx`

## Validation Gate

- Scope confirmed from user prompt
- Validate config before run: passed with warnings. See
  `config/wf3-produce-hs-order-5050-reduced-ss-config-validation.md`.
- Run gate: satisfied. The validated deliverable workbook was copied to
  the Windows simulation codebase and wf3 was started in database mode.

## Output Paths

- Scope file:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf3-produce-hs-order-5050-reduced-ss/scope.yaml`
- Config directory:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf3-produce-hs-order-5050-reduced-ss/config/`
- Results directory:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf3-produce-hs-order-5050-reduced-ss/results/`
- Analysis file:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf3-produce-hs-order-5050-reduced-ss/analysis.md`

## Run Execution

- Windows codebase:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight`
- Config artifact used:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\config\wf3-produce-hs-order-5050-reduced-ss.xlsx`
- Command used:
  `./.venv/Scripts/python.exe run.py --config config/wf3-produce-hs-order-5050-reduced-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive`
- Start date: `2026-05-04`
- End date: `2026-06-19`
- Output location:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf3-produce-hs-order-5050-reduced-ss_20260511_160859`
- Run status:
  started; output directory has been created and the process remains active

## Rerun Execution

- Rerun command used:
  `./.venv/Scripts/python.exe run.py --config config/wf3-produce-hs-order-5050-reduced-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive --force-restart`
- Rerun issued at: `2026-05-11 17:53:51`
- Rerun terminal session ID: `e0be05e4-7500-4a52-b0b2-fc63ac14b3f1`
- Latest tracked output location:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf3-produce-hs-order-5050-reduced-ss_20260511_172739`
- Rerun status:
  active; fresh process launched with `--force-restart`, and the latest
  observed scenario log is still under the most recent scenario-specific
  output directory above

  ## Rerun Execution 2026-05-12

  - Rerun command used:
    `./.venv/Scripts/python.exe run.py --config config/wf3-produce-hs-order-5050-reduced-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive --force-restart`
  - Rerun issued at: `2026-05-12 19:37:59`
  - Rerun terminal session ID: `ada45ff7-df5b-431b-899f-08afbb87ad12`
  - WSL process ID: `727435`
  - Latest tracked output location:
    not yet observed under the Windows outputs path as of `2026-05-12 19:40:04`
  - Rerun status:
    active; a fresh process is running and visible in the WSL process
    list, but the scenario-specific output directory has not yet been
    observed
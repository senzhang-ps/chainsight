# Scenario: What-if 2

## Goal Link

Compare a 50/50 production choice against a high-side ordering policy
under current safety stock for the full Line M/Y/C sourcing network.

## Scenario Intent

Hold production closer to the 50/50 baseline while increasing order-side
pressure through the high-side choice and current safety stock.

## Parameters / Levers

- Produce policy: 50/50
- Order policy: high-side x current safety stock
- Safety stock policy: current safety stock following iBPI after override
- High-side delta rule: maintain delta in supply choice
- Forecast error CoV: follow iBPI
- AO: 0%

## Expected Trade-offs

- Potentially lower production-side build-up than what-if 1
- Potential order amplification risk if the high-side order signal is too
  aggressive for the current network state
- Useful contrast against what-if 1 to isolate whether risk sits more on
  the produce side or the order side

## Scope

- Network: full network for Line M/Y/C sourcing SKUs
- Period: 2026-05-04 to 2026-06-19
- BOP version: Apr 27 LBE 50/50
- Initial inventory: Apr 27 actual stock on hand + in-transit

## Data Plan

- Covered by skill: sourcing network extraction for Line M/Y/C SKUs,
  demand forecast inputs for the 50/50 BOP version, safety stock inputs
  after iBPI override, production and ordering config exports needed to
  represent the high-side order policy
- Manual / unresolved: exact config workbook sheet mapping, simulation
  runner path, and any scenario identifier required by the external sim
  codebase
- Config target: assemble workbook at
  `workspace/line-myc-network-supply-choice-whatif-202605/deliverables/wf2-produce-5050-order-hs-current-ss.xlsx`

## Validation Gate

- Scope confirmed from user prompt
- Validate config before run: passed with warnings. See
  `config/wf2-produce-5050-order-hs-current-ss-config-validation.md`.
- Run gate: satisfied. The validated deliverable workbook was copied to
  the Windows simulation codebase and wf2 was started in database mode.

## Output Paths

- Scope file:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf2-produce-5050-order-hs-current-ss/scope.yaml`
- Config directory:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf2-produce-5050-order-hs-current-ss/config/`
- Results directory:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf2-produce-5050-order-hs-current-ss/results/`
- Analysis file:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf2-produce-5050-order-hs-current-ss/analysis.md`

## Run Execution

- Windows codebase:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight`
- Config artifact used:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\config\wf2-produce-5050-order-hs-current-ss.xlsx`
- Command used:
  `./.venv/Scripts/python.exe run.py --config config/wf2-produce-5050-order-hs-current-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive`
- Start date: `2026-05-04`
- End date: `2026-06-19`
- Output location:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf2-produce-5050-order-hs-current-ss_20260511_160735`
- Run status:
  started; startup log initialized at `2026-05-11 16:07:35`

## Rerun Execution

- Rerun command used:
  `./.venv/Scripts/python.exe run.py --config config/wf2-produce-5050-order-hs-current-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive --force-restart`
- Rerun issued at: `2026-05-11 17:53:46`
- Rerun terminal session ID: `d2a1440f-8d20-44c7-902c-71b2f0adc961`
- Latest tracked output location:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf2-produce-5050-order-hs-current-ss_20260511_172738`
- Rerun status:
  active; fresh process launched with `--force-restart`, and the latest
  observed scenario log is still under the most recent scenario-specific
  output directory above

  ## Rerun Execution 2026-05-12

  - Rerun command used:
    `./.venv/Scripts/python.exe run.py --config config/wf2-produce-5050-order-hs-current-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive --force-restart`
  - Rerun issued at: `2026-05-12 19:37:49`
  - Rerun terminal session ID: `8d413427-b9e8-40df-8e8e-b8bb9a6df0b4`
  - WSL process ID: `727340`
  - Latest tracked output location:
    not yet observed under the Windows outputs path as of `2026-05-12 19:40:04`
  - Rerun status:
    active; a fresh process is running and visible in the WSL process
    list, but the scenario-specific output directory has not yet been
    observed
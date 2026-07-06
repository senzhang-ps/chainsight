# Scenario: What-if 1

## Goal Link

Compare a high-side production choice against a 50/50 ordering policy
under current safety stock for the full Line M/Y/C sourcing network.

## Scenario Intent

Stress the network with more aggressive production while keeping order
creation anchored to the 50/50 forecast and current safety stock.

## Parameters / Levers

- Produce policy: high-side
- Order policy: 50/50 forecast x current safety stock
- Safety stock policy: current safety stock following iBPI after override
- High-side delta rule: maintain delta in supply choice
- Forecast error CoV: follow iBPI
- AO: 0%

## Expected Trade-offs

- Potential service upside from higher production bias
- Potential inventory build if demand realization stays closer to 50/50
- Need to observe whether maintaining current safety stock dampens or
  amplifies the inventory impact

## Scope

- Network: full network for Line M/Y/C sourcing SKUs
- Period: 2026-05-04 to 2026-06-19
- BOP version: Apr 27 LBE 50/50
- Initial inventory: Apr 27 actual stock on hand + in-transit

## Data Plan

- Covered by skill: sourcing network extraction for Line M/Y/C SKUs,
  demand forecast inputs for the 50/50 BOP version, safety stock inputs
  after iBPI override, production config exports needed to represent the
  high-side supply choice delta
- Manual / unresolved: exact config workbook sheet mapping, simulation
  runner path, and any scenario identifier required by the external sim
  codebase
- Config target: assemble workbook at
  `workspace/line-myc-network-supply-choice-whatif-202605/deliverables/wf1-produce-hs-order-5050-current-ss.xlsx`

## Validation Gate

- Scope confirmed from user prompt
- Validate config before run: passed with warnings. See
  `config/wf1-produce-hs-order-5050-current-ss-config-validation.md`.
- Run gate: satisfied. The validated deliverable workbook was copied to
  the Windows simulation codebase and wf1 was started in database mode.

## Output Paths

- Scope file:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf1-produce-hs-order-5050-current-ss/scope.yaml`
- Config directory:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf1-produce-hs-order-5050-current-ss/config/`
- Results directory:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf1-produce-hs-order-5050-current-ss/results/`
- Analysis file:
  `workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf1-produce-hs-order-5050-current-ss/analysis.md`

## Run Execution

- Windows codebase:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight`
- Config artifact used:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\config\wf1-produce-hs-order-5050-current-ss.xlsx`
- Command used:
  `./.venv/Scripts/python.exe run.py --config config/wf1-produce-hs-order-5050-current-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive`
- Start date: `2026-05-04`
- End date: `2026-06-19`
- Output location:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf1-produce-hs-order-5050-current-ss_20260511_160207`
- Run status:
  started; initialization log confirms the runner entered config sync at
  `2026-05-11 16:03:08`

## Rerun Execution

- Rerun command used:
  `./.venv/Scripts/python.exe run.py --config config/wf1-produce-hs-order-5050-current-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive --force-restart`
- Rerun issued at: `2026-05-11 17:53:40`
- Rerun terminal session ID: `2af3940e-e3b9-4cad-a172-a9a608fcec0d`
- Latest tracked output location:
  `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf1-produce-hs-order-5050-current-ss_20260511_172659`
- Rerun status:
  active; fresh process launched with `--force-restart`, and the latest
  observed scenario log is still under the most recent scenario-specific
  output directory above

  ## Rerun Execution 2026-05-12

  - Rerun command used:
    `./.venv/Scripts/python.exe run.py --config config/wf1-produce-hs-order-5050-current-ss.xlsx --start-date 2026-05-04 --end-date 2026-06-19 --use-db --non-interactive --force-restart`
  - Rerun issued at: `2026-05-12 19:36:53`
  - Rerun terminal session ID: `c0ac4f14-7327-4395-bb67-9145e999e531`
  - WSL process ID: `727082`
  - Latest tracked output location:
    `C:\Users\zhang.s.37\OneDrive - Procter and Gamble\Documents\GitHub\chainsight\outputs\db_wf1-produce-hs-order-5050-current-ss_20260512_193852`
  - Rerun status:
    active; a fresh process is running and a new scenario-specific output
    directory has already been observed under the Windows outputs path
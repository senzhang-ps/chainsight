# Scenario: S1 FEM HPFD Network 0386

## Goal Link

Run what-if scenario S1 for the FEM category on the HPFD production line
across the full sourcing network of network 0386, and compare its behavior
against the confirmed baseline (`baseline-fem-hpfd-network-0386`) on the same
KPI set and period.

## Scenario Intent

Lengthen the replenishment review frequency on the HPFD line by raising the
Lot Size Key (`lsk`, which defines the review-frequency interval) in
`M4_MaterialLocationLineCfg`. The scenario tests how a coarser / less
frequent review cycle affects the KPI set versus the baseline. Verified by
diffing `config/s1-fem-hpfd-network-0386.xlsx` against the baseline workbook:
this is the only sheet that changed.

## Parameters / Levers

- Scenario mode: what-if (variant of baseline)
- Parameter basis: baseline settings with S1 overrides
- Specific parameter changes (sheet `M4_MaterialLocationLineCfg`, column
  `lsk` = lot size key / review frequency):
  - `lsk` 30 -> 45 (19 material-location rows)
  - `lsk` 60 -> 90 (7 material-location rows)
  - `lsk` 90 -> 90 (3 rows unchanged)
- All other sheets are byte-for-content identical to the baseline workbook
- Category: FEM
- Production line: HPFD
- Network: full sourcing network associated with network 0386
- Time window: 2026-02-02 to 2026-05-03

## Expected Trade-offs

- Results are interpreted as deltas versus the baseline scenario
- Primary evaluation focuses on CFR (customer fulfillment rate), DFC
  performance, CU (capacity utilization), and changeover count and
  percentage by changeover type
- Because no explicit KPI threshold is defined yet, business judgment is
  required to decide whether the S1 change is favorable

## Scope

- Category scope: FEM
- Line scope: HPFD
- Network scope: entire sourcing network for network 0386 related to the
  HPFD line
- Time period: 2026-02-02 to 2026-05-03
- Geography / plant scope: inherits the full 0386 sourcing network from the
  baseline; exact node list not yet enumerated in the workspace
- Run mode: single what-if scenario

## Data Plan

- Config workbook present at
  `config/s1-fem-hpfd-network-0386.xlsx`
- Inputs should match the baseline study except for the S1 levers
- Actual-operation / baseline reference data is required for the same period
  to compute deltas on CFR, DFC, CU, and changeover metrics
- Exact simulation runner path, command, and output schema are
  user_not_provided

## Validation Gate

- Scenario intent confirmed from user prompt: directory scaffold only
- Config workbook supplied: yes
- Parameter-change list extracted and reconciled vs baseline: extracted
  (single lever isolated: `M4_MaterialLocationLineCfg.lsk`)
- Exact KPI list for validation: confirmed (CFR, DFC, CU, changeover
  count/percentage by type)
- Acceptance threshold: TBD
- Config package readiness: config present, validation not run
- Simulation run readiness: blocked until config validation and runner
  details are supplied

## Output Paths

- Scope file:
  `workspace/fem-cs-test/scenarios/s1-fem-hpfd-network-0386/scope.yaml`
- Config directory:
  `workspace/fem-cs-test/scenarios/s1-fem-hpfd-network-0386/config/`
- Results directory:
  `workspace/fem-cs-test/scenarios/s1-fem-hpfd-network-0386/results/`
- Analysis directory:
  `workspace/fem-cs-test/scenarios/s1-fem-hpfd-network-0386/analysis/`

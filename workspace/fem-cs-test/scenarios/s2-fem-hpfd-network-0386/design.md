# Scenario: S2 FEM HPFD Network 0386

## Goal Link

Run what-if scenario S2 for the FEM category on the HPFD production line
across the full sourcing network of network 0386, and compare its behavior
against the confirmed baseline (`baseline-fem-hpfd-network-0386`) on the same
KPI set and period.

## Scenario Intent

Reduce the available daily production capacity on the HPFD line
(capacity-down what-if) for the full simulation window, to test how a tighter
line-hours constraint affects the KPI set versus the baseline. Verified by
diffing `config/s2-fem-hpfd-network-0386.xlsx` against the baseline workbook:
this is the only sheet that changed.

## Parameters / Levers

- Scenario mode: what-if (variant of baseline)
- Parameter basis: baseline settings with S2 overrides
- Specific parameter changes (sheet `M4_LineCapacity`, column `capacity` =
  production-line capacity in hours/day):
  - `capacity` 24 -> 16 hours/day for all 91 daily rows
    (location 0386, line HPFD, 2026-02-02 to 2026-05-03)
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
  required to decide whether the S2 change is favorable

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
  `config/s2-fem-hpfd-network-0386.xlsx`
- Inputs should match the baseline study except for the S2 levers
- Actual-operation / baseline reference data is required for the same period
  to compute deltas on CFR, DFC, CU, and changeover metrics
- Exact simulation runner path, command, and output schema are
  user_not_provided

## Validation Gate

- Scenario intent confirmed from user prompt: directory scaffold only
- Config workbook supplied: yes
- Parameter-change list extracted and reconciled vs baseline: extracted
  (single lever isolated: `M4_LineCapacity.capacity`)
- Exact KPI list for validation: confirmed (CFR, DFC, CU, changeover
  count/percentage by type)
- Acceptance threshold: TBD
- Config package readiness: config present, validation not run
- Simulation run readiness: blocked until config validation and runner
  details are supplied

## Output Paths

- Scope file:
  `workspace/fem-cs-test/scenarios/s2-fem-hpfd-network-0386/scope.yaml`
- Config directory:
  `workspace/fem-cs-test/scenarios/s2-fem-hpfd-network-0386/config/`
- Results directory:
  `workspace/fem-cs-test/scenarios/s2-fem-hpfd-network-0386/results/`
- Analysis directory:
  `workspace/fem-cs-test/scenarios/s2-fem-hpfd-network-0386/analysis/`

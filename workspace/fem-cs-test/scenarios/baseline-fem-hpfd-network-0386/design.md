# Scenario: Baseline FEM HPFD Network 0386

## Goal Link

Create a baseline simulation for the FEM category on the HPFD production
line across the full sourcing network of network 0386, so the business team
can compare simulated behavior with current actual operations and assess
whether the model is robust enough for future what-if analysis.

## Scenario Intent

Run a no-change baseline scenario that mirrors current operating conditions.
This scenario is intended to validate directional consistency between model
output and actual operations rather than test any new planning lever.

## Parameters / Levers

- Scenario mode: baseline only
- Parameter changes: none
- Category: FEM
- Production line: HPFD
- Network: full sourcing network associated with network 0386
- Time window: 2026-02-02 to 2026-05-03
- Operating logic: reuse current settings and current planning logic as-is

## Expected Trade-offs

- This is not a comparison across alternative parameter settings; the main
  value is model validation against current operations
- If baseline output does not align with actual operations, the result may
  indicate model-input gaps, scope mismatch, or logic issues rather than a
  business-change opportunity
- The primary evaluation will focus on CFR (customer fulfillment rate),
  DFC performance, CU (capacity utilization), and changeover count and
  percentage by changeover type
- Because no explicit KPI threshold has been defined yet, business judgment
  will still be required to determine whether the baseline is sufficiently
  consistent

## Scope

- Category scope: FEM
- Line scope: HPFD
- Network scope: entire sourcing network for network 0386 related to the
  HPFD line
- Time period: 2026-02-02 to 2026-05-03
- Geography / plant scope: included through the full 0386 sourcing network;
  exact node list is not yet enumerated in the workspace
- Run mode: single baseline scenario

## Data Plan

- Simulation inputs should represent current operating conditions for FEM on
  the HPFD line across the 0386 sourcing network
- Actual-operation reference data is required for the same period so the
  business team can compare simulated versus observed behavior on CFR
  (customer fulfillment rate), DFC performance, CU (capacity utilization),
  and changeover count and percentage by changeover type
- Required input tables, data sources, and config workbook paths are not yet
  provided in the workspace
- Exact simulation runner path, command, and output schema are
  user_not_provided
- If a config package is needed, it should be assembled under this scenario's
  `config/` directory before run execution

## Validation Gate

- Scope intent confirmed from user prompt: yes
- Baseline-only scenario confirmed: yes
- Additional variable changes requested: no
- Exact KPI list for validation: confirmed as CFR (customer fulfillment
  rate), DFC performance, CU (capacity utilization), and changeover count
  and percentage by changeover type
- Acceptance threshold for baseline consistency: TBD
- Exact node list inside the 0386 HPFD sourcing network: TBD
- Config package readiness: not_started
- Simulation run readiness: blocked until config artifact and runner details
  are supplied

## Output Paths

- Scope file:
  `workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/scope.yaml`
- Config directory:
  `workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/config/`
- Results directory:
  `workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/results/`
- Analysis file:
  `workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/analysis.md`

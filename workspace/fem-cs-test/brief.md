# FEM Category — HPFD Line Baseline Simulation

## Goal

Run a baseline simulation for the FEM category on the HPFD production line
and compare the simulated operating pattern with current actual operations.
The purpose is to assess whether the model reproduces the current state with
acceptable consistency and to test model robustness before any parameter
change scenarios are considered.

## User

Business stakeholders

## Scope Hypothesis

- Category scope: FEM
- Line scope: HPFD production line
- Network scope: the full sourcing network associated with network 0386 for
  the HPFD line
- Scenario type: baseline only
- Simulation period: 2026-02-02 to 2026-05-03
- Geographic / plant scope: user identified as the full network linked to
  0386 HPFD line sourcing; exact node list is not yet enumerated in the
  workspace

## Constraints & Assumptions

- This is a baseline run only; no intentional parameter perturbation is in
  scope
- The main validation question is alignment between simulation output and
  current operations
- Model robustness will be judged based on whether the baseline can
  reasonably reproduce observed operating behavior over the stated period
- FEM is treated as the category name
- HPFD is treated as the production line identifier
- Primary KPI focus: CFR (customer fulfillment rate), DFC performance,
  CU (capacity utilization), and changeover count and percentage by
  changeover type
- Tolerance thresholds and detailed data source references are
  user_not_provided
- Exact simulation codebase path, runner command, and config package path are
  user_not_provided

## Baseline

- Baseline definition: current operation / current planning logic as
  represented by the existing model inputs
- Parameter change: none
- Comparison target: current actual operation in the same period
- Validation purpose: consistency check and robustness check
- Primary KPIs: CFR (customer fulfillment rate), DFC performance, CU
  (capacity utilization), and changeover count and percentage by
  changeover type
- Primary scenario: baseline FEM x HPFD x network 0386 sourcing network

## Success Criteria

- A baseline scenario package is prepared for FEM on HPFD across the 0386
  sourcing network
- The run covers 2026-02-02 to 2026-05-03
- The output enables business review of simulated versus actual operating
  behavior
- The result is sufficient to judge whether the model is directionally
  consistent with current operations and robust enough for later what-if use

## Candidate Scenarios

1. Confirmed baseline only: FEM category on HPFD line across the full 0386
   sourcing network, with no extra variable changes

## Open Items

- Exact KPI set for baseline validation: confirmed as CFR (customer
  fulfillment rate), DFC performance, CU (capacity utilization), and
  changeover count and percentage by changeover type
- Acceptance threshold for declaring the baseline sufficiently consistent:
  TBD
- Exact node / plant / location list inside the 0386 HPFD sourcing network:
  TBD
- Data sources used for actual-versus-simulated comparison: TBD
- Simulation runner details and config artifact paths: user_not_provided

# SDC Space RCCP — Baseline Simulation

## Goal

Estimate the should-be Space RCCP under current parameter settings for
the SDC node and its immediate upstream network from 2026-06-29 to
2027-01-03, across all categories, so the team can identify monthly
peak space pressure before any parameter tuning starts.

## Scope Hypothesis

- Network scope: one SDC node plus its immediate upstream network
- Location identity: the SDC code list is only partially available in
  system and must be reconciled with offline data
- Simulation period: 2026-06-29 to 2027-01-03
- Category scope: all categories
- Scenario type: baseline only, no parameter perturbation

## Constraints & Assumptions

- Current parameter settings are treated as the baseline configuration
- Some downstream SDC codes do not exist in the system extract and must
  be supplemented offline
- Baseline config workbook has been provided offline and imported into
  the workspace
- Simulation codebase path and run command are user_not_provided
- Space RCCP should be reported as monthly peak inventory quantity in
  CBM, primarily by month x location and optionally by month x location
  x category

## Baseline

- Parameter basis: current settings
- KPI focus: Space RCCP
- Exact SDC location list and offline-to-system mapping: embedded in the
  imported offline workbook and not yet re-extracted as a separate list
- Baseline workbook: `deliverables/baseline-current-params-should-be-rccp.xlsx`
- Scenario identifier in external runner: TBD

## Success Criteria

- The baseline scenario has a resolved scope, explicit data reconciliation
  plan, and agreed config target path
- All in-scope SDC locations are reconciled across system and offline
  data before config generation starts
- The project is ready for config package assembly and run handoff once
  workbook and runner details are supplied

## Confirmed Scenarios

1. Baseline: current parameter settings x should-be Space RCCP
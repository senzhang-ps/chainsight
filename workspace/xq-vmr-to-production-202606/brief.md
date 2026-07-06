# xq-vmr-to-production-202606 Brief
## Goal
Assess the baseline production and inventory behavior for HC/PCC flow at XQ plant (1864) from VMR to production, and establish the reference point for subsequent what-if simulations on AO ratio and advance lead time.

## Focus Metrics
- MOE
- National inventory
- Service

## Scope Hypothesis
- Period: 2026-06-29 to 2026-11-01
- Plant / network anchor: XQ plant (1864)
- Category / products: HC and PCC
- Supply chain focus: VMR to production

## Constraints & Assumptions
- This first scenario is baseline only.
- Follow-on what-if scenarios will change AO ratio and advance lead time.
- Detailed configuration values will be provided directly by the user.
- Material-level scope is TBD until user provides the exact configuration inputs.

## Baseline
- Baseline scenario to be established by simulation.
- Current reference values for MOE, national inventory, and service are user_not_provided.

## Success Criteria
- A runnable baseline scenario is defined for the agreed period and scope.
- Baseline outputs can serve as the comparison point for later AO-ratio and lead-time what-if scenarios.
- Results include at least MOE, national inventory, and service for downstream comparison.

## Candidate Scenarios
- Baseline: current configuration for HC/PCC at XQ plant
- Future what-if 1: change AO ratio
- Future what-if 2: change advance lead time
- Future what-if 3: combined AO ratio and advance lead time changes

## Confirmed Scenarios
|Scenario name|Description|Key Variable(s) Changed|Expected Impact on Focus Metrics|Watchout/Risk|Priority|
|---|---|---|---|---|---|
|baseline|Baseline simulation for HC/PCC XQ plant VMR to production|None; current-state configuration|Establish comparison point for MOE, national inventory, and service|Material scope and exact config values still pending user input|P0|
|baseline-hc|HC-only variant of baseline; all confirmed-PCC config removed|Scope filter: drop 249 confirmed-PCC materials and PCC-only production lines|Isolate HC MOE, national inventory, and service without PCC interaction|XQHD/XQHG are mixed lines kept for HC; non-produced inert materials retained to preserve HC sourcing paths|P1|

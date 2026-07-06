# Line M/Y/C Sourcing Network — Supply Choice & Safety Stock What-If

## Goal

Evaluate four what-if scenarios that combine supply choice policy and
safety stock policy across the full sourcing network for Line M/Y/C
SKUs, so the team can compare service and inventory implications before
running simulation.

## Scope Hypothesis

- Network scope: the full network for Line M/Y/C sourcing SKUs
- Simulation period: 2026-05-04 to 2026-06-19
- Baseline planning version: 50/50 BOP using the 2026-04-27 LBE version
- Initial inventory basis: 2026-04-27 actual inventory, including stock
  on hand and in-transit

## Constraints & Assumptions

- High-side logic: maintain delta in supply choice
- Forecast error CoV: follow iBPI
- AO: assume 0%
- Safety stock for what-if 1 and what-if 2: follow iBPI after override
- Safety stock for what-if 3 and what-if 4: DTC locations reduce from
  30 days to 15 days
- Run command / simulation codebase path: user_not_provided

## Baseline

- Produce policy baseline anchor: 50/50 BOP, Apr 27 version LBE
- Order policy baseline anchor: current safety stock unless scenario
  states reduced safety
- KPI baseline values for service, inventory, and supply stability: TBD

## Success Criteria

- All four confirmed scenarios have aligned scope, assumptions, config
  targets, and output paths
- The project is ready for config package generation and simulation run
  handoff without re-asking the core business setup
- Comparison KPIs and decision rule for selecting a preferred scenario:
  TBD

## Confirmed Scenarios

1. What-if 1: produce to high-side, order as 50/50 forecast x current safety
2. What-if 2: produce to 50/50, order as high-side x current safety
3. What-if 3: produce to high-side, order as 50/50 forecast x reduced safety
4. What-if 4: produce to 50/50, order as high-side x reduced safety
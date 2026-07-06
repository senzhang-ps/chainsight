# South Space RCCP JASON26 Brief

## Goal
Assess South space RCCP under current system parameters while explicitly considering factory production cycle behavior for the South network, with focus on C816/C810/D873/C866/C867, over 2026-06-29 to 2026-11-29.

## Focus Metrics
- Space RCCP
- Space pressure by month and by key location
- Production-cycle-driven inventory build / release pattern at key plants and connected nodes
- Optional supporting metrics: inventory quantity and deployment / flow pressure where needed to explain space peaks

## Scope Hypothesis
- Network scope: the national network that passes through C816, including in-house, ESS, and both upstream and downstream nodes connected to C816
- Key focus nodes: C816, C810, D873, C866, C867
- Time range: 2026-06-29 to 2026-11-29
- Scenario type: baseline on current system parameters, evaluated through production cycle logic

## Constraints & Assumptions
- Current system parameters are the baseline unless the user later confirms parameter changes
- Network scope should include the full relevant C816-centered national flow, not only one upstream layer
- Production cycle is a core modeling lens and must be reflected in the design and config plan
- Exact material list is not yet provided and remains TBD
- Exact KPI output grain remains TBD (monthly peak by location is assumed unless user changes it)
- Existing South/SDC RCCP project artifacts may be used as reference only, not copied blindly

## Baseline
- Parameter basis: current system parameters
- Business question: under existing settings, how does production cycle shape South space RCCP across the C816-centered network?
- Baseline workbook / config package: TBD
- Current known key locations: C816, C810, D873, C866, C867

## Success Criteria
- The project has a clear executable scope for the full C816-centered national network
- The scenario design clearly explains how production cycle will be represented in the simulation inputs
- Required config tables and data sources are identified before extraction starts
- The scenario is ready for config preparation and validation

## Candidate Scenarios
- Baseline current system parameters with production-cycle view for South space RCCP
- Optional follow-up what-if scenarios on production cycle, line capacity, safety stock, or deployment policy after baseline is established

## Confirmed Scenarios

|Scenario name|Description|Key Variable(s) Changed|Expected Impact on Focus Metrics|Watchout/Risk|Priority|
|---|---|---|---|---|---|
|production-cycle|Baseline South space RCCP using current system parameters, evaluated across the full C816-centered national network with explicit production cycle consideration|None in baseline; production cycle representation is the main modeling focus|Reveal where and when production cycle drives monthly space peaks across C816/C810/D873/C866/C867 and connected nodes|Material scope and production-cycle parameterization are not yet fully resolved|High|

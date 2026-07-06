# Scenario: Production Cycle

## Goal Link
Assess South space RCCP under current system parameters for the C816-centered national network, with explicit attention to how production cycle drives space usage across key South nodes.

## Scenario Intent
Run a baseline scenario to quantify South space RCCP from 2026-06-29 to 2026-11-29 across the national network passing through C816, including in-house, ESS, and all relevant upstream and downstream nodes. The design should make production cycle a first-class consideration so the team can see how cycle-driven inventory patterns contribute to space peaks at C816/C810/D873/C866/C867.

## Parameters / Levers
- Scenario mode: baseline only
- Parameter basis: current system parameters
- Explicit modeling lens: production cycle
- Key locations: C816, C810, D873, C866, C867
- Parameter changes: none confirmed yet
- Open design point: identify which config tables carry the production cycle logic in the current simulation setup

## Expected Trade-offs
- A baseline with current settings gives a realistic current-state RCCP view, but may not isolate improvement levers yet
- Expanding to the full national network around C816 increases realism but also raises data volume and config complexity
- If production cycle is not represented correctly in M4 or related tables, space peaks may be materially misestimated

## Scope
- Network: national network passing through C816
- Coverage: include in-house, ESS, and both upstream and downstream nodes connected to C816
- Focus locations: C816, C810, D873, C866, C867
- Time range: 2026-06-29 to 2026-11-29
- Material scope: TBD
- Run mode: single baseline scenario

## Data Plan
- Resolve the exact material scope for the South RCCP study before extraction
- Use an imported offline workbook as the baseline configuration package source: `config/PDS2.xlsx`
- Use an imported companion row-level table for safety stock: `config/M3_SafetyStock.csv`
- Extract or assemble `Global_Network` for the full C816-centered network, including ESS virtual rows if needed
- Review whether `Global_LeadTime`, `Global_SpaceCapacity`, `M1_InitialInventory`, `M1_DemandForecast`, `M3_SafetyStock`, `M4_MaterialLocationLineCfg`, `M4_LineCapacity`, and possibly changeover / reliability tables are required to represent production cycle correctly
- Confirm whether `PDS2.xlsx` already contains all required tabs for this scenario or needs supplementation
- Reuse prior South/SDC RCCP project artifacts only as references for table coverage, validation approach, and output structure
- Validate config package before any simulation run

## Validation Gate
- Scope intent confirmed from user prompt: yes
- Project / scenario identity confirmed: yes
- Time range confirmed: yes
- Network anchor confirmed: yes, centered on C816 and connected national network
- Material scope confirmed: no, still TBD
- Config generation ready: not yet
- Simulation run ready: not yet

## Output Paths
- Brief: `workspace/south-space-rccp-jason26/brief.md`
- Scope file: `workspace/south-space-rccp-jason26/scenarios/production-cycle/scope.yaml`
- Config directory: `workspace/south-space-rccp-jason26/scenarios/production-cycle/config/`
- Results directory: `workspace/south-space-rccp-jason26/scenarios/production-cycle/results/`
- Analysis file: `workspace/south-space-rccp-jason26/scenarios/production-cycle/analysis.md`

# Scenario: Baseline Current Params Should-Be RCCP

## Goal Link

Create a baseline simulation that estimates should-be Space RCCP for the
SDC node and its immediate upstream network under current settings.

## Scenario Intent

Run a no-change baseline so the team can quantify monthly peak space
requirements in CBM before deciding whether any space, inventory, or
network what-if levers are needed.

## Parameters / Levers

- Scenario mode: baseline only
- Parameter changes: none
- Baseline settings: reuse current planning parameters as-is
- Space capacity inputs: use current capacity records where available
- SDC mapping rule: merge system-visible nodes with offline SDC data
  before config assembly; this has been materialized in the imported
  workbook

## Expected Trade-offs

- This is not a comparison scenario; the primary value is a baseline
  capacity view rather than a lever trade-off
- The main execution risk is underestimating space pressure if offline
  SDC nodes or capacity records are missing from the assembled config

## Scope

- Network: one SDC node and its immediate upstream network
- SDC identity: exact code list is represented in the imported offline
  workbook; a standalone reconciled list has not been extracted yet
- Period: 2026-06-29 to 2027-01-03
- Categories: all categories
- Run mode: single baseline scenario

## Data Plan

- System-covered inputs: baseline network records for the plant and any
  system-visible SDC nodes, demand forecast, initial inventory, lead
  time, and material master data needed for CBM conversion
- Manual / offline supplements: missing SDC codes, node mapping,
  space-capacity records for offline SDCs, and any missing lanes needed
  to connect the plant to those SDC nodes. These have been assembled in
  the imported workbook from the offline source package
- Standalone companion config file: `M3_SafetyStock.csv` has also been
  provided offline and imported into the scenario config folder as a
  row-level safety stock table with columns `material`, `location`,
  `date`, and `safety_stock_qty`
- Config tables to prioritize: `Global_Network`, `Global_SpaceCapacity`,
  `Global_LeadTime`, `M1_InitialInventory`, `M1_DemandForecast`, and
  `M6_MaterialMD`; the imported workbook currently contains 28 sheets,
  including the required baseline tabs and optional tabs used by the
  user's offline preparation flow
- Config target: assemble workbook at
  `workspace/sdc-space-rccp-simulation-202605/deliverables/baseline-current-params-should-be-rccp.xlsx`

## Validation Gate

- Scope intent confirmed from user prompt: yes
- Exact in-scope SDC location list confirmed: encoded in the imported
  workbook; separate scope extraction remains optional
- Validate config before run: completed using the validation-skill
  workflow at the config-folder level
- Validation method: choose the only workbook in the folder, then use
  `M3_SafetyStock.csv` as authoritative over the workbook tab because
  both forms exist for the same table
- Validation result: `PASS WITH ACCEPTED BUSINESS EXCEPTIONS` / ready
  with accepted risk
- Validation quality review: independent worker review completed against
  the validation skill checklist
- Imported companion file: `M3_SafetyStock.csv`
- Accepted risk summary: M4 tabs treated as non-blocking for this
  scenario, shortened forecast tail accepted, and 7 materials retain
  zero conversion factors because the source master data does not supply
  usable values
- Skill-based spot checks confirmed: all mandatory tabs are present at
  the package level, all conditional tabs are present, and
  `M3_SafetyStock.csv` covers the full simulation period with no null key
  fields, no bad dates, and no negative safety stock quantity
- Run gate: blocked until workbook template or simulation command /
  codebase path is supplied

## Output Paths

- Scope file:
  `workspace/sdc-space-rccp-simulation-202605/scenarios/baseline-current-params-should-be-rccp/scope.yaml`
- Config directory:
  `workspace/sdc-space-rccp-simulation-202605/scenarios/baseline-current-params-should-be-rccp/config/`
- Imported config workbook:
  `workspace/sdc-space-rccp-simulation-202605/scenarios/baseline-current-params-should-be-rccp/config/baseline-current-params-should-be-rccp.xlsx`
- Imported companion CSV:
  `workspace/sdc-space-rccp-simulation-202605/scenarios/baseline-current-params-should-be-rccp/config/M3_SafetyStock.csv`
- Results directory:
  `workspace/sdc-space-rccp-simulation-202605/scenarios/baseline-current-params-should-be-rccp/results/`
- Analysis file:
  `workspace/sdc-space-rccp-simulation-202605/scenarios/baseline-current-params-should-be-rccp/analysis.md`
# baseline Design
## Goal Link
This baseline scenario supports project `xq-vmr-to-production-202606` by establishing the reference simulation output for later AO ratio and advance lead time what-if comparisons.

## Scenario Intent
Run the current-state baseline for HC/PCC at XQ plant (1864) across the VMR-to-production flow for 2026-06-29 to 2026-11-01.

## Parameters / Levers
- Baseline only; no intentional parameter overrides in this scenario.
- Future scenarios are expected to vary:
  - `ao_percent` via `M1_AOConfig`
  - `advance_days` via `M1_AOConfig`
  - and/or related lead-time fields in `Global_LeadTime` if the user defines lead-time changes at lane level

## Expected Trade-offs
- Baseline should quantify current MOE, national inventory, and service.
- Future higher AO ratio or earlier order timing may improve service and production visibility, but may also shift inventory and MOE behavior.
- Exact trade-offs to be quantified after baseline is run.

## Scope
- Time window: 2026-06-29 to 2026-11-01
- Plant: XQ (1864)
- Category scope: HC and PCC
- Material scope: TBD from user-provided configuration
- Process scope: VMR to production

## Data Plan
Required baseline config tables expected for run readiness:
- `Global_Network`
- `Global_LeadTime`
- `M1_InitialInventory`
- `M1_DemandForecast`
- `M3_SafetyStock`

Potentially needed depending on model path and plant setup:
- `M1_AOConfig` (kept as baseline values if applicable)
- `M4_MaterialLocationLineCfg`
- `M4_LineCapacity`
- `M4_ChangeoverDefinition`
- `M4_ChangeoverMatrix`
- `M5_DeployConfig`
- `M5_PushPullModel`

Config source strategy:
- User will provide the detailed baseline configuration directly.
- We will package the config under `workspace/xq-vmr-to-production-202606/scenarios/baseline/config/`.
- We will validate the package before simulation.

## Validation Gate
Before simulation run:
- required tabs must exist
- schemas must match config mapping
- time window and scope must be internally consistent
- no blocking ERROR in config validation

## Output Paths
- Brief: `workspace/xq-vmr-to-production-202606/brief.md`
- Scope: `workspace/xq-vmr-to-production-202606/scenarios/baseline/scope.yaml`
- Config: `workspace/xq-vmr-to-production-202606/scenarios/baseline/config/`
- Results: `workspace/xq-vmr-to-production-202606/scenarios/baseline/results/`
- Analysis: `workspace/xq-vmr-to-production-202606/scenarios/baseline/analysis.md`

# baseline-hc Design
## Goal Link
This scenario supports project `xq-vmr-to-production-202606` by providing an HC-only
view derived from the `baseline` scenario. It removes all PCC-related configuration so
HC behavior (MOE, national inventory, service) can be analyzed without PCC interaction.

## Scenario Intent
Take the `baseline` config and strip every confirmed-PCC material (and the production
lines left empty as a result), keeping only HC at XQ plant (1864) across the
VMR-to-production flow for 2026-06-29 to 2026-11-01.

## Derivation From Baseline
- Source: `scenarios/baseline/config/`
- HC/PCC classification (local files only, no Databricks):
  - HC if any of: in `SUF for XQ HC ChainSight.xlsx`; cached `category_en == Hair`
    (analysis extract `material_category_databricks.csv`); manual
    `missing category manual input.xlsx` `category_en == Hair`; `delegate_line == XQHK`.
  - PCC if (and not HC) any of: cached `category_en == PCC`; manual `category_en == PCC`;
    `delegate_line` in pure-PCC lines (XQ Line 10, XQHA, XQHH, XQHJ, XQHV).
- Confirmed-PCC materials are removed from every material-keyed config table.
- Non-produced, unclassified materials (network/push-pull only, ~zero demand) are kept
  to preserve HC sourcing/network integrity.

## Parameters / Levers
- No parameter overrides. This is a scope filter only (remove PCC).

## Scope
- Time window: 2026-06-29 to 2026-11-01
- Plant: XQ (1864)
- Category scope: HC only
- Process scope: VMR to production

## Validation Gate
Before simulation run:
- required tabs must exist
- schemas must match config mapping
- no HC material loses a sourcing path due to the PCC removal
- no blocking ERROR in config validation

## Output Paths
- Scope: `workspace/xq-vmr-to-production-202606/scenarios/baseline-hc/scope.yaml`
- Config: `workspace/xq-vmr-to-production-202606/scenarios/baseline-hc/config/`
- Results: `workspace/xq-vmr-to-production-202606/scenarios/baseline-hc/results/`
- Analysis: `workspace/xq-vmr-to-production-202606/scenarios/baseline-hc/analysis/`

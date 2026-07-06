# WF1 Analysis

## Scenario

- Scenario: `wf1-produce-hs-order-5050-current-ss`
- Run ID: `db_wf1-produce-hs-order-5050-current-ss_20260511_180415`
- Intent: high-side production with 50/50 ordering under current safety stock

## Facts

- CFR: `96.93%`
- Cut rate: `3.07%`
- Average inventory: `171,340.43`
- Ending inventory: `168,529`
- Peak inventory: `183,983`
- Produced quantity: `68,958`
- Changeovers: `24`
- Changeover time: `35.00`

Top DC shortfalls:

- `C937`: shortfall `431`, CFR `79.07%`
- `A668`: shortfall `262`, CFR `89.05%`
- `C816`: shortfall `222`, CFR `96.81%`

## Interpretation

`wf1` is the strongest service scenario in the set and the clearest
match to a service-first posture. The expected inventory-build trade-off
appears in the output: June average inventory is higher than May, and
both average and ending inventory are the highest among the four
scenarios.

## Recommendation

Use `wf1` as the reference case if the business wants to protect service
first and is willing to absorb the highest inventory load. If inventory
cost or space is already tight, `wf1` is likely too heavy unless the
service premium is explicitly worth that carry.# Analysis Placeholder

Status: awaiting config generation and simulation results.

## Planned Comparison Focus

- Service level / fill rate impact versus the other three what-if scenarios
- Inventory and coverage impact under current safety stock
- Evidence of over-build or under-supply caused by the high-side produce policy

## Evidence Level

- Current status: planning-only
- Missing for formal analysis: validated config workbook, simulation outputs,
  KPI extraction logic confirmation, and a declared decision rule
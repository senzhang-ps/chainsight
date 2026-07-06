# WF2 Analysis

## Scenario

- Scenario: `wf2-produce-5050-order-hs-current-ss`
- Run ID: `db_wf2-produce-5050-order-hs-current-ss_20260511_180310`
- Intent: 50/50 production with high-side ordering under current safety stock

## Facts

- CFR: `96.54%`
- Cut rate: `3.46%`
- Average inventory: `155,279.81`
- Ending inventory: `141,532`
- Peak inventory: `173,449`
- Produced quantity: `81,052`
- Changeovers: `25`
- Changeover time: `38.67`

Top DC shortfalls:

- `E094`: shortfall `723`, CFR `96.72%`
- `C937`: shortfall `532`, CFR `73.45%`
- `E353`: shortfall `483`, CFR `95.18%`

## Interpretation

`wf2` is the most balanced scenario in this result set. It keeps service
close to `wf1`, but materially lowers average and ending inventory. The
high-side order policy increases total order and shipment volume, yet the
network still preserves a CFR above `96.5%`, which is stronger than the
reduced-safety-stock alternatives.

## Recommendation

Take `wf2` forward as the preferred compromise candidate if the goal is
to relieve inventory from `wf1` without paying the larger service penalty
seen in `wf3`. The main caution is the persistent `C937` gap and the high
absolute shortfall at `E094`.# Analysis Placeholder

Status: awaiting config generation and simulation results.

## Planned Comparison Focus

- Service level / fill rate impact versus the other three what-if scenarios
- Inventory and coverage impact under current safety stock
- Evidence of demand signal amplification caused by the high-side order policy

## Evidence Level

- Current status: planning-only
- Missing for formal analysis: validated config workbook, simulation outputs,
  KPI extraction logic confirmation, and a declared decision rule
# WF4 Analysis

## Scenario

- Scenario: `wf4-produce-5050-order-hs-reduced-ss`
- Run ID: `db_wf4-produce-5050-order-hs-reduced-ss_20260511_180323`
- Intent: 50/50 production with high-side ordering and reduced DTC safety stock

## Facts

- CFR: `96.10%`
- Cut rate: `3.90%`
- Average inventory: `146,047.15`
- Ending inventory: `124,302`
- Peak inventory: `171,709`
- Produced quantity: `57,733`
- Changeovers: `16`
- Changeover time: `25.25`

Top DC shortfalls:

- `C937`: shortfall `709`, CFR `65.06%`
- `E094`: shortfall `644`, CFR `96.90%`
- `C816`: shortfall `436`, CFR `93.96%`

## Interpretation

`wf4` is the lowest-inventory scenario by a clear margin, especially in
June, where average inventory falls to `128,034.47`. It retains better
service than `wf3`, which suggests reduced safety stock is more workable
when paired with 50/50 production than with high-side production.
However, it also produces the largest localized service failure at
`C937`, making it the riskiest scenario from a node-level service view.

## Recommendation

Use `wf4` only if inventory reduction is the dominant objective and the
business can actively manage the resulting service risk. It is a serious
lean-inventory candidate, but it should be accompanied by mitigation on
`C937` before being treated as a preferred operating policy.# Analysis Placeholder

Status: awaiting config generation and simulation results.

## Planned Comparison Focus

- Service level / fill rate impact versus the other three what-if scenarios
- Inventory and coverage impact after reducing DTC safety stock from 30 to 15 days
- Whether the combination of high-side ordering and reduced safety stock creates instability

## Evidence Level

- Current status: planning-only
- Missing for formal analysis: validated config workbook, simulation outputs,
  KPI extraction logic confirmation, and a declared decision rule
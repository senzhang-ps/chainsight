# South Space RCCP JASON26 — Configuration Validation Audit Report (final rerun)

## 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | PASSED WITH WARNING |
| Simulation readiness | Ready to simulate |
| Configuration file | `workspace/south-space-rccp-jason26/scenarios/production-cycle/config/PDS2.xlsx` + `workspace/south-space-rccp-jason26/scenarios/production-cycle/config/M3_SafetyStock.csv` |
| Simulation period | 2026-06-29 to 2026-11-29 |
| Scope benchmark | `Global_Network` union scope: `(material, location)` ∪ `(material, sourcing)`; sending-receiving benchmark from `Global_Network (sourcing, location)` |
| DFC thresholds | low = 15, high = 30 |
| Total ERROR count | 0 |
| Total WARNING count | 2 |

## 2. Accepted exceptions in current validation posture
- `M1_DemandForecast` week coverage kept as-is and accepted.
- `M1_AOConfig` precision issue accepted.
- `M1_DPSConfig` and `M1_SupplyChoiceConfig` empty but unused in scenario.
- `Global_LeadTime` `PDT=OTD` rule handled by business rule; `D455->C816` accepted.
- `M6_TruckReleaseCon` `D455->C816` accepted.
- `M5_PushPullModel`: `Pull` rows removed; remaining values accepted.
- `M4_MaterialLocationLineCfg` 21 off-network materials accepted as business exception.

## 3. Remaining warning detail

| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISSUE-001 | SC-M1INV-001 | WARNING | M1_InitialInventory | DFC low vs week1 demand | 1910 rows | Initial inventory DFC below low threshold 15 | DFC should be reviewed vs thresholds | Review low DFC rows |
| ISSUE-002 | SC-M1INV-001 | WARNING | M1_InitialInventory | DFC high vs week1 demand | 2459 rows | Initial inventory DFC above high threshold 30 | DFC should be reviewed vs thresholds | Review high DFC rows |

## 4. Readiness conclusion

| Decision | Condition | Conclusion text |
| --- | --- | --- |
| PASSED WITH WARNING | No ERROR remains | Configuration is ready for simulation. Remaining items are DFC soft warnings only. |

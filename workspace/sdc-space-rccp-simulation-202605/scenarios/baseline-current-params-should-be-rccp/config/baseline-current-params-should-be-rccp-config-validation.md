## 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | PASS WITH WARNINGS |
| Simulation readiness | Ready with risk |
| Configuration file | `workspace/sdc-space-rccp-simulation-202605/scenarios/baseline-current-params-should-be-rccp/config/baseline-current-params-should-be-rccp.xlsx` with CSV override `workspace/sdc-space-rccp-simulation-202605/scenarios/baseline-current-params-should-be-rccp/config/M3_SafetyStock.csv` |
| Simulation period | 2026-06-29 to 2027-01-03 |
| Scope benchmark | `Global_Network` material-location combinations |
| Total ERROR count | 0 |
| Total WARNING count | 1 |

## 2. Issue detail

| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WARN-001 | SC-M1INV-DFCLOW | WARNING | M1_InitialInventory | DFC vs week 1 daily average demand | 19,361 material-location rows | Initial inventory is below the low DFC threshold of 15 days for part of the network. | DFC should be at least 15 days unless the business explicitly accepts lower initial coverage for the scenario. | Review the low-DFC material-location rows and confirm they are acceptable for the baseline run. |

## 3. Tab-level summary

| Tab | Status | ERROR count | WARNING count | Main issue type | Remark |
| --- | --- | --- | --- | --- | --- |
| Global_Network | PASS | 0 | 0 | None | Present in workbook and used as benchmark scope. |
| Global_LeadTime | PASS | 0 | 0 | None | Present in workbook. |
| Global_DemandPriority | PASS | 0 | 0 | None | Present in workbook. |
| M1_DemandForecast | PASS | 0 | 0 | None | Present in workbook. |
| M1_OrderCalendar | PASS | 0 | 0 | None | Present in workbook. |
| M1_InitialInventory | WARNING | 0 | 1 | Soft risk | Low DFC warning under thresholds low=15, high=50. |
| M1_ForecastError | PASS | 0 | 0 | None | Present in workbook. |
| M3_SafetyStock | PASS | 0 | 0 | None | CSV override used; 2,986,578 rows, no null key fields, no bad dates, no negative quantity, full 189-day coverage per material-location. |
| M5_PushPullModel | PASS | 0 | 0 | None | Present in workbook. |
| M5_DeployConfig | PASS | 0 | 0 | None | Present in workbook. |
| M6_MaterialMD | PASS | 0 | 0 | Accepted business exception | Offline audit accepted 7 materials with zero conversion factors because source master data lacks usable values. |
| M6_TruckReleaseCon | PASS | 0 | 0 | None | Present in workbook. |
| M6_TruckTypeSpecs | PASS | 0 | 0 | None | Present in workbook. |
| M6_DeliveryDelayDistribution | PASS | 0 | 0 | None | Present in workbook. |
| M1_AOConfig | PASS | 0 | 0 | Conditional tab present | Present in workbook. |
| M1_DPSConfig | PASS | 0 | 0 | Conditional tab present | Present in workbook. |
| M1_SupplyChoiceConfig | PASS | 0 | 0 | Conditional tab present | Present in workbook. |
| M4_MaterialLocationLineCfg | PASS | 0 | 0 | Conditional tab present | Present in workbook; offline audit treated M4 as non-blocking for this scenario. |
| M4_LineCapacity | PASS | 0 | 0 | Conditional tab present | Present in workbook; offline audit treated M4 as non-blocking for this scenario. |
| M4_ProductionReliability | PASS | 0 | 0 | Conditional tab present | Present in workbook; offline audit treated M4 as non-blocking for this scenario. |
| M4_ChangeoverDefinition | PASS | 0 | 0 | Conditional tab present | Present in workbook; offline audit treated M4 as non-blocking for this scenario. |
| M4_ChangeoverMatrix | PASS | 0 | 0 | Conditional tab present | Present in workbook; offline audit treated M4 as non-blocking for this scenario. |
| Global_seed | PASS | 0 | 0 | Optional tab present | Present in workbook. |
| Global_SpaceCapacity | PASS | 0 | 0 | Optional tab present | Present in workbook. |
| M6_MDQBypassRules | PASS | 0 | 0 | Optional tab present | Present in workbook. |
| M6_TruckCapacityPlan | PASS | 0 | 0 | Optional tab present | Present in workbook. |

## 4. Readiness conclusion

| Decision | Condition | Conclusion text |
| --- | --- | --- |
| FAILED | Any ERROR exists | Configuration is not ready for simulation. Blocking issues must be fixed before running ChainSight engine. |
| PASS WITH WARNINGS | No ERROR exists, but at least one WARNING exists | Configuration can be used for simulation, but warning risks should be reviewed. This package is ready with risk for the SDC Space RCCP baseline scenario. |
| PASS | No ERROR and no WARNING | Configuration is ready for simulation. |
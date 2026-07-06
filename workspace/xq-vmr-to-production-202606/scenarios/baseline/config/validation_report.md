### 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | PASS WITH WARNINGS |
| Simulation readiness | Ready with risk |
| Configuration file | /home/zhangs37/ai_chainsight/workspace/xq-vmr-to-production-202606/scenarios/baseline/config |
| Simulation period | 2026-06-29 to 2026-11-01 |
| Scope benchmark | Materials from M4_MaterialLocationLineCfg; material-location nodes must be traceable to 1864 in Global_Network; sending-receiving requirements exclude self-loops where sending==receiving. |
| Total ERROR count | 0 |
| Total WARNING count | 6 |

### 2. Issue detail

| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISSUE-008 | CONS-ML-002 | WARNING | M5_DeployConfig | material-receiving | 111 extra pairs | Tab contains out-of-benchmark scope | Keep scope aligned with 1864-traceable benchmark | Remove extra out-of-scope rows if unintended |
| ISSUE-013 | SC-INV-001 | WARNING | M1_InitialInventory | DFC days | 115 rows below 20 | Initial inventory too low vs week 1 daily average demand | DFC should be >= low threshold | Review low initial inventory risk |
| ISSUE-014 | SC-INV-002 | WARNING | M1_InitialInventory | DFC days | 427 rows above 40 | Initial inventory too high vs week 1 daily average demand | DFC should be <= high threshold | Review high initial inventory risk |

### 3. Tab-level summary

| Tab | Status | ERROR count | WARNING count | Main issue type | Remark |
| --- | --- | --- | --- | --- | --- |
| Global_DemandPriority | PASS | 0 | 0 |  |  |
| Global_LeadTime | PASS | 0 | 0 |  |  |
| Global_Network | PASS | 0 | 0 |  |  |
| Global_SpaceCapacity | NOT MAINTAINED | 0 | 0 |  | Optional tab not provided |
| Global_seed | NOT MAINTAINED | 0 | 0 |  | Optional tab not provided |
| M1_AOConfig | PASS | 0 | 0 |  |  |
| M1_DPSConfig | NOT MAINTAINED | 0 | 0 |  | Conditional tab not provided |
| M1_DemandForecast | PASS | 0 | 0 | accepted by user | Current alignment accepted by user |
| M1_ForecastError | PASS | 0 | 0 | accepted by user | Current alignment accepted by user |
| M1_InitialInventory | WARNING | 0 | 2 | accepted by user | Current alignment accepted by user; only DFC risk remains |
| M1_OrderCalendar | PASS | 0 | 0 |  |  |
| M1_SupplyChoiceConfig | NOT MAINTAINED | 0 | 0 |  | Conditional tab not provided |
| M3_SafetyStock | PASS | 0 | 0 | accepted by user | Current alignment accepted by user |
| M4_ChangeoverDefinition | PASS | 0 | 0 |  |  |
| M4_ChangeoverMatrix | PASS | 0 | 0 |  |  |
| M4_LineCapacity | PASS | 0 | 0 |  |  |
| M4_MaterialLocationLineCfg | PASS | 0 | 0 |  |  |
| M4_ProductionReliability | PASS | 0 | 0 |  |  |
| M5_DeployConfig | WARNING | 0 | 1 | consistency | Tab contains out-of-benchmark scope |
| M5_PushPullModel | PASS | 0 | 0 |  |  |
| M6_DeliveryDelayDistribution | PASS | 0 | 0 |  |  |
| M6_MDQBypassRules | PASS | 0 | 0 |  |  |
| M6_MaterialMD | PASS | 0 | 0 |  |  |
| M6_TruckCapacityPlan | NOT MAINTAINED | 0 | 0 |  | Optional tab not provided |
| M6_TruckReleaseCon | PASS | 0 | 0 |  |  |
| M6_TruckTypeSpecs | PASS | 0 | 0 |  |  |

### 4. Readiness conclusion

| Decision | Condition | Conclusion text |
| --- | --- | --- |
| FAILED | Any ERROR exists | Configuration is not ready for simulation. Blocking issues must be fixed before running ChainSight engine. |
| PASS WITH WARNINGS | No ERROR exists, but at least one WARNING exists | Configuration can be used for simulation, but warning risks should be reviewed. |
| PASS | No ERROR and no WARNING | Configuration is ready for simulation. |
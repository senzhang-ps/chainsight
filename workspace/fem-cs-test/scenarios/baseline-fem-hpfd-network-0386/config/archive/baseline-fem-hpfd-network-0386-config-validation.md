## 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | FAILED |
| Simulation readiness | Not ready |
| Configuration file | /home/zhangs37/ai_chainsight/workspace/fem-cs-test/scenarios/baseline-fem-hpfd-network-0386/config/baseline-fem-hpfd-network-0386.xlsx (workbook read in tolerant mode because the stylesheet XML contains invalid font-family metadata; source workbook was not modified) |
| Simulation period | 2026-02-02 to 2026-05-03 |
| Scope benchmark | Seed = M1_DemandForecast material-location; expanded benchmark = recursive upstream Global_Network material-location closure (seed=388, expanded=411) |
| Total ERROR count | 38 |
| Total WARNING count | 2 |

## 2. Issue detail

| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISSUE-001 | SCHEMA-NUMFMT | ERROR | M1_DemandForecast | week | 4959 row(s); sample index 0, 1, 2 | Numeric field contains non-numeric values. | Numeric fields must be parseable numbers. | Fix the non-numeric values in this column. |
| ISSUE-002 | SCHEMA-EMPTYCOL | ERROR | M4_MaterialLocationLineCfg | min_batch | min_batch | Required column is fully empty. | Required columns should contain maintained values. | Populate the required column. |
| ISSUE-003 | SCHEMA-EMPTYCOL | ERROR | M4_MaterialLocationLineCfg | rv | rv | Required column is fully empty. | Required columns should contain maintained values. | Populate the required column. |
| ISSUE-004 | SCHEMA-EMPTYCOL | ERROR | M4_MaterialLocationLineCfg | ptf | ptf | Required column is fully empty. | Required columns should contain maintained values. | Populate the required column. |
| ISSUE-005 | SCHEMA-EMPTYCOL | ERROR | M4_MaterialLocationLineCfg | day | day | Required column is fully empty. | Required columns should contain maintained values. | Populate the required column. |
| ISSUE-006 | SCHEMA-EMPTYCOL | ERROR | M4_MaterialLocationLineCfg | MCT | MCT | Required column is fully empty. | Required columns should contain maintained values. | Populate the required column. |
| ISSUE-007 | SCHEMA-EMPTYCOL | ERROR | M4_ChangeoverDefinition | cost | cost | Required column is fully empty. | Required columns should contain maintained values. | Populate the required column. |
| ISSUE-008 | SCHEMA-EMPTYCOL | ERROR | M4_ChangeoverDefinition | mu_loss | mu_loss | Required column is fully empty. | Required columns should contain maintained values. | Populate the required column. |
| ISSUE-009 | XREF-ML-MISSING | ERROR | Global_Network | material/location | 36 combo(s); sample 80728859-0386, 80728861-0386, 80799571-0386 | Global_Network is missing expanded benchmark material-location combinations. | Global_Network should cover the full expanded benchmark scope. | Backfill the missing benchmark combinations in Global_Network. |
| ISSUE-010 | XREF-ML-EXTRA | ERROR | Global_Network | material/location | 3639 combo(s); sample 80684724-A668, 80684724-A680, 80684724-A715 | Global_Network contains material-location combinations outside the expanded benchmark scope. | Global_Network should align to the expanded benchmark scope. | Remove or reconcile out-of-scope material-location rows. |
| ISSUE-011 | XREF-ML-MISSING | ERROR | M1_InitialInventory | material/location | 28 combo(s); sample 80728859-0386, 80728861-0386, 80799571-0386 | InitialInventory is missing expanded benchmark material-location combinations. | M1_InitialInventory should cover the full expanded benchmark scope. | Backfill the missing benchmark combinations in M1_InitialInventory. |
| ISSUE-012 | XREF-ML-EXTRA | ERROR | M1_InitialInventory | material/location | 36 combo(s); sample 80728859-386, 80728861-386, 80799571-386 | InitialInventory contains material-location combinations outside the expanded benchmark scope. | M1_InitialInventory should align to the expanded benchmark scope. | Remove or reconcile out-of-scope rows. |
| ISSUE-013 | XREF-ML-MISSING | ERROR | M1_ForecastError | material/location | 13 combo(s); sample 80858047-C867, 80858048-C867, 80858048-D608 | ForecastError is missing expanded benchmark material-location combinations. | M1_ForecastError should cover the expanded benchmark scope defined for this validation. | Backfill the missing benchmark combinations in M1_ForecastError. |
| ISSUE-014 | XREF-ML-EXTRA | ERROR | M1_ForecastError | material/location | 112 combo(s); sample 80728859-D901, 80728859-E546, 80728861-D901 | ForecastError contains material-location combinations outside the expanded benchmark scope. | M1_ForecastError should align to the expanded benchmark scope defined for this validation. | Remove or reconcile out-of-scope rows. |
| ISSUE-015 | XREF-ML-MISSING | ERROR | M3_SafetyStock | material/location | 31 combo(s); sample 80799573-E568, 80799574-C563, 80799575-E556 | SafetyStock is missing expanded benchmark material-location combinations. | M3_SafetyStock should cover the full expanded benchmark scope. | Backfill the missing benchmark combinations in M3_SafetyStock. |
| ISSUE-016 | XREF-ML-EXTRA | ERROR | M3_SafetyStock | material/location | 2681 combo(s); sample 80684724-A668, 80684724-A672, 80684724-A673 | SafetyStock contains material-location combinations outside the expanded benchmark scope. | M3_SafetyStock should align to the expanded benchmark scope. | Remove or reconcile out-of-scope rows. |
| ISSUE-017 | XREF-ML-MISSING | ERROR | M5_DeployConfig | material/receiving | 36 combo(s); sample 80728859-0386, 80728861-0386, 80799571-0386 | DeployConfig is missing expanded benchmark material-location combinations. | M5_DeployConfig receiving scope should cover the full expanded benchmark scope. | Backfill the missing benchmark combinations in M5_DeployConfig. |
| ISSUE-018 | XREF-ML-EXTRA | ERROR | M5_DeployConfig | material/receiving | 3639 combo(s); sample 80684724-A668, 80684724-A680, 80684724-A715 | DeployConfig contains material-location combinations outside the expanded benchmark scope. | M5_DeployConfig should align to the expanded benchmark scope. | Remove or reconcile out-of-scope rows. |
| ISSUE-019 | XREF-MATERIAL | ERROR | Global_Network | material | 324 material(s); sample 80684724, 80684725, 80684726 | Tab contains materials outside the expanded benchmark material set. | Maintained materials should align to the benchmark material set. | Remove or reconcile out-of-scope materials. |
| ISSUE-020 | XREF-MATERIAL | ERROR | M3_SafetyStock | material | 285 material(s); sample 80684724, 80684725, 80684726 | Tab contains materials outside the expanded benchmark material set. | Maintained materials should align to the benchmark material set. | Remove or reconcile out-of-scope materials. |
| ISSUE-021 | XREF-MATERIAL | ERROR | M4_MaterialLocationLineCfg | material | 1 material(s); sample 90464033 | Tab contains materials outside the expanded benchmark material set. | Maintained materials should align to the benchmark material set. | Remove or reconcile out-of-scope materials. |
| ISSUE-022 | XREF-MATERIAL | ERROR | M5_DeployConfig | material | 324 material(s); sample 80684724, 80684725, 80684726 | Tab contains materials outside the expanded benchmark material set. | Maintained materials should align to the benchmark material set. | Remove or reconcile out-of-scope materials. |
| ISSUE-023 | XREF-MATERIAL | ERROR | M6_MaterialMD | material | 274 material(s); sample 80684724, 80684725, 80684726 | Tab contains materials outside the expanded benchmark material set. | Maintained materials should align to the benchmark material set. | Remove or reconcile out-of-scope materials. |
| ISSUE-024 | XREF-LANE | ERROR | M6_TruckReleaseCon | sending/receiving | 16 lane(s); sample 2799->C816, A672->C816, A672->D352 | Tab contains sending-receiving combinations outside Global_Network. | Sending-receiving combinations should exist in Global_Network sourcing-location. | Remove or reconcile out-of-scope lanes. |
| ISSUE-025 | XREF-LOCATION | ERROR | M3_SafetyStock | location | 1 location(s); sample 0386 | Location reference does not exist in Global_Network. | Locations should exist in Global_Network. | Remove or reconcile out-of-network locations. |
| ISSUE-026 | XREF-LOCATION | ERROR | M4_MaterialLocationLineCfg | location | 1 location(s); sample 386 | Location reference does not exist in Global_Network. | Locations should exist in Global_Network. | Remove or reconcile out-of-network locations. |
| ISSUE-027 | XREF-LOCATION | ERROR | M4_LineCapacity | location | 1 location(s); sample 386 | Location reference does not exist in Global_Network. | Locations should exist in Global_Network. | Remove or reconcile out-of-network locations. |
| ISSUE-028 | XREF-LOCATION | ERROR | M4_ProductionReliability | location | 1 location(s); sample 386 | Location reference does not exist in Global_Network. | Locations should exist in Global_Network. | Remove or reconcile out-of-network locations. |
| ISSUE-029 | XREF-LOCATION | ERROR | M5_PushPullModel | sending | 1 location(s); sample 0386 | Location reference does not exist in Global_Network. | Locations should exist in Global_Network. | Remove or reconcile out-of-network locations. |
| ISSUE-030 | HC-GLT-001 | ERROR | Global_LeadTime | OTD/PDT | 52 row(s); sample C810->D767, A673->C819, 0386->A716 | OTD/PDT validity rule failed. | 0 <= OTD < PDT. | Fix rows where OTD/PDT is blank, negative, or not ordered correctly. |
| ISSUE-031 | HC-GDP-001 | ERROR | Global_DemandPriority | demand_element coverage | 10 missing pattern(s); sample net demand for net demand for net demand for normal, net demand for net demand for net demand for net demand for normal, net demand for net demand for net demand for ao | Demand element coverage does not span the maximum network depth. | Demand elements should cover all required demand types through the maximum network layer. | Add the missing demand-element rows. |
| ISSUE-032 | HC-M1DF-001 | ERROR | M1_DemandForecast | week | min=6, max=18 | Forecast week coverage is incomplete for the simulation period plus 2 weeks. | week should start at 1 and end at least at 15. | Rebase or regenerate forecast weeks to the required horizon. |
| ISSUE-033 | HC-M1OC-001 | ERROR | M1_OrderCalendar | date | 90 missing date(s); sample 2026-02-02, 2026-02-03, 2026-02-04 | Order calendar does not cover the full simulation period. | Dates should cover every day from 2026-02-02 to 2026-05-03. | Backfill missing calendar dates for the simulation period. |
| ISSUE-034 | HC-M3SS-001 | ERROR | M3_SafetyStock | material/location daily coverage | 3061 combo(s); sample 80684724-A668, 80684724-A672, 80684724-A673 | SafetyStock does not cover every simulation date for some material-location combinations. | Every material-location should have daily safety-stock records for the full simulation period. | Backfill missing daily safety-stock rows. |
| ISSUE-035 | HC-M4MLL-002 | ERROR | M4_MaterialLocationLineCfg | numeric fields | 30 row(s); sample 80878361-386, 80878362-386, 80878369-386 | One or more M4 numeric fields are non-numeric or blank. | prd_rate, min_batch, rv, ptf, lsk, day, and MCT must be numeric. | Fix blank or non-numeric M4 numeric fields. |
| ISSUE-036 | HC-M4LC-001 | ERROR | M4_LineCapacity | location/line date coverage | 1 combo(s); sample 386-HPFD | LineCapacity does not cover the full simulation period for some maintained lines. | Each location-line should cover every day from 2026-02-02 to 2026-05-03. | Backfill missing line-capacity dates for affected lines. |
| ISSUE-037 | HC-M4CO-002 | ERROR | M4_ChangeoverMatrix | line pair completeness | 435 missing pair(s); sample HPFD:80728859->80799571, HPFD:80728859->80799573, HPFD:80728859->80799574 | ChangeoverMatrix is incomplete for at least one line. | Each line with n delegated materials should have n^2 - n non-self pairs. | Add the missing non-self changeover pairs. |
| ISSUE-038 | HC-M5PP-001 | ERROR | M5_PushPullModel | model | 1 invalid value(s); sample pull | PushPullModel contains invalid values. | model can only be push or soft push. | Replace invalid model values with push or soft push. |
| ISSUE-039 | SC-M1INV-DFCNA | WARNING | M1_InitialInventory | DFC denominator | 419 row(s); sample 80728859-386, 80728859-A668, 80728859-A672 | Week 1 daily average demand is zero or missing, so DFC cannot be evaluated for part of the inventory scope. | Week 1 demand should exist when DFC is used as a soft-risk signal. | Add or rebase week 1 forecast rows, or confirm that these rows should be excluded from DFC review. |
| ISSUE-040 | SC-M4MLL-DEMANDNA | WARNING | M4_MaterialLocationLineCfg | min_batch vs demand availability | 30 row(s); sample 80878361-386, 80878362-386, 80878369-386 | Near-term demand is zero or missing for maintained M4 rows, so min_batch risk cannot be fully assessed. | Week 1 to week 4 demand should exist when evaluating batch-size risk. | Confirm whether these materials are intentionally inactive in weeks 1 to 4 or add the required demand rows. |

## 3. Tab-level summary

| Tab | Status | ERROR count | WARNING count | Main issue type | Remark |
| --- | --- | --- | --- | --- | --- |
| Global_Network | FAILED | 3 | 0 | XREF-ML-MISSING / XREF-ML-EXTRA / XREF-MATERIAL | Present in workbook; blocking issues were found. |
| Global_LeadTime | FAILED | 1 | 0 | HC-GLT-001 | Present in workbook; blocking issues were found. |
| Global_DemandPriority | FAILED | 1 | 0 | HC-GDP-001 | Present in workbook; blocking issues were found. |
| M1_DemandForecast | FAILED | 2 | 0 | SCHEMA-NUMFMT / HC-M1DF-001 | Present in workbook; blocking issues were found. |
| M1_OrderCalendar | FAILED | 1 | 0 | HC-M1OC-001 | Present in workbook; blocking issues were found. |
| M1_InitialInventory | FAILED | 2 | 1 | XREF-ML-MISSING / XREF-ML-EXTRA / SC-M1INV-DFCNA | Present in workbook; blocking issues were found. |
| M1_ForecastError | FAILED | 2 | 0 | XREF-ML-MISSING / XREF-ML-EXTRA | Present in workbook; blocking issues were found. |
| M3_SafetyStock | FAILED | 5 | 0 | XREF-ML-MISSING / XREF-ML-EXTRA / XREF-MATERIAL / XREF-LOCATION / HC-M3SS-001 | Present in workbook; blocking issues were found. |
| M5_PushPullModel | FAILED | 2 | 0 | XREF-LOCATION / HC-M5PP-001 | Present in workbook; blocking issues were found. |
| M5_DeployConfig | FAILED | 3 | 0 | XREF-ML-MISSING / XREF-ML-EXTRA / XREF-MATERIAL | Present in workbook; blocking issues were found. |
| M6_MaterialMD | FAILED | 1 | 0 | XREF-MATERIAL | Present in workbook; blocking issues were found. |
| M6_TruckReleaseCon | FAILED | 1 | 0 | XREF-LANE | Present in workbook; blocking issues were found. |
| M6_TruckTypeSpecs | PASS | 0 | 0 | None | Present in workbook and checked against the configured rules. |
| M6_DeliveryDelayDistribution | PASS | 0 | 0 | None | Present in workbook and checked against the configured rules. |
| M1_AOConfig | PASS | 0 | 0 | None | Present in workbook and checked against the configured rules. |
| M1_DPSConfig | NOT MAINTAINED | 0 | 0 | Not maintained | Tab is absent or present but empty, so it was not treated as maintained for this validation pass. |
| M1_SupplyChoiceConfig | NOT MAINTAINED | 0 | 0 | Not maintained | Tab is absent or present but empty, so it was not treated as maintained for this validation pass. |
| M4_MaterialLocationLineCfg | FAILED | 8 | 1 | SCHEMA-EMPTYCOL / XREF-MATERIAL / XREF-LOCATION / HC-M4MLL-002 / SC-M4MLL-DEMANDNA | Present in workbook; blocking issues were found. |
| M4_LineCapacity | FAILED | 2 | 0 | XREF-LOCATION / HC-M4LC-001 | Present in workbook; blocking issues were found. |
| M4_ProductionReliability | FAILED | 1 | 0 | XREF-LOCATION | Present in workbook; blocking issues were found. |
| M4_ChangeoverDefinition | FAILED | 2 | 0 | SCHEMA-EMPTYCOL | Present in workbook; blocking issues were found. |
| M4_ChangeoverMatrix | FAILED | 1 | 0 | HC-M4CO-002 | Present in workbook; blocking issues were found. |
| Global_seed | PASS | 0 | 0 | None | Present in workbook and checked against the configured rules. |
| Global_SpaceCapacity | NOT MAINTAINED | 0 | 0 | Not maintained | Tab is absent or present but empty, so it was not treated as maintained for this validation pass. |
| M6_MDQBypassRules | PASS | 0 | 0 | None | Present in workbook and checked against the configured rules. |
| M6_TruckCapacityPlan | NOT MAINTAINED | 0 | 0 | Not maintained | Tab is absent or present but empty, so it was not treated as maintained for this validation pass. |

## 4. Readiness conclusion

| Decision | Condition | Conclusion text |
| --- | --- | --- |
| FAILED | Any ERROR exists | Configuration is not ready for simulation. Blocking issues must be fixed before running ChainSight engine. The current blockers are mainly expanded benchmark-scope mismatches, forecast week and date-coverage gaps, incomplete safety-stock / M4 coverage, and invalid lead-time or push-pull settings. |
| PASS WITH WARNINGS | No ERROR exists, but at least one WARNING exists | Configuration can be used for simulation, but warning risks should be reviewed. |
| PASS | No ERROR and no WARNING | Configuration is ready for simulation. |

## 1. Validation summary

| Field | Value |
| --- | --- |
| Validation status | PASS WITH WARNINGS |
| Simulation readiness | Ready with risk |
| Configuration file | workspace/line-myc-network-supply-choice-whatif-202605/scenarios/wf1-produce-hs-order-5050-current-ss/config/wf1-produce-hs-order-5050-current-ss.xlsx |
| Simulation period | 2026-05-04 to 2026-06-19 |
| Scope benchmark | Global_Network material-location combinations |
| Total ERROR count | 0 |
| Total WARNING count | 5 |

## 2. Issue detail

| Issue ID | Rule ID | Severity | Tab | Field / Scope | Key value | Issue | Expected | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WARN-001 | HC-M4CO-002 | WARNING | M4_ChangeoverMatrix | self pairs | 39 self-pair rows across HPSCPACK, HPSMPACK, HPSYPACK | Changeover matrix still contains self-pair rows. The missing non-self HPSMPACK pairs were appended, with unknown `changeover_id` and `validation_flag` left blank per user instruction. | The skill's completeness rule counts only non-self changeover pairs. | Remove self-pair rows if unnecessary, and fill the blank `changeover_id` / `validation_flag` cells later if the simulator requires them. |
| WARN-002 | SC-M1INV-DFCLOW | WARNING | M1_InitialInventory | DFC vs week 1 daily average demand | 38 rows; sample `80799343-E354` DFC `0.00`, `80857025-E354` DFC `0.16` | Initial inventory DFC is below the low threshold for part of the network. | DFC should be at least 15 days unless lower opening coverage is explicitly accepted. | Review low-DFC rows and confirm whether they are acceptable for the scenario. |
| WARN-003 | SC-M1INV-DFCHIGH | WARNING | M1_InitialInventory | DFC vs week 1 daily average demand | 132 rows; sample `80793243-C816` DFC `10122.00`, `80857021-E361` DFC `2383.63` | Initial inventory DFC is above the high threshold for part of the network. | DFC should stay below 50 days unless excess opening stock is intentional. | Review high-DFC rows and confirm whether the excess opening coverage is acceptable. |
| WARN-004 | SC-M1INV-DFCNA | WARNING | M1_InitialInventory | DFC denominator | 295 rows; sample `80768594-A668`, `80781776-A668` | Week 1 daily average demand is zero or missing for many inventory rows, so DFC cannot be evaluated for them. | Week 1 demand should exist when DFC is used as a soft-risk signal. | Confirm whether zero-demand items should be excluded from DFC review or add the required forecast rows. |
| WARN-005 | SC-M4MLL-DEMANDNA | WARNING | M4_MaterialLocationLineCfg | min_batch vs demand availability | 39 rows, all maintained M4 line-config rows | Near-term demand is zero or missing for all maintained M4 line-config rows, so min_batch risk cannot be fully assessed. | Week 1 to week 4 demand should exist when evaluating batch-size risk. | Confirm whether these materials are intentionally inactive in weeks 1 to 4 or add the required demand rows before using min_batch risk as a decision input. |

## 3. Tab-level summary

| Tab | Status | ERROR count | WARNING count | Main issue type | Remark |
| --- | --- | --- | --- | --- | --- |
| Global_Network | PASS | 0 | 0 | None | Present and used as benchmark scope. |
| Global_LeadTime | PASS | 0 | 0 | None | Present and basic hard checks passed. |
| Global_DemandPriority | PASS | 0 | 0 | None | Updated to the user-specified 27-row priority mapping and now covers the required network depth. |
| M1_DemandForecast | PASS | 0 | 0 | None | Out-of-network rows removed; table now aligns to Global_Network benchmark scope. |
| M1_OrderCalendar | PASS | 0 | 0 | None | Covers the simulation period. |
| M1_InitialInventory | WARNING | 0 | 3 | Soft risk | Out-of-network rows removed; only DFC soft-risk warnings remain. |
| M1_ForecastError | PASS | 0 | 0 | None | Present and basic hard checks passed. |
| M3_SafetyStock | PASS | 0 | 0 | None | Out-of-network rows removed and full daily simulation-period coverage backfilled with zero quantity where missing. |
| M5_PushPullModel | PASS | 0 | 0 | None | Present and model values are valid. |
| M5_DeployConfig | PASS | 0 | 0 | None | Rebuilt to match Global_Network `material-sourcing-location` scope exactly; all lanes now exist in Global_LeadTime. |
| M6_MaterialMD | PASS | 0 | 0 | Accepted exception | User accepted that AO and SupplyChoice can reference materials not present in M6_MaterialMD for this workbook review. |
| M6_TruckReleaseCon | PASS | 0 | 0 | None | The 4 missing Global_Network lanes were backfilled. Extra scenario lanes were retained per user instruction and are not treated as blockers in this validation pass. |
| M6_TruckTypeSpecs | PASS | 0 | 0 | None | Present and all referenced truck types exist. |
| M6_DeliveryDelayDistribution | PASS | 0 | 0 | None | Present; ALL-ALL global rule treated as valid override rather than lane error. |
| M1_AOConfig | PASS | 0 | 0 | Accepted exception | User accepted missing M6_MaterialMD coverage for AOConfig in this pass. |
| M1_DPSConfig | PASS | 0 | 0 | Conditional tab present | Present but empty; no hard-rule violation detected. |
| M1_SupplyChoiceConfig | PASS | 0 | 0 | Accepted exception | User accepted missing M6_MaterialMD coverage for SupplyChoiceConfig in this pass. |
| M4_MaterialLocationLineCfg | WARNING | 0 | 1 | Soft risk data gap | min_batch soft-risk check cannot be completed because weeks 1 to 4 demand is missing or zero for all maintained rows. |
| M4_LineCapacity | PASS | 0 | 0 | Conditional tab present | Present and period coverage checks passed. |
| M4_ProductionReliability | PASS | 0 | 0 | Conditional tab present | Present and consistent with maintained lines. |
| M4_ChangeoverDefinition | PASS | 0 | 0 | Conditional tab present | Present and all matrix changeover_id values are defined. |
| M4_ChangeoverMatrix | WARNING | 0 | 1 | Completeness / placeholder values | The 34 missing HPSMPACK non-self pairs were appended. Workbook still contains 39 self-pair rows, and the newly appended rows keep `changeover_id` and `validation_flag` blank per user instruction. |
| Global_seed | PASS | 0 | 0 | Optional tab present | Present as optional tab. |
| Global_SpaceCapacity | PASS | 0 | 0 | Optional tab present | Present but empty, with no blocking issue for this scenario. |
| M6_MDQBypassRules | PASS | 0 | 0 | Optional tab present | Present; ALL-ALL global rule treated as valid override rather than lane error. |
| M6_TruckCapacityPlan | PASS | 0 | 0 | Optional tab present | Present but empty, with no blocking issue for this scenario. |

## 4. Readiness conclusion

| Decision | Condition | Conclusion text |
| --- | --- | --- |
| FAILED | Any ERROR exists | Configuration is not ready for simulation. Blocking issues must be fixed before running ChainSight engine. |
| PASS WITH WARNINGS | No ERROR exists, but at least one WARNING exists | Configuration can be used for simulation, but warning risks should be reviewed. This wf1 package is now ready with risk: no blocking structural errors remain; M6 missing lanes have been backfilled, extra scenario-only M6 lanes were retained per user instruction, and the remaining review points are self-pair rows, blank changeover fields, and DFC soft-risk warnings. |
| PASS | No ERROR and no WARNING | Configuration is ready for simulation. |